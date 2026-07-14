"""Measure whether a native Core ML .mlpackage encoder is materially smaller
resident than onnxruntime's CoreML EP (the ~4.5GB / #6 question), on a macOS CI
runner (Apple Silicon = ANE, closest proxy to the iPhone).

Subcommands (each run in its own process so RSS is isolated):
  convert        PyTorch encoder -> encoder_fp16.mlpackage AND encoder_fp16.onnx,
                 plus ref_hidden.npy (PyTorch output) + input_pv.npy for A/B checks.
  measure-native load the .mlpackage via Core ML, predict, report cosine vs ref + RSS.
  measure-ort    load the ONNX via ORT CoreMLExecutionProvider, predict, report RSS.

The A/B on one machine is the answer: if native RSS << ORT RSS, ORT's dual
retention (ONNX graph + CoreML copy) is the bloat and the native path is worth it.
"""
import argparse
import os
import resource
import sys
import time

import numpy as np

MODEL = "nvidia/NVIDIA-Nemotron-Parse-v1.1"
H, W = 768, 624
MLPKG = "encoder_fp16.mlpackage"
ONNX_FP32 = "encoder_fp32.onnx"
ONNX_FP16 = "encoder_fp16.onnx"
REF = "ref_hidden.npy"
INP = "input_pv.npy"


def rss_mb():
    """Current RSS (psutil) + peak (ru_maxrss; bytes on macOS), in MB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # macOS: bytes
    try:
        import psutil
        cur = psutil.Process().memory_info().rss / 1e6
    except Exception:
        cur = float("nan")
    return cur, peak


def fixed_input():
    rng = np.random.default_rng(0)
    return (rng.standard_normal((1, 3, H, W)).astype(np.float32) * 0.5)


def cosine(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class EncWrap:
    """Lazy: only imports torch/transformers when convert runs."""

    @staticmethod
    def build():
        import torch
        from transformers import AutoModel

        class _W(torch.nn.Module):
            def __init__(self, e):
                super().__init__()
                self.e = e

            def forward(self, pv):
                o = self.e(pv)
                return o[0] if isinstance(o, (tuple, list)) else getattr(o, "last_hidden_state", o)

        model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).float().eval()
        return _W(model.encoder).eval()


def convert():
    import torch
    import coremltools as ct

    pv_np = fixed_input()
    pv = torch.from_numpy(pv_np)
    wrap = EncWrap.build()
    with torch.no_grad():
        wrap(pv)  # warmup: RADIO materializes resolution buffers
        ref = wrap(pv).cpu().numpy()
        traced = torch.jit.trace(wrap, pv)
    np.save(INP, pv_np)
    np.save(REF, ref)
    print(f"ref hidden shape {ref.shape}", flush=True)

    # --- native Core ML .mlpackage (fp16 MLProgram) ---
    t0 = time.time()
    # Match the call that reached 985/985 ops locally: don't name the output
    # (measure-native reads it positionally), avoids a rename-mismatch risk.
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixel_values", shape=(1, 3, H, W))],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    ml.save(MLPKG)
    sz = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(MLPKG) for f in fs)
    print(f"MLPACKAGE saved {sz/1e6:.0f} MB in {time.time()-t0:.0f}s", flush=True)

    # --- ONNX fp16 for the ORT A/B (same proven path as the app's stage12) ---
    torch.onnx.export(
        wrap, (pv,), ONNX_FP32,
        input_names=["pixel_values"], output_names=["hidden"],
        opset_version=17, do_constant_folding=True,
    )
    import onnx
    from onnxruntime.transformers.onnx_model import OnnxModel
    m = OnnxModel(onnx.load(ONNX_FP32))
    m.convert_float_to_float16(keep_io_types=True)
    m.save_model_to_file(ONNX_FP16, use_external_data_format=False)
    print(f"ONNX fp16 saved {os.path.getsize(ONNX_FP16)/1e6:.0f} MB", flush=True)


def measure_native():
    import coremltools as ct

    pv = np.load(INP)
    ref = np.load(REF)
    cur0, _ = rss_mb()
    ml = ct.models.MLModel(MLPKG, compute_units=ct.ComputeUnit.ALL)
    out = ml.predict({"pixel_values": pv})
    hidden = list(out.values())[0]
    cur, peak = rss_mb()
    print(f"NATIVE_COREML  cosine={cosine(ref, np.asarray(hidden)):.6f}  "
          f"rss={cur:.0f}MB  peak={peak:.0f}MB  (load delta {cur-cur0:.0f}MB)", flush=True)


def measure_ort():
    import onnxruntime as ort

    pv = np.load(INP)
    ref = np.load(REF)
    cur0, _ = rss_mb()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    sess = ort.InferenceSession(ONNX_FP16, so, providers=providers)
    hidden = sess.run(None, {"pixel_values": pv})[0]
    cur, peak = rss_mb()
    print(f"ORT_COREML_EP  providers={providers}  cosine={cosine(ref, hidden):.6f}  "
          f"rss={cur:.0f}MB  peak={peak:.0f}MB  (load delta {cur-cur0:.0f}MB)", flush=True)


# ── Decoded-output A/B: the REAL gate (cosine can hide argmax-flipping errors) ──
from PIL import Image  # noqa: E402

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown>"
TEST_IMAGES = ["test_lease.png", "test_dense.png"]
EOS = 2


def letterbox_pv(img):
    s = min(W / img.width, H / img.height)
    rw, rh = round(img.width * s), round(img.height * s)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(img.resize((rw, rh), Image.BILINEAR), (0, 0))
    arr = (np.asarray(canvas).astype(np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


def tiles_2x2(img, overlap=0.08):
    tw, th = img.width / 2, img.height / 2
    ox, oy = tw * overlap, th * overlap
    out = []
    for r in range(2):
        for c in range(2):
            out.append(img.crop((max(0, round(c * tw - ox)), max(0, round(r * th - oy)),
                                 min(img.width, round((c + 1) * tw + ox)), min(img.height, round((r + 1) * th + oy)))))
    return out


def degeneracy(text):
    w = text.split()
    if len(w) < 20:
        return 0.0
    from collections import Counter
    grams = [" ".join(w[i:i + 6]) for i in range(len(w) - 6)]
    _, n = Counter(grams).most_common(1)[0]
    return round(n * 6 / max(len(w), 1), 2)


def cer(a, b):
    a2 = "".join(c for c in a.lower() if c.isalnum())
    b2 = "".join(c for c in b.lower() if c.isalnum())
    if not a2 and not b2:
        return 0.0
    # cheap edit-distance ratio
    import difflib
    return round(1 - difflib.SequenceMatcher(None, a2, b2).ratio(), 3)


def strip_markup(text):
    """Text-only: drop coordinate tokens + class/format tags so CER measures
    RECOGNITION divergence, not coordinate jitter (a tiny box change = many chars)."""
    import re
    t = re.sub(r"<x_[0-9.]+>|<y_[0-9.]+>|<class_[A-Za-z-]+>|<br>|</?sup>", " ", text)
    t = re.sub(r"\\begin\{tabular\}\{[^}]*\}|\\end\{tabular\}|\\\\|&|#", " ", t)
    return " ".join(t.split())


def decode_ab():
    import torch
    import coremltools as ct
    import onnxruntime as ort
    from transformers import AutoModel, AutoProcessor
    from transformers.modeling_outputs import BaseModelOutput

    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).float().eval()
    prompt_ids = proc(images=[Image.new("RGB", (W, H), (255, 255, 255))], text=PROMPT,
                      return_tensors="pt", add_special_tokens=False)["input_ids"]
    mlmodel = ct.models.MLModel(MLPKG, compute_units=ct.ComputeUnit.ALL)
    so = ort.SessionOptions(); so.log_severity_level = 3
    provs = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    ort_enc = ort.InferenceSession(ONNX_FP16, so, providers=provs)

    def native_hidden(pv):
        return np.asarray(list(mlmodel.predict({"pixel_values": pv}).values())[0], np.float32)

    def ort_hidden(pv):
        return ort_enc.run(None, {"pixel_values": pv})[0].astype(np.float32)

    def decode(hidden, penalty=1.3, no_repeat=3, max_new=256):
        try:
            enc = BaseModelOutput(last_hidden_state=torch.from_numpy(np.ascontiguousarray(hidden)))
            with torch.no_grad():
                out = model.generate(encoder_outputs=enc, decoder_input_ids=prompt_ids,
                                     decoder_attention_mask=torch.ones_like(prompt_ids),
                                     max_new_tokens=max_new, do_sample=False, num_beams=1,
                                     repetition_penalty=penalty, no_repeat_ngram_size=no_repeat,
                                     eos_token_id=EOS, pad_token_id=EOS, use_cache=True)
            gen = [int(x) for x in out[0][prompt_ids.shape[1]:]]
            text = proc.batch_decode([gen], skip_special_tokens=True)[0]
            # Distinguish the stop states — empty and cap are NOT "clean".
            if len(gen) == 0 or (len(gen) == 1 and gen[0] == EOS):
                st = "empty"
            elif EOS in gen:
                st = "eos"
            elif len(gen) >= max_new:
                st = "cap"
            else:
                st = "stop"
            return text, len(gen), st
        except Exception as e:
            return f"<decode error: {e}>", 0, "error"

    from collections import Counter
    print("\n===== DECODED-OUTPUT A/B (native vs ORT encoder, SAME input + SAME decoder) =====")
    for name in TEST_IMAGES:
        if not os.path.exists(name):
            print(f"  (skip {name} — not in repo)", flush=True)
            continue
        img = Image.open(name).convert("RGB")
        pv = letterbox_pv(img)
        nt, nn, ns = decode(native_hidden(pv))
        ot, on, ostate = decode(ort_hidden(pv))
        print(f"--- {name} ({img.width}x{img.height}) FULL PAGE ---")
        print(f"  NATIVE: {nn} tok stop={ns} deg={degeneracy(nt)} | {nt[:140]!r}")
        print(f"  ORT   : {on} tok stop={ostate} deg={degeneracy(ot)} | {ot[:140]!r}")
        print(f"  CER  raw={cer(nt, ot)}  text-only={cer(strip_markup(nt), strip_markup(ot))}")
        states = []
        for i, tile in enumerate(tiles_2x2(img)):
            tt, tn, ts = decode(native_hidden(letterbox_pv(tile)))
            states.append(ts)
            print(f"  NATIVE tile {i+1}/4: {tn} tok stop={ts} deg={degeneracy(tt)} | {tt[:70]!r}")
        print(f"  NATIVE 2x2 stop-states: {dict(Counter(states))}  (only eos-with-content is a genuine parse)\n")


def precision_ab():
    """The KEY diagnostic (do this on ENCODER outputs before the decoder): is
    native's degradation from ANE precision or from conversion? Feed the IDENTICAL
    saved tensor to PyTorch/ORT/Core ML — no independent preprocessing — and report
    the per-token distribution, not just a global cosine that hides worst tokens."""
    import coremltools as ct
    import onnxruntime as ort

    pv = np.load(INP)
    ref = np.load(REF).astype(np.float32)  # PyTorch fp32 [1, T, D]

    def report(label, h):
        h = np.asarray(h, np.float32)
        if h.shape != ref.shape:
            print(f"  {label:18}: SHAPE {h.shape} != ref {ref.shape}", flush=True); return
        r2 = ref.reshape(ref.shape[1], -1)
        h2 = h.reshape(h.shape[1], -1)
        den = np.linalg.norm(r2, axis=1) * np.linalg.norm(h2, axis=1) + 1e-9
        pt = (r2 * h2).sum(1) / den
        print(f"  {label:18}: cos={cosine(ref, h):.5f} "
              f"tok[min={pt.min():.4f} mean={pt.mean():.4f} p5={np.percentile(pt, 5):.4f} <0.9={int((pt < 0.9).sum())}/{len(pt)}] "
              f"MAE={np.abs(ref - h).mean():.4f} maxAbs={np.abs(ref - h).max():.3f} "
              f"NaN={int(np.isnan(h).sum())} Inf={int(np.isinf(h).sum())}", flush=True)

    print("\n===== ENCODER PRECISION A/B (vs PyTorch fp32, IDENTICAL input) =====")
    for label, cu in [("coreml-cpuOnly", ct.ComputeUnit.CPU_ONLY),
                      ("coreml-cpuAndGPU", ct.ComputeUnit.CPU_AND_GPU),
                      ("coreml-all(ANE)", ct.ComputeUnit.ALL)]:
        try:
            m = ct.models.MLModel(MLPKG, compute_units=cu)
            report(label, list(m.predict({"pixel_values": pv}).values())[0])
        except Exception as e:
            print(f"  {label:18}: ERROR {e}", flush=True)
    so = ort.SessionOptions(); so.log_severity_level = 3
    provs = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    try:
        report("ort-" + provs[0].replace("ExecutionProvider", ""),
               ort.InferenceSession(ONNX_FP16, so, providers=provs).run(None, {"pixel_values": pv})[0])
    except Exception as e:
        print(f"  ort: ERROR {e}", flush=True)
    try:
        report("ort-cpu", ort.InferenceSession(ONNX_FP16, so, providers=["CPUExecutionProvider"]).run(None, {"pixel_values": pv})[0])
    except Exception as e:
        print(f"  ort-cpu: ERROR {e}", flush=True)
    print("  → cpuOnly≈ort but all≠ ⇒ ANE precision;  all coreml modes ≈each other but ≠ort ⇒ conversion;")
    print("    worst-token ≪ mean ⇒ a few catastrophic tokens that flip greedy decode.")


def precision_fix():
    """Is the fp16 conversion gap fixable? Convert two variants and measure vs
    PyTorch: FLOAT32 (the faithfulness ceiling) and selective-fp16 that keeps the
    fp16-unstable ops (norms/softmax/reductions) in fp32 — what ORT does, which is
    why ORT is 0.997. If fp16-safe ≈ 0.99+, native is fixable AND stays small."""
    import torch
    import coremltools as ct

    pv_np = np.load(INP)
    ref = np.load(REF).astype(np.float32)
    pv = torch.from_numpy(pv_np)
    wrap = EncWrap.build()
    with torch.no_grad():
        wrap(pv)
        traced = torch.jit.trace(wrap, pv)

    def report(label, mlpath):
        sz = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(mlpath) for f in fs) / 1e6
        m = ct.models.MLModel(mlpath, compute_units=ct.ComputeUnit.CPU_AND_GPU)
        h = np.asarray(list(m.predict({"pixel_values": pv_np}).values())[0], np.float32)
        r2 = ref.reshape(ref.shape[1], -1)
        h2 = h.reshape(h.shape[1], -1)
        den = np.linalg.norm(r2, axis=1) * np.linalg.norm(h2, axis=1) + 1e-9
        pt = (r2 * h2).sum(1) / den
        print(f"  {label:14} ({sz:5.0f}MB): cos={cosine(ref, h):.5f} "
              f"tok[min={pt.min():.4f} <0.9={int((pt < 0.9).sum())}/{len(pt)}] MAE={np.abs(ref - h).mean():.4f}", flush=True)

    common = dict(inputs=[ct.TensorType(name="pixel_values", shape=(1, 3, H, W))],
                  minimum_deployment_target=ct.target.iOS16, convert_to="mlprogram")
    print("\n===== PRECISION-FIX candidates (vs PyTorch fp32, identical input) =====")
    try:
        ct.convert(traced, compute_precision=ct.precision.FLOAT32, **common).save("enc_fp32.mlpackage")
        report("fp32-ceiling", "enc_fp32.mlpackage")
    except Exception as e:
        print(f"  fp32-ceiling: ERROR {e}", flush=True)
    SENSITIVE = {"layer_norm", "batch_norm", "instance_norm", "l2_norm",
                 "reduce_mean", "reduce_sum", "reduce_l2_norm", "reduce_max",
                 "rsqrt", "sqrt", "softmax", "gelu", "erf"}
    try:
        from coremltools.converters.mil.mil.passes.defs.quantization import FP16ComputePrecision
        cp = FP16ComputePrecision(op_selector=lambda op: op.op_type not in SENSITIVE)
        ct.convert(traced, compute_precision=cp, **common).save("enc_fp16_safe.mlpackage")
        report("fp16-safe", "enc_fp16_safe.mlpackage")
    except Exception as e:
        print(f"  fp16-safe: ERROR {e}", flush=True)
    print("  → fp32=1.0 confirmed it's pure precision; fp16-safe≈0.99 = the fix (small + faithful).")


def precision_probe():
    """fp32 is perfect (cos 1.0), yet blanket-fp16 (0.958) AND
    norms/softmax/reductions-in-fp32 (0.962) both fail. So the fp16-unstable op is
    NOT in that family. Localize it in ONE run, two ways:

      (A) by op-TYPE — keep exactly one op_type in fp32 (rest fp16); the type whose
          protection jumps cos toward 1.0 is the culprit family.
      (B) by graph POSITION — keep the back-half / back-quarter of ops in fp32; if
          only the deep tail matters, that's ViT-H residual-stream growth overflowing
          fp16 (activations exceed 65504 in late layers), not a single op type.

    Whichever recovers cos becomes the shipping recipe's fp32 carve-out."""
    import torch
    import coremltools as ct
    from coremltools.converters.mil.mil.passes.defs.quantization import FP16ComputePrecision
    from collections import Counter

    pv_np = np.load(INP)
    ref = np.load(REF).astype(np.float32)
    pv = torch.from_numpy(pv_np)
    wrap = EncWrap.build()
    with torch.no_grad():
        wrap(pv)
        traced = torch.jit.trace(wrap, pv)

    common = dict(inputs=[ct.TensorType(name="pixel_values", shape=(1, 3, H, W))],
                  minimum_deployment_target=ct.target.iOS16, convert_to="mlprogram")

    def measure(mlpath):
        m = ct.models.MLModel(mlpath, compute_units=ct.ComputeUnit.CPU_AND_GPU)
        h = np.asarray(list(m.predict({"pixel_values": pv_np}).values())[0], np.float32)
        return cosine(ref, h)

    def conv_cos(label, op_selector):
        try:
            cp = FP16ComputePrecision(op_selector=op_selector)
            path = f"probe_{label}.mlpackage"
            ct.convert(traced, compute_precision=cp, **common).save(path)
            c = measure(path)
            print(f"    {label:22} cos={c:.5f}", flush=True)
            return (label, c)
        except Exception as e:
            print(f"    {label:22} ERROR {e}", flush=True)
            return (label, -1.0)

    # Op names/types in execution order (structure is precision-independent).
    ordered = []
    try:
        prog = ct.convert(traced, convert_to="milinternal",
                          inputs=[ct.TensorType(name="pixel_values", shape=(1, 3, H, W))],
                          minimum_deployment_target=ct.target.iOS16)
        ordered = [(o.name, o.op_type) for o in prog.functions["main"].operations]
    except Exception as e:
        print(f"  (milinternal introspection failed: {e})", flush=True)
    hist = Counter(t for _, t in ordered)
    if hist:
        print("\n===== OP-TYPE HISTOGRAM (top 30) =====", flush=True)
        for t, c in hist.most_common(30):
            print(f"    {c:5d}  {t}", flush=True)

    results = []
    print("\n===== (A) SINGLE-OP-TYPE fp32 PROBE (keep ONE type fp32, rest fp16) =====", flush=True)
    SUSPECTS = ["add", "matmul", "mul", "linear", "conv", "einsum", "sub",
                "softmax", "gelu", "erf", "layer_norm", "batch_norm",
                "reduce_mean", "real_div", "sqrt", "rsqrt", "scaled_dot_product_attention"]
    cands = [t for t in SUSPECTS if not hist or t in hist]
    for t in cands:
        results.append(conv_cos(f"type={t}", lambda op, _t=t: op.op_type != _t))

    if ordered:
        print("\n===== (B) POSITIONAL fp32 PROBE (keep a contiguous slab fp32) =====", flush=True)
        names = [n for n, _ in ordered]
        N = len(names)
        for label, keep in [
            ("back-50%", set(names[N // 2:])),
            ("back-25%", set(names[3 * N // 4:])),
            ("front-50%", set(names[:N // 2])),
        ]:
            results.append(conv_cos(label, lambda op, _k=keep: op.name not in _k))

    results = [r for r in results if r[1] >= 0]
    results.sort(key=lambda x: -x[1])
    print("\n  RANK (best first):", flush=True)
    for label, c in results:
        print(f"    {c:.5f}  {label}", flush=True)
    print("  → a TYPE at ~0.99 ⇒ add it to SENSITIVE; only POSITION recovers ⇒ deep", flush=True)
    print("    residual overflow ⇒ carve out the back slab (or raise its accum to fp32).", flush=True)


def leak_check(n=12):
    import coremltools as ct
    pv = np.load(INP) if os.path.exists(INP) else fixed_input()
    ml = ct.models.MLModel(MLPKG, compute_units=ct.ComputeUnit.ALL)
    print(f"\n===== LEAK CHECK: {n} sequential predicts on ONE reused instance =====")
    base = None
    for i in range(n):
        ml.predict({"pixel_values": pv})
        cur, peak = rss_mb()
        if base is None:
            base = cur
        print(f"  predict {i+1:2}/{n}: rss={cur:.0f}MB  peak={peak:.0f}MB  delta_from_first={cur-base:+.0f}MB", flush=True)
    print("  → flat delta = no per-run leak; steady climb = retention.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["convert", "measure-native", "measure-ort", "precision-ab", "precision-fix", "precision-probe", "decode-ab", "leak-check"])
    cmd = ap.parse_args().cmd
    {"convert": convert, "measure-native": measure_native, "measure-ort": measure_ort,
     "precision-ab": precision_ab, "precision-fix": precision_fix, "precision-probe": precision_probe,
     "decode-ab": decode_ab, "leak-check": leak_check}[cmd]()
