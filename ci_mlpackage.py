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

    def decode(hidden, penalty=1.3, no_repeat=3, max_new=512):
        try:
            enc = BaseModelOutput(last_hidden_state=torch.from_numpy(np.ascontiguousarray(hidden)))
            with torch.no_grad():
                out = model.generate(encoder_outputs=enc, decoder_input_ids=prompt_ids,
                                     max_new_tokens=max_new, do_sample=False, num_beams=1,
                                     repetition_penalty=penalty, no_repeat_ngram_size=no_repeat,
                                     eos_token_id=EOS, pad_token_id=EOS, use_cache=True)
            gen = [int(x) for x in out[0][prompt_ids.shape[1]:]]
            text = proc.batch_decode([gen], skip_special_tokens=True)[0]
            return text, len(gen), (EOS in gen)
        except Exception as e:
            return f"<decode error: {e}>", 0, False

    print("\n===== DECODED-OUTPUT A/B (native Core ML vs ORT encoder, same decoder) =====")
    print("decode: greedy, repetition_penalty=1.3, no_repeat_ngram_size=3, cap 512\n")
    for name in TEST_IMAGES:
        img = Image.open(name).convert("RGB")
        pv = letterbox_pv(img)
        nt, nn, ne = decode(native_hidden(pv))
        ot, on, oe = decode(ort_hidden(pv))
        print(f"--- {name} ({img.width}x{img.height}) FULL PAGE ---")
        print(f"  NATIVE: {nn} tok EOS={ne} deg={degeneracy(nt)} | {nt[:150]!r}")
        print(f"  ORT   : {on} tok EOS={oe} deg={degeneracy(ot)} | {ot[:150]!r}")
        print(f"  native-vs-ort CER={cer(nt, ot)}")
        # native full-page vs native 2x2 tiled (does tiling help the native encoder?)
        clean = 0
        for i, tile in enumerate(tiles_2x2(img)):
            tt, tn, te = decode(native_hidden(letterbox_pv(tile)))
            clean += te and degeneracy(tt) < 0.25
            print(f"  NATIVE tile {i+1}/4: {tn} tok EOS={te} deg={degeneracy(tt)} | {tt[:80]!r}")
        print(f"  NATIVE 2x2: {clean}/4 tiles clean\n")


def leak_check(n=20):
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
    ap.add_argument("cmd", choices=["convert", "measure-native", "measure-ort", "decode-ab", "leak-check"])
    cmd = ap.parse_args().cmd
    {"convert": convert, "measure-native": measure_native, "measure-ort": measure_ort,
     "decode-ab": decode_ab, "leak-check": leak_check}[cmd]()
