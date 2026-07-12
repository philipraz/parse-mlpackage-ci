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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["convert", "measure-native", "measure-ort"])
    cmd = ap.parse_args().cmd
    {"convert": convert, "measure-native": measure_native, "measure-ort": measure_ort}[cmd]()
