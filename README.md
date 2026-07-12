# parse-mlpackage-ci

One-off macOS CI job to answer the "**#6**" question for the on-device Nemotron
Parse encoder: is a **native Core ML `.mlpackage`** materially smaller resident
than **onnxruntime's CoreML EP** (which is ~4.5 GB on device — suspected to
retain the ONNX graph alongside its CoreML copy)?

It runs on a GitHub Actions **Apple-Silicon** runner (has the Neural Engine, the
closest proxy to the iPhone), converts the encoder to `.mlpackage`, and measures
the resident memory of **both** load paths on the same machine.

This repo contains **no app code** — it downloads the public model
`nvidia/NVIDIA-Nemotron-Parse-v1.1` from Hugging Face at run time.

## Run it

1. Create a new **private** GitHub repo (e.g. `parse-mlpackage-ci`).
2. Push these three files (`ci_mlpackage.py`, `requirements.txt`,
   `.github/workflows/mlpackage-measure.yml`) to it:
   ```bash
   cd parse-mlpackage-ci
   git init && git add -A && git commit -m "mlpackage measurement CI"
   git branch -M main
   git remote add origin git@github.com:<you>/parse-mlpackage-ci.git
   git push -u origin main
   ```
3. GitHub → the repo → **Actions** tab → **mlpackage-measure** → **Run workflow**.

## What to read in the logs

Two lines decide it — compare the `rss`/`peak` numbers:

```
NATIVE_COREML  cosine=0.99xxxx  rss=####MB  peak=####MB   <- native .mlpackage
ORT_COREML_EP  providers=[...]  cosine=0.99xxxx  rss=####MB  peak=####MB  <- ORT path
```

- **Native << ORT** (e.g. ~1.5 GB vs ~4.5 GB) → ORT's dual retention is the
  bloat; the native Core ML module is worth building (and would lift the
  resolution ceiling). The converted `encoder_fp16.mlpackage` is uploaded as a
  build artifact for reuse.
- **Similar** → the resident cost is intrinsic to Core ML at this size; the
  native path won't help and tiling is the way.

`cosine` should be ~0.99+ on both (validates the conversion is faithful).
