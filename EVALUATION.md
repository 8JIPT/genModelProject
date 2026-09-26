# Full variants and checkpoint evaluation

All seven variants use the same SiT model implementation, transport objective,
VAE cache, training loop, and checkpoint loader. Block numbers below are zero
based. Skips connect **source block output to target block input**, concatenate
`[current/deep, source/shallow]`, and use independent `Linear(768, 384)`
projections initialized to `[I, 0]` with zero bias.

| Variant | Linear attention blocks | Skip pairs | Total parameters | Trainable parameters |
|---|---|---|---:|---:|
| baseline | none | none | 32,963,360 | 32,865,056 |
| linear | 8–11 | 2→9, 1→10 | 32,970,016 | 2,372,096 |
| uvit | none | 2→9, 1→10 | 33,553,952 | 590,592 |
| linear_uvit | 8–11 | 2→9, 1→10 | 33,560,608 | 2,962,688 |
| full_linear | 0–11 | none | 32,983,328 | 7,116,288 |
| full_uvit | none | 0→11, 1→10, 2→9, 3→8, 4→7 | 34,439,840 | 1,476,480 |
| full_linear_uvit | 0–11 | 0→11, 1→10, 2→9, 3→8, 4→7 | 34,459,808 | 8,592,768 |

Counts use the local pretrained SiT-S/2 checkpoint's `learn_sigma=True` shape.
Modified variants use `--freeze-backbone`; baseline uses normal full fine-tuning
with its fixed position embeddings still frozen. All attention blocks not
listed use original attention. Blocks 5 and 6 are the central region for full
skips. For other even-depth SiT sizes, full skips pair `i → depth-1-i` while
leaving two central blocks; the old partial variants keep their original maps.
Old state-dict keys and initialization are unchanged. Pretrained loading omits
original attention keys only in replaced blocks and audits all missing keys.

## Train the three new variants

From the repository root, with the existing latent cache:

```bash
conda activate SiT
train_common=(
  --model SiT-S/2 --image-size 256
  --pretrained pretrained_models/SiT-S-2-256.pt
  --latent-path data/imagenet50k_latents --vae ema
  --global-batch-size 1 --grad-accum-steps 8 --precision bf16
  --gradient-checkpointing --freeze-backbone --no-ema
  --learning-rate 1e-4 --global-seed 0 --num-workers 4 --device cuda
  --max-steps 40000 --ckpt-every 10000 --log-every 100 --sample-every 0
)
python train.py "${train_common[@]}" --variant full_linear
python train.py "${train_common[@]}" --variant full_uvit
python train.py "${train_common[@]}" --variant full_linear_uvit
```

The new variants are also accepted by `benchmark_train.py`, `sample.py`, and
`sample_ddp.py`. Resume uses `--ckpt` instead of `--pretrained`, as before.

## Metric protocol

Install the optional dependency once:

```bash
python -m pip install -r requirements-eval.txt
```

The evaluator uses **torch-fidelity 0.4.0**, `inception-v3-compat`, 2048-dimensional
FID features, and `logits_unbiased` for Inception Score. IS uses 10 shuffled
splits by default, with a fixed seed. Inception runs in FP32 with TF32 disabled.
The library computes the metrics; this repository does not implement their
formulas. Library weights download on first use. Metric caches contain library
serialized objects and should be treated as trusted local files.

The reference is the directory explicitly supplied with `--reference`. Images
are converted to RGB and receive the same deterministic ADM center crop as SiT,
with **no flip**. With `data/imagenet50k/train`, the reported FID is
**subset-FID against the 50k training subset**, even when generating 50,000 images.
It measures similarity to that subset and is not equivalent to a published
ImageNet/ADM FID-50K. The evaluator deliberately does not label arbitrary
reference folders as official ImageNet statistics or import incompatible ADM
statistics. For paper-comparable scores, use the original ADM evaluation suite
with its appropriate ImageNet reference and protocol instead.

Reference features/statistics are computed on the first metric evaluation and
cached under `evaluation_cache/` for subsequent checkpoints and future runs.
The key includes the reference inventory (relative paths, sizes, mtimes), crop,
resolution, library version, device, and metric batch size. Retain the cache and
reference directory unchanged to reuse it. Changing `--reference` selects a
different real distribution; use `--reference-name` to identify it in output.
The protocol remains explicitly labeled subset/custom-reference FID.

**Small-sample FID is noisy.** A 1,000-sample quick result cannot be compared
directly to FID-50K. Tiny rank-deficient smoke fixtures may also produce small
negative numerical FID values; values are reported as returned by the library.

Sources: [torch-fidelity](https://github.com/toshas/torch-fidelity),
[ADM evaluation](https://github.com/openai/guided-diffusion/tree/main/evaluations).

## Explicit evaluation commands

Define common inference settings once. These match the existing `sample_ddp.py`
ODE defaults: velocity/Linear transport, Dopri5, 250 requested output points,
CFG 1, EMA VAE, and seed 0. Sampling uses raw **model** weights by default because
the training commands disable EMA. `--weights ema` requires an EMA in every
selected checkpoint and never silently falls back to model weights.

```bash
eval_common=(
  --reference data/imagenet50k/train
  --reference-name imagenet50k-training-subset
  --model SiT-S/2 --image-size 256 --vae ema --weights model
  --device cuda --precision fp32 --batch-size 1 --seed 0
  --mode ODE --sampling-method dopri5 --num-sampling-steps 250
  --cfg-scale 1.0 --warmup-batches 2 --metric-batch-size 16
)
```

Single checkpoint (an actual existing local path), quick 1k evaluation:

```bash
python evaluate.py "${eval_common[@]}" --num-samples 1000 \
  --checkpoint linear 40000 results/002-SiT-S-2-linear-Linear-velocity-None/checkpoints/0040000.pt
```

Selected steps from one experiment, final 50k evaluation per checkpoint:

```bash
python evaluate.py "${eval_common[@]}" --num-samples 50000 \
  --experiment-dir results/002-SiT-S-2-linear-Linear-velocity-None \
  --variants linear --steps 10000 40000
```

Uneven selections across variants, with no other checkpoints evaluated:

```bash
python evaluate.py "${eval_common[@]}" --num-samples 50000 \
  --checkpoint linear 10000 results/002-SiT-S-2-linear-Linear-velocity-None/checkpoints/0010000.pt \
  --checkpoint linear 40000 results/002-SiT-S-2-linear-Linear-velocity-None/checkpoints/0040000.pt \
  --checkpoint linear_uvit 20000 results/001-SiT-S-2-linear_uvit-Linear-velocity-None/checkpoints/0020000.pt
```

Original untouched pretrained baseline, reported as `baseline/pretrained`, step 0:

```bash
python evaluate.py "${eval_common[@]}" --num-samples 50000 \
  --checkpoint baseline 0 pretrained_models/SiT-S-2-256.pt
```

Every explicit entry is `--checkpoint VARIANT STEP PATH`. Use the same syntax
for `uvit` and all three full variants after training them. Alternatively,
`--manifest evaluation_selection.json` accepts:

```json
{
  "checkpoints": [
    {"variant": "linear", "step": 10000, "checkpoint": "results/002-SiT-S-2-linear-Linear-velocity-None/checkpoints/0010000.pt"},
    {"variant": "linear_uvit", "step": 20000, "checkpoint": "results/001-SiT-S-2-linear_uvit-Linear-velocity-None/checkpoints/0020000.pt"},
    {"variant": "baseline", "step": 0, "checkpoint": "pretrained_models/SiT-S-2-256.pt"}
  ]
}
```

Manifest paths are relative to the manifest's directory. `--experiment-dir` can
be repeated or point to `results/`, but it requires both `--variants` and
`--steps`, selects their Cartesian product, and rejects missing or ambiguous
matches. Use a manifest or repeated explicit entries for uneven selections.
Add `--dry-run` to print the selection without generation or metrics.

For a real-generation plumbing smoke test only (no metric installation needed):

```bash
python evaluate.py --checkpoint baseline 0 pretrained_models/SiT-S-2-256.pt \
  --device cpu --num-samples 1 --num-sampling-steps 2 --sampling-method euler \
  --warmup-batches 1 --skip-metrics
```

Two-step Euler is only a plumbing test; keep the same proper sampler settings
across all quality comparisons. SDE is also supported via `--mode SDE
--sampling-method Euler` or `Heun`, with the existing diffusion/last-step options.
No likelihood or reversed (data-to-noise) integration is allowed for generation.

## Timing, sample reuse, and results

`sampling_utils.py` is shared by the evaluator and both existing sampling
scripts. It calls the repository's transport sampler, existing CFG implementation
(including its three-channel guidance convention), decoder scaling `1/0.18215`,
and original `sample_ddp.py` uint8 conversion. SiT can use FP32/BF16/FP16; the
integrator state and VAE decoding stay FP32. Labels are sampled uniformly over
0..999. Batch `i` uses seed `seed+i`, independently of checkpoint initialization;
warm-ups use separate seeds. Keep batch size and all sampler settings identical
across compared variants. This deterministic per-batch policy also allows
interrupted sample generation to resume without replaying previous batches.

Timing includes transport integration, CFG, and VAE decoding. It excludes
checkpoint/model loading, warm-up, noise/label creation, CPU image conversion,
disk writes, hashing, and metric calculation. CUDA is synchronized immediately
before and after each measured batch. The report includes seconds/image,
images/second, generation seconds, GPU, precision, batch size, CFG, resolution,
and actual model-call NFE. CFG doubles the model batch, not this call count.
For adaptive Dopri5, requested output points are **not** NFE; NFE varies with
the model. Fixed Euler with 250 output points uses 249 integration intervals.

Samples live under `evaluation_samples/VARIANT/STEP/CONFIG_HASH/` as one uint8
memory-mapped `samples.npy` plus metadata/progress. At 256 pixels, 50k images
occupy about **9.16 GiB per checkpoint/configuration**. Checkpoint content hashes,
all generation settings, library/runtime and source-code fingerprints prevent
incompatible reuse. Completed samples are checksum-verified before reuse for
metrics. Partial generation resumes at committed batch boundaries. An abrupt
kill may leave `.writer.lock`; remove it only after confirming no writer remains.

Every invocation creates a unique timestamped directory under
`evaluation_results/`, containing `evaluation_results.csv` and `.json`, written
after each checkpoint. Matching completed evaluations are reused from
`evaluation_results/completed/`. `--recompute-metrics` recalculates metrics from
verified existing samples; timing then remains the stored original generation
time and is labeled `cached_generation`. Reused completed rows set
`result_reused=true`; these are historical measurements, not a fresh speed test.
Failures produce error rows while successful results are retained; evaluation
continues by default, then exits nonzero if any entry failed. `--fail-fast`
stops after saving the first failure. Delete neither caches nor old outputs
unless you intend to discard them; new runs never overwrite old result tables.

## Lightweight verification

```bash
python verify_variants.py --device cpu --check-backward \
  --pretrained pretrained_models/SiT-S-2-256.pt \
  --trained-checkpoint results/002-SiT-S-2-linear-Linear-velocity-None/checkpoints/0040000.pt \
  --trained-checkpoint results/001-SiT-S-2-linear_uvit-Linear-velocity-None/checkpoints/0040000.pt
python verify_evaluation.py
```

These check all seven architectures, exact index maps, skip-source semantics,
conservative skip initialization, pretrained loading, frozen gradient masks,
strict trained-checkpoint loads, ODE/SDE/CFG equivalence, explicit selection,
resumable generation, cache corruption rejection, and failure/result persistence.
The evaluator loads trained weights strictly; it never substitutes randomly
initialized adapters when evaluating a modified variant from baseline weights.

Optional real metrics/cache smoke test (downloads Inception weights on first use):

```bash
SIT_TEST_METRICS=1 python verify_evaluation.py
```
