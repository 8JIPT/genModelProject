## Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers (SiT)<br><sub>Official PyTorch Implementation</sub>

### [Paper](https://arxiv.org/pdf/2401.08740.pdf) | [Project Page](https://scalable-interpolant.github.io/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/willisma/SiT/blob/main/run_SiT.ipynb)

![SiT samples](visuals/visual.png)

This repo contains PyTorch model definitions, pre-trained weights and training/sampling code for our paper exploring 
interpolant models with scalable transformers (SiTs). 

> [**Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers**](https://arxiv.org/pdf/2401.08740.pdf)<br>
> [Nanye Ma](https://willisma.github.io), [Mark Goldstein](https://marikgoldstein.github.io/), [Michael Albergo](http://malbergo.me/), [Nicholas Boffi](https://nmboffi.github.io/), [Eric Vanden-Eijnden](https://wp.nyu.edu/courantinstituteofmathematicalsciences-eve2/), [Saining Xie](https://www.sainingxie.com)
> <br>New York University<br>

We present Scalable Interpolant Transformers (SiT), a family of generative models built on the backbone of Diffusion Transformers (DiT). The interpolant framework, which allows for connecting two distributions in a more flexible way than standard diffusion models, makes possible a modular study of various design choices impacting generative models built on dynamical transport: using discrete vs. continuous time learning, deciding the model to learn, choosing the interpolant connecting the distributions, and deploying a deterministic or stochastic sampler. By carefully introducing the above ingredients, SiT surpasses DiT uniformly across model sizes on the conditional ImageNet 256x256 benchmark using the exact same backbone, number of parameters, and GFLOPs. By exploring various diffusion coefficients, which can be tuned separately from learning, SiT achieves an FID-50K score of 2.06.

This repository contains:

* 🪐 A simple PyTorch [implementation](models.py) of SiT
* ⚡️ Pre-trained class-conditional SiT models trained on ImageNet 256x256
* 🛸 A SiT [training script](train.py) using PyTorch DDP

## Setup

First, download and set up the repo:

```bash
git clone https://github.com/willisma/SiT.git
cd SiT
```

We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda env create -f environment.yml
conda activate SiT
```


## Sampling [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/willisma/SiT/blob/main/run_SiT.ipynb)
![More SiT samples](visuals/visual_2.png)

**Pre-trained SiT checkpoints.** You can sample from our pre-trained SiT models with [`sample.py`](sample.py). Weights for our pre-trained SiT model will be 
automatically downloaded depending on the model you use. The script has various arguments to adjust sampler configurations (ODE & SDE), sampling steps, change the classifier-free guidance scale, etc. For example, to sample from
our 256x256 SiT-XL model with default ODE setting, you can use:

```bash
python sample.py ODE --image-size 256 --seed 1
```

For convenience, our pre-trained SiT models can be downloaded directly here as well:

| SiT Model     | Image Resolution | FID-50K | Inception Score | Gflops | 
|---------------|------------------|---------|-----------------|--------|
| [XL/2](https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0) | 256x256          | 2.06    | 270.27         | 119    |
<!-- | [XL/2](https://dl.fbaipublicfiles.com/SiT/models/SiT-XL-2-512x512.pt) | 512x512          | 2.62    |   252.21       | 525    | -->


**Custom SiT checkpoints.** If you've trained a new SiT model with [`train.py`](train.py) (see [below](#training-SiT)), you can add the `--ckpt`
argument to use your own checkpoint instead. For example, to sample from the EMA weights of a custom 
256x256 SiT-L/4 model with ODE sampler, run:

```bash
python sample.py ODE --model SiT-L/4 --image-size 256 --ckpt /path/to/model.pt
```

### Advanced sampler settings

|     |          |          |                         |
|-----|----------|----------|--------------------------|
| ODE | `--atol` | `float` |  Absolute error tolerance |
|     | `--rtol` | `float` | Relative error tolenrace |   
|     | `--sampling-method` | `str` | Sampling methods (refer to [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) ) |

|     |          |          |                         |
|-----|----------|----------|--------------------------|
| SDE | `--diffusion-form` | `str` | Form of SDE's diffusion coefficient (refer to Tab. 2 in [paper]()) |
|     | `--diffusion-norm` | `float` | Magnitude of SDE's diffusion coefficient |
|     | `--last-step` | `str` | Form of SDE's last step |
|     |               |       | None - Single SDE integration step |
|     |               |       | "Mean" - SDE integration step without diffusion coefficient |
|     |               |       | "Tweedie" - [Tweedie's denoising](https://efron.ckirby.su.domains/papers/2011TweediesFormula.pdf) step | 
|     |               |       | "Euler" - Single ODE integration step
|     | `--sampling-method` | `str` | Sampling methods |
|     |               |       | "Euler" - First order integration | 
|     |               |       | "Heun" - Second order integration | 

There are some more options; refer to [`train_utils.py`](train_utils.py) for details.

## Training SiT

We provide a training script for SiT in [`train.py`](train.py). To launch SiT-XL/2 (256x256) training with `N` GPUs on 
one node:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model SiT-XL/2 --data-path /path/to/imagenet/train
```

**Logging.** To enable `wandb`, firstly set `WANDB_KEY`, `ENTITY`, and `PROJECT` as environment variables:

```bash
export WANDB_KEY="key"
export ENTITY="entity name"
export PROJECT="project name"
```

Then in training command add the `--wandb` flag:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model SiT-XL/2 --data-path /path/to/imagenet/train --wandb
```

**Interpolant settings.** We also support different choices of interpolant and model predictions. For example, to launch SiT-XL/2 (256x256) with `Linear` interpolant and `noise` prediction: 

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model SiT-XL/2 --data-path /path/to/imagenet/train --path-type Linear --prediction noise
```

**Resume training.** To resume training from custom checkpoint:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model SiT-L/2 --data-path /path/to/imagenet/train --ckpt /path/to/model.pt
```

**Caution.** Resuming training will automatically restore both model, EMA, and optimizer states and training configs to be the same as in the checkpoint.

## Evaluation (FID, Inception Score, etc.)

We include a [`sample_ddp.py`](sample_ddp.py) script which samples a large number of images from a SiT model in parallel. This script 
generates a folder of samples as well as a `.npz` file which can be directly used with [ADM's TensorFlow
evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and
other metrics. For example, to sample 50K images from our pre-trained SiT-XL/2 model over `N` GPUs under default ODE sampler settings, run:

```bash
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py ODE --model SiT-XL/2 --num-fid-samples 50000
```

**Likelihood.** Likelihood evaluation is supported. To calculate likelihood, you can add the `--likelihood` flag to ODE sampler:

```bash
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py ODE --model SiT-XL/2 --likelihood
```

Notice that only under ODE sampler likelihood can be calculated; see [`sample_ddp.py`](sample_ddp.py) for more details and settings. 

## SiT-S/2 architecture ablations

The `--variant` option selects four architectures without changing the original
`SiT-S/2` baseline state-dict keys:

| Variant | Attention in blocks 0–7 | Attention in blocks 8–11 | Long skips |
| --- | --- | --- | --- |
| `baseline` | Original full attention | Original full attention | None |
| `linear` | Original full attention | LiT linear attention | None |
| `uvit` | Original full attention | Original full attention | Output 2 → input 9; output 1 → input 10 |
| `linear_uvit` | Original full attention | LiT linear attention | Same two skips |

Block numbers are zero based. Each skip concatenates `[current_deep, saved_shallow]`
and applies its own `Linear(2 * hidden_size, hidden_size)` immediately before the
target block. Its weight starts as `[I, 0]` with zero bias, so the skip is exactly
the identity at initialization. The two shallow features are the only saved skip
activations. The original block residual, MLP, timestep/class conditioning, and
adaLN are unchanged.

The provided `LiT_linearAttn.LinearAttention` is used directly. It has separate
`q` and `kv` projections and a depthwise 5×5 convolution over a square token grid.
SiT-S/2 at 256×256 has a 32×32 VAE latent and a 16×16 token grid (256 tokens,
384 channels, six 64-channel heads), which satisfies its shape requirements. The
LiT source had only its unused `einops` import and constructor print removed; a
square-grid check was added. Its `q`, `kv`, and convolution parameters are newly
initialized. Original full-attention `qkv` weights are omitted only for replaced
blocks; all other pretrained weights load unchanged. The loader reports every
omitted, missing, and unexpected key and rejects any mismatch beyond those new
modules. Baseline loads strictly. An original SiT-S/2 checkpoint must be supplied
locally; this repository auto-downloads only SiT-XL/2.

Run from this directory with an image-folder dataset and a local original
SiT-S/2 checkpoint. The same starting checkpoint, dataset, random seed,
transport, VAE, effective batch size, and training schedule should be used for
all four runs. With batch size 1 and accumulation 8, the effective batch is 8.
`--no-ema` and `--sample-every 0` reduce peak GPU memory; use them consistently
across runs. Single-GPU training runs directly with `python train.py` on Windows
or Linux. Multi-GPU training still uses `torchrun`.

For the local Windows conda environment, run these PowerShell commands after
replacing the two placeholder paths:

```powershell
$python = 'D:\Users\CLIENTE\anaconda3\envs\SiT\python.exe'
$common = @('--model', 'SiT-S/2', '--image-size', '256', '--data-path', 'D:\path\to\image-folder', '--pretrained', 'D:\path\to\original-SiT-S-2.pt', '--global-batch-size', '1', '--grad-accum-steps', '8', '--precision', 'fp16', '--gradient-checkpointing', '--no-ema', '--sample-every', '0', '--num-workers', '0', '--global-seed', '0')
& $python train.py @common --variant baseline
& $python train.py @common --variant linear
& $python train.py @common --variant uvit
& $python train.py @common --variant linear_uvit
```

The baseline trains all original SiT parameters. By default, each modified
variant freezes all unchanged SiT parameters (`requires_grad=False`) and trains
only its four linear-attention modules, its two skip projections, or both,
respectively. `--no-freeze-backbone` explicitly enables full fine-tuning. This
default changes the number of trainable parameters between runs; report it with
results because it affects the interpretation of an architecture ablation.
Startup logs show total/trainable parameters, attention block indices, and skips.
The optimizer includes only trainable parameters. Frozen prefixes naturally
retain no backward graph; after a trainable skip or attention module, autograd
still tracks the input path through frozen layers so gradients reach the new
module. `--gradient-checkpointing` uses non-reentrant PyTorch checkpointing;
PyTorch 2.x is recommended. For 8 GB VRAM, try `--precision fp16` first, or
`--precision bf16` on supported GPUs. Adjust accumulation without changing the
effective batch across experiments.

`--pretrained` starts a new run from original model weights. `--ckpt` resumes a
full training checkpoint, including optimizer and step; it cannot be combined
with `--pretrained`. For sampling a trained variant:

```bash
python sample.py ODE --model SiT-S/2 --variant linear_uvit --image-size 256 --ckpt /path/to/trained-checkpoint.pt
```

For a local smoke test, including exact equality between default and explicit
baseline initialization, run:

```bash
python verify_variants.py --device cpu --pretrained /path/to/original-SiT-S-2.pt
```

Omit `--pretrained` to check construction and forward passes without weights.
With a checkpoint, the baseline strict load and each variant's expected key
differences are also checked. Sampling from original weights with a modified
variant is possible for inspection but its new modules are untrained.
Add `--check-backward` to verify gradients through the frozen backbone and
checkpointed blocks using a synthetic nonzero output/gate signal. Add `--amp`
with `--device cuda` to run those checks under fp16 autocast.

## Cached VAE posteriors and training-time benchmark

`train.py` originally encodes RGB images with
`AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")` (`--vae mse` selects
`sd-vae-ft-mse`). Its transform is ADM resize/center crop, random horizontal
flip with probability 0.5, `ToTensor`, and normalization with mean/std 0.5.
It samples `vae.encode(x).latent_dist.sample()` on every visit and multiplies
by **0.18215**. The scaled SiT-S/2 input at 256 pixels is **[B, 4, 32, 32]**.
The transform and scale now live in `latent_data.py` and are shared by RGB
training and preprocessing; the transport objective and architectures are unchanged.

`precompute_latents.py` encodes the original and horizontally flipped RGB
orientations separately, in FP32. Flipping an already encoded latent would
not be equivalent. `posterior.npy` is a memory-mapped float32 array with shape
`[N, 2 orientations, 2 parameters (mean/logvar), 4, 32, 32]`. The log-variance
is the actual Diffusers posterior's clamped value; neither parameter is scaled.
`manifest.json` stores the VAE identifier, transform/version, scale, class
mapping/counts, and ordered relative image paths with labels, source sizes,
and modification times. `progress.json` records the committed prefix.
The complete 50k cache occupies approximately **3.05 GiB** plus JSON metadata.
There are three files, rather than one metadata file per image.

Each training visit selects an orientation with probability 0.5, samples
`mean + exp(0.5 * logvar) * randn_like(mean)`, and multiplies by 0.18215.
Both stochastic posterior sampling and flip augmentation are retained.
FP32 storage avoids posterior quantization. This preserves the original
training distribution, but does not promise bitwise identical full runs:
encoding with different batch sizes/devices/library kernels can cause small
floating-point differences, and worker scheduling/RNG consumption can differ.
The VAE remains frozen. Cache generation uses the same TF32 backend flags as
training; SiT AMP precision does not change the VAE encoding precision.

Preprocessing verifies all 1,000 classes by default and optionally enforces
50 images/class. Numeric folder names must agree with ImageFolder indices;
missing/reordered numeric labels cause an error. Corrupt images fail with their
path, without skipping/relabeling samples. Data is flushed before each progress
commit. Rerunning the same command resumes an incomplete cache and avoids
encoding an already completed cache. Changed source paths, sizes, timestamps,
or configuration are rejected. After an abrupt process kill, a `.writer.lock`
may remain: remove it only after verifying that no writer is running. An
incomplete cache cannot be used for training. Completed caches are portable;
training does not require the source RGB tree.

From this repository, in the `SiT` environment, precompute the actual local
50k subset (batch size 1 keeps VAE encoding memory modest):

```bash
conda activate SiT
python precompute_latents.py \
  --data-path data/imagenet50k/train \
  --output-path data/imagenet50k_latents \
  --image-size 256 --vae ema --num-classes 1000 --samples-per-class 50 \
  --batch-size 1 --num-workers 4 --device cuda
```

The VAE downloads from Hugging Face on first use. `HF_HOME` can point to an
existing download cache. CPU preprocessing is supported with `--device cpu`.

Define the common experiment arguments once in Bash. Here the global
micro-batch is 1 and accumulation is 8, so every update sees 8 examples on
one GPU; 50,000 is divisible by 8. Keep these settings identical across runs.
BF16 is checked for device support; if unavailable, choose FP16 or FP32 for
**all** runs. The commands require a working CUDA-compatible PyTorch/driver.

```bash
common=(
  --model SiT-S/2 --image-size 256
  --pretrained pretrained_models/SiT-S-2-256.pt
  --latent-path data/imagenet50k_latents --vae ema
  --global-batch-size 1 --grad-accum-steps 8
  --precision bf16 --learning-rate 1e-4 --num-workers 4
  --device cuda --global-seed 0 --gradient-checkpointing
  --no-ema --sample-every 0
)

python benchmark_train.py "${common[@]}" --variant baseline --no-freeze-backbone --output-json benchmark_baseline.json
python benchmark_train.py "${common[@]}" --variant linear --freeze-backbone --output-json benchmark_linear.json
python benchmark_train.py "${common[@]}" --variant uvit --freeze-backbone --output-json benchmark_uvit.json
python benchmark_train.py "${common[@]}" --variant linear_uvit --freeze-backbone --output-json benchmark_linear_uvit.json
```

Each benchmark executes **200 successful optimizer updates**, excludes the
first **20** for warm-up, and measures the remaining **180** with CUDA
synchronization around each update. It calls `train.main` and the same
`optimizer_step` as full training, reusing all model creation, checkpoint
loading, freezing, AdamW setup, transport loss, accumulation, and EMA behavior.
Timing includes DataLoader iteration/startup, transfers, posterior sampling,
forward/loss/backward, optimizer updates, and EMA if enabled. Checkpointing,
sample generation, W&B and periodic loss logging are disabled for the benchmark.
It reports mean/median step time, throughput, configuration, and peak CUDA
allocated/reserved memory during the timed phase. DDP uses the slowest rank's
step time and largest rank memory. FP16 overflow attempts do not increment
the step counter; their time is charged to the next successful update.
`--steps` and `--warmup-steps` can shorten a development smoke test.

The 10k/20k/40k estimates are simply `mean_seconds_per_update * target_steps`.
They exclude startup, checkpoints, image generation and evaluation, and a
short run may benefit from the OS file cache. They are training-time estimates,
not model quality measurements. As in the original loop, a partial accumulation
group at an epoch boundary uses its actual group size; the benchmark reports
its effective-batch range when this happens. Choose batch/accumulation values
that divide the dataset if a constant effective batch is required.

After reviewing those timings, a 40,000-update combined-variant run is:

```bash
python train.py "${common[@]}" --variant linear_uvit --freeze-backbone \
  --max-steps 40000 --ckpt-every 5000 --log-every 100 --results-dir results
```

For `linear` or `uvit`, change only the variant; for the baseline use
`--variant baseline --no-freeze-backbone`. This retains the existing policy:
baseline fine-tunes all original trainable weights, while modified variants
train only their added/replaced modules. The untouched pretrained baseline
remains a separate evaluation reference. `--freeze-backbone` on baseline is
rejected because it has no new modules to optimize.

`--max-steps` overrides the epoch limit and counts successful optimizer updates.
Training saves a final checkpoint at that step even when it is not a periodic
save step. To resume, replace `--pretrained ...` with `--ckpt ...` and retain the
other settings; 40,000 then means the **total** checkpoint step target, not
40,000 additional steps. Model/optimizer/EMA/scaler states are restored, but,
as before, data-position and RNG states are not restored for bitwise replay.
Full checkpoints contain Python configuration objects; only load trusted files.

`--latent-path` and `--data-path` are mutually exclusive. Replacing the former
with `--data-path data/imagenet50k/train` retains RGB + stochastic VAE training.
With cached data and `--sample-every 0`, no VAE is instantiated. If image
sampling is explicitly enabled, its decoder is loaded lazily at the first
sampling event and retained; this adds memory and is outside benchmark timing.

Run the small real-image smoke test (temporary two-class fixture, actual VAE
and SiT checkpoint, a few optimizer updates; no 200-step run):

```bash
python verify_latents.py --device cpu \
  --image data/imagenet50k/train/0279/000.jpg \
  --pretrained pretrained_models/SiT-S-2-256.pt
python verify_variants.py --device cpu --check-backward \
  --pretrained pretrained_models/SiT-S-2-256.pt
```

The first checks both orientations, source/label preservation, fresh posterior
noise, identical latent values with controlled noise on the tested device,
cache restart/validation, corrupt input detection, spawned workers, all four
cached optimizer updates, the RGB pathway, short benchmark accounting,
absence of VAE loading in cached training, and checkpoint save/resume.

### Enhancements

Training (and sampling) could likely be speed-up significantly by:
- [ ] using [Flash Attention](https://github.com/HazyResearch/flash-attention) in the SiT model
- [ ] using `torch.compile` in PyTorch 2.0

Basic features that would be nice to add:
- [ ] Monitor FID and other metrics
- [x] AMP/bfloat16 support in `train.py`

Precision in likelihood calculation could likely be improved by:
- [ ] Uniform / Gaussian Dequantization


## Differences from JAX

Our models were originally trained in JAX on TPUs. The weights in this repo are ported directly from the JAX models. 
There may be minor differences in results stemming from sampling on different platforms (TPU vs. GPU). We observed that sampling on TPU performs marginally worse than GPU (2.15 FID 
versus 2.06 in the paper).


## License
This project is under the MIT license. See [LICENSE](LICENSE.txt) for details.


