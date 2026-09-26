"""Explicit, resumable checkpoint evaluation with shared SiT sampling and subset-FID."""

import argparse
import csv
from datetime import datetime, timezone
import gc
import importlib.metadata
import io
import json
import os
from pathlib import Path
from time import perf_counter
import uuid
import warnings

import numpy as np
import torch
from tqdm.auto import tqdm

from checkpoint_utils import infer_learn_sigma, model_weights, read_checkpoint
from evaluation_utils import (ReferenceImages, calculate_metrics, file_hash, fingerprint,
                              metric_version, select_checkpoints)
from latent_data import load_vae
from models import SiT_models, SiT_VARIANTS
from precompute_latents import atomic_json
from sampling_utils import build_sampler, sample_latents, decode_latents, images_to_uint8
from train_utils import parse_transport_args, parse_ode_args, none_or_str


SAMPLE_OPTIONS = ("model", "image_size", "num_classes", "vae", "weights", "num_samples", "batch_size",
                  "precision", "seed", "mode", "sampling_method", "num_sampling_steps", "cfg_scale",
                  "path_type", "prediction", "loss_weight", "train_eps", "sample_eps", "atol", "rtol",
                  "reverse", "diffusion_form", "diffusion_norm", "last_step", "last_step_size", "tf32")


def load_model(entry, args):
    checkpoint = read_checkpoint(entry["checkpoint"])
    config = checkpoint.get("args")
    if config is not None:
        expected = dict(model=args.model, image_size=args.image_size, num_classes=args.num_classes,
                        variant=entry["variant"], path_type=args.path_type, prediction=args.prediction)
        for key, value in expected.items():
            if getattr(config, key, "baseline" if key == "variant" else None) != value:
                raise ValueError(f"Checkpoint {key} conflicts with requested evaluation setting {value}")
    if "step" in checkpoint and checkpoint["step"] != entry["step"]:
        raise ValueError("Selected training step does not match checkpoint contents")
    if args.weights == "ema" and "ema" not in checkpoint:
        raise ValueError("EMA requested but checkpoint contains no EMA")
    weights = model_weights(checkpoint, prefer_ema=args.weights == "ema")
    model = SiT_models[args.model](input_size=args.image_size // 8, num_classes=args.num_classes,
                                  variant=entry["variant"],
                                  learn_sigma=infer_learn_sigma(weights, int(args.model.split("/")[-1])))
    # Evaluation never initializes random adapters on original weights.
    model.load_state_dict(weights, strict=True)
    print(model.architecture_summary())
    return model.to(args.device).eval().requires_grad_(False)


def runtime_identity(args):
    device = torch.device(args.device)
    source_root = Path(__file__).resolve().parent
    source_files = ["models.py", "LiT_linearAttn.py", "sampling_utils.py", "evaluate.py",
                    "latent_data.py", "checkpoint_utils.py"]
    source_files += [str(p.relative_to(source_root)) for p in sorted((source_root / "transport").glob("*.py"))]
    return dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),
                diffusers=importlib.metadata.version("diffusers"),
                timm=importlib.metadata.version("timm"),
                torchdiffeq=importlib.metadata.version("torchdiffeq"),
                device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                code={name: file_hash(source_root / name) for name in source_files})


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate_batch(model, vae, sample_fn, args, count, batch_index):
    # Independent seeds per batch make interrupted runs reproduce the same
    # noise, uniform labels, and SDE RNG draws without replaying earlier batches.
    torch.manual_seed((args.seed + batch_index) % (2**63 - 1))
    device = torch.device(args.device)
    z = torch.randn(count, 4, args.image_size // 8, args.image_size // 8, device=device)
    y = torch.randint(args.num_classes, (count,), device=device)
    synchronize(device)
    start = perf_counter()
    latents, nfe = sample_latents(model, sample_fn, z, y, args.cfg_scale, args.precision)
    images = decode_latents(vae, latents)
    synchronize(device)
    elapsed = perf_counter() - start
    if not torch.isfinite(images).all():
        raise ValueError("Non-finite generated images; try fp32/bf16 or inspect the checkpoint")
    return images_to_uint8(images), elapsed, nfe


def generate_samples(entry, args, identity, directory):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".writer.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Writer active or interrupted: {lock}; remove only after verifying no writer remains") from error
    os.close(fd)
    try:
        metadata = directory / "metadata.json"
        progress_file = directory / "progress.json"
        path = directory / "samples.npy"
        shape = (args.num_samples, args.image_size, args.image_size, 3)
        if metadata.exists():
            if json.loads(metadata.read_text()) != identity:
                raise ValueError("Existing sample cache metadata differs; refusing reuse")
        else:
            if path.exists() or progress_file.exists():
                raise ValueError("Sample cache lacks metadata")
            atomic_json(metadata, identity)
        progress = (json.loads(progress_file.read_text()) if progress_file.exists() else
                    dict(completed=0, generation_seconds=0.0, nfe=[], complete=False))
        if not 0 <= progress["completed"] <= args.num_samples:
            raise ValueError("Invalid sample-cache progress")
        if path.exists():
            array = np.load(path, mmap_mode="r+", allow_pickle=False)
            if array.shape != shape or array.dtype != np.uint8:
                raise ValueError("Sample cache shape/dtype mismatch")
        else:
            if progress["completed"]:
                raise ValueError("Sample cache data missing")
            array = np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=np.uint8)
        if progress["complete"]:
            if progress["completed"] != args.num_samples or file_hash(path) != progress["sha256"]:
                raise ValueError("Completed sample cache is damaged")
            print(f"Reusing verified samples: {directory}")
            return path, progress, True
        if progress["completed"] % args.batch_size and progress["completed"] != args.num_samples:
            raise ValueError("Sample prefix is not a complete batch")
        if progress["completed"] < args.num_samples:
            model = load_model(entry, args)
            vae = load_vae(args.vae, args.device)
            sample_fn = build_sampler(args.mode, args)
            for warmup in range(args.warmup_batches):
                print(f"Warm-up generation {warmup + 1}/{args.warmup_batches}")
                generate_batch(model, vae, sample_fn, args, args.batch_size, 1_000_000_000 + warmup)
            with tqdm(total=args.num_samples, initial=progress["completed"], unit="image") as bar:
                while progress["completed"] < args.num_samples:
                    start = progress["completed"]
                    n = min(args.batch_size, args.num_samples - start)
                    images, duration, nfe = generate_batch(model, vae, sample_fn, args, n, start // args.batch_size)
                    array[start:start + n] = images
                    array.flush()
                    with path.open("rb") as handle:
                        os.fsync(handle.fileno())
                    progress["completed"] += n
                    progress["generation_seconds"] += duration
                    progress["nfe"].append(nfe)
                    atomic_json(progress_file, progress)
                    bar.update(n)
            del model, vae, sample_fn
            gc.collect()
            if torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()
        progress.update(complete=True, sha256=file_hash(path))
        atomic_json(progress_file, progress)
        return path, progress, False
    finally:
        lock.unlink()


def save_results(directory, rows, args):
    atomic_json(directory / "evaluation_results.json", dict(settings=vars(args), results=rows))
    columns = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    temporary = directory / "evaluation_results.csv.tmp"
    with temporary.open("w", newline="") as handle:
        handle.write(buffer.getvalue())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, directory / "evaluation_results.csv")


def main(args):
    entries = select_checkpoints(args)
    print("Explicit evaluation selection:\n" + json.dumps(entries, indent=2))
    if args.dry_run:
        return
    if min(args.num_samples, args.batch_size, args.metric_batch_size) < 1 or args.warmup_batches < 1:
        raise ValueError("Counts and batch sizes must be positive; at least one warm-up batch is required")
    if args.num_sampling_steps < 2 or args.cfg_scale < 1 or args.likelihood or args.reverse:
        raise ValueError("Generation requires >=2 sampling points, CFG >=1, no likelihood or reverse integration")
    if args.mode == "SDE" and args.sampling_method not in ("Euler", "Heun"):
        raise ValueError("SDE requires --sampling-method Euler or Heun")
    if not args.skip_metrics and (args.num_samples < 2 or not 1 <= args.is_splits <= args.num_samples):
        raise ValueError("Metrics require >=2 samples and 1 <= IS splits <= sample count")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu or cuda[:index]")
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 unsupported on this device")
    elif args.precision == "fp16":
        raise ValueError("Use fp32 or bf16 on CPU")
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    reference = None
    metrics_identity = dict(backend=None)
    if not args.skip_metrics:
        if not args.reference:
            raise ValueError("FID requires --reference ImageFolder directory")
        reference = ReferenceImages(args.reference, args.image_size)
        metrics_identity = dict(backend="torch-fidelity-" + metric_version(), reference=reference.identity,
                                reference_name=args.reference_name, is_splits=args.is_splits,
                                metric_batch_size=args.metric_batch_size,
                                scipy=importlib.metadata.version("scipy"),
                                code=file_hash(Path(__file__).with_name("evaluation_utils.py")))
        if args.num_samples < 50000:
            warnings.warn("Small-sample FID is noisy and must not be compared with published FID-50K.")
        print(f"Reporting subset-FID against {len(reference):,} real images ({args.reference_name}); not official ADM/ImageNet FID.")
    runtime = runtime_identity(args)
    output_root = Path(args.output_dir)
    run_dir = output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    completed_dir = output_root / "completed"
    completed_dir.mkdir(exist_ok=True)
    rows = []
    for entry in entries:
        print(f"\nEvaluating {entry['variant']} step {entry['step']}: {entry['checkpoint']}", flush=True)
        row = dict(variant=entry["variant"], checkpoint=entry["checkpoint"], training_step=entry["step"],
                   identifier="baseline/pretrained" if entry["variant"] == "baseline" and entry["step"] == 0
                   else f"{entry['variant']}/{entry['step']}")
        try:
            identity = dict(version=1, checkpoint=entry, checkpoint_sha256=file_hash(entry["checkpoint"]),
                            settings={key: getattr(args, key) for key in SAMPLE_OPTIONS}, runtime=runtime,
                            label_policy="uniform-random-per-batch", rng_policy="seed+batch-index-v1",
                            warmup_batches=args.warmup_batches)
            key = fingerprint(identity)
            directory = Path(args.sample_dir) / entry["variant"] / str(entry["step"]) / key
            result_key = fingerprint(dict(samples=key, metrics=metrics_identity))
            saved_result = completed_dir / f"{result_key}.json"
            if saved_result.exists() and not args.recompute_metrics:
                row = json.loads(saved_result.read_text())
                row["result_reused"] = True
                print("Reusing completed evaluation (identical checkpoint/settings/reference).")
            else:
                samples, progress, reused = generate_samples(entry, args, identity, directory)
                seconds = progress["generation_seconds"]
                row.update(status="ok", FID=None, IS_mean=None, IS_std=None,
                           fid_protocol="subset-FID/torch-fidelity-0.4.0" if reference else "not computed",
                           reference=args.reference, reference_name=args.reference_name,
                           reference_count=len(reference) if reference else None,
                           reference_id=fingerprint(metrics_identity),
                           seconds_per_image=seconds / args.num_samples,
                           images_per_second=args.num_samples / seconds,
                           total_generation_seconds=seconds, num_generated_samples=args.num_samples,
                           sampling_method=args.sampling_method, mode=args.mode,
                           sampling_steps=args.num_sampling_steps, nfe_total=sum(progress["nfe"]),
                           nfe_mean_per_batch=float(np.mean(progress["nfe"])),
                           cfg_scale=args.cfg_scale, precision=args.precision, vae_precision="fp32",
                           batch_size=args.batch_size, seed=args.seed, GPU=runtime["gpu"],
                           resolution=args.image_size, weights=args.weights, is_splits=args.is_splits,
                           sample_dir=str(directory.resolve()), samples_reused=reused,
                           timing_source="cached_generation" if reused else "generation",
                           result_reused=False, configuration=json.dumps(identity, sort_keys=True))
                if reference:
                    row.update(calculate_metrics(samples, reference, args))
                atomic_json(saved_result, row)
        except Exception as error:
            row.update(status="error", error=f"{type(error).__name__}: {error}")
            print(row["error"], flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if row["status"] == "ok":
            print(f"{row['identifier']}: {row['seconds_per_image']:.4f} s/image, "
                  f"{row['images_per_second']:.3f} images/s; "
                  f"FID={row.get('FID')}, IS={row.get('IS_mean')} +/- {row.get('IS_std')}")
        rows.append(row)
        save_results(run_dir, rows, args)
        print(f"Results saved: {run_dir}", flush=True)
        if row["status"] == "error" and args.fail_fast:
            raise RuntimeError(row["error"])
    if any(row["status"] == "error" for row in rows):
        raise RuntimeError(f"Some evaluations failed; successful results preserved in {run_dir}")
    return run_dir


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--checkpoint", nargs=3, action="append", metavar=("VARIANT", "STEP", "PATH"))
    selection.add_argument("--manifest", help="JSON with checkpoints: [{variant, step, checkpoint}]")
    selection.add_argument("--experiment-dir", action="append", help="Experiment folder or results root; requires variants and steps")
    parser.add_argument("--variants", nargs="+", choices=SiT_VARIANTS)
    parser.add_argument("--steps", nargs="+", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", choices=SiT_models, default="SiT-S/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", choices=["ema", "mse"], default="ema")
    parser.add_argument("--weights", choices=["model", "ema"], default="model")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["ODE", "SDE"], default="ODE")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parse_transport_args(parser)
    parse_ode_args(parser)
    parser.add_argument("--diffusion-form", choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"], default="sigma")
    parser.add_argument("--diffusion-norm", type=float, default=1.0)
    parser.add_argument("--last-step", type=none_or_str, choices=[None, "Mean", "Tweedie", "Euler"], default="Mean")
    parser.add_argument("--last-step-size", type=float, default=0.04)
    parser.add_argument("--reference", help="Real reference ImageFolder; no published ADM statistics are assumed")
    parser.add_argument("--reference-name", default="imagenet50k-training-subset")
    parser.add_argument("--metric-cache", default="evaluation_cache")
    parser.add_argument("--metric-batch-size", type=int, default=16)
    parser.add_argument("--is-splits", type=int, default=10)
    parser.add_argument("--skip-metrics", action="store_true", help="Generation/timing smoke test only")
    parser.add_argument("--sample-dir", default="evaluation_samples")
    parser.add_argument("--output-dir", default="evaluation_results")
    parser.add_argument("--recompute-metrics", action="store_true", help="Recompute metrics, reuse compatible generated samples")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after saving the first failed row")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
