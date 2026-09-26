"""Cache both image orientations' VAE posteriors, with resumable prefix commits."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from latent_data import (CACHE_VERSION, FLIP_PROBABILITY, LATENT_SCALE, TRANSFORM_VERSION,
                         encode_posterior, image_transform, load_vae, vae_identifier,
                         validate_numeric_labels)


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class SourceImages(ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as error:
            raise RuntimeError(f"Invalid/corrupt image: {self.samples[index][0]}") from error


def precompute(args, vae=None):
    if args.batch_size < 1 or args.num_workers < 0 or args.num_classes < 1:
        raise ValueError("Batch size/classes must be positive and workers nonnegative")
    if args.image_size % 8:
        raise ValueError("Image size must be divisible by 8")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    # Match the pre-existing training flags. VAE encoding itself remains FP32.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    source = Path(args.data_path).resolve()
    dataset = SourceImages(source, transform=image_transform(args.image_size, random_flip=False))
    validate_numeric_labels(dataset.class_to_idx)
    counts = Counter(dataset.targets)
    if set(counts) != set(range(args.num_classes)):
        raise ValueError(f"Expected all {args.num_classes} classes; found {len(counts)}")
    if args.samples_per_class is not None and any(
            count != args.samples_per_class for count in counts.values()):
        raise ValueError(f"Expected exactly {args.samples_per_class} images/class; counts={dict(counts)}")
    print(f"Found {len(dataset):,} images; all {args.num_classes} classes represented.")
    print(f"Class counts: {dict(sorted(counts.items()))}")
    samples = []
    for path, label in dataset.samples:
        stat = Path(path).stat()
        samples.append(dict(path=Path(path).relative_to(source).as_posix(), label=label,
                            size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    manifest = dict(version=CACHE_VERSION, image_size=args.image_size,
                    vae=vae_identifier(args.vae), latent_scale=LATENT_SCALE,
                    flip_probability=FLIP_PROBABILITY, dtype="float32",
                    representation="mean_logvar", transform=TRANSFORM_VERSION,
                    num_classes=args.num_classes, class_to_idx=dataset.class_to_idx,
                    class_counts=[counts[i] for i in range(args.num_classes)],
                    source_root=str(source), samples=samples,
                    axes=["image", "original_or_flipped", "mean_or_logvar", "channel", "height", "width"])
    root = Path(args.output_path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents two writers from damaging the same cache.
    lock = root / ".writer.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Writer lock exists: {lock}. If the earlier process was killed, "
                           "verify it has stopped before removing this lock.") from error
    os.close(fd)
    try:
        manifest_path = root / "manifest.json"
        progress_path = root / "progress.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Existing cache configuration or source files changed; use a new output path")
        else:
            if (root / "posterior.npy").exists() or progress_path.exists():
                raise ValueError("Cache data exists without a manifest; use a new output path")
            atomic_json(manifest_path, manifest)
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else dict(processed=0, complete=False)
        start = progress["processed"]
        if not 0 <= start <= len(dataset):
            raise ValueError("Invalid cache progress")
        shape = (len(dataset), 2, 2, 4, args.image_size // 8, args.image_size // 8)
        array_path = root / "posterior.npy"
        if start and not array_path.exists():
            raise ValueError("Posterior data missing for an existing progress record")
        if array_path.exists():
            array = np.load(array_path, mmap_mode="r+", allow_pickle=False)
            if array.shape != shape or array.dtype != np.float32:
                raise ValueError("Existing posterior array shape/dtype does not match")
        else:
            array = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.float32, shape=shape)
        atomic_json(progress_path, dict(processed=start, complete=False))
        if start < len(dataset):
            vae = load_vae(args.vae, device) if vae is None else vae.to(device).requires_grad_(False).eval()
            loader = DataLoader(Subset(dataset, range(start, len(dataset))),
                                batch_size=args.batch_size, num_workers=args.num_workers,
                                shuffle=False, pin_memory=device.type == "cuda")
            with tqdm(total=len(dataset), initial=start, unit="image", desc="VAE posteriors") as bar:
                for images, labels in loader:
                    images = images.to(device)
                    # Encode the flipped RGB input separately. Flipping a latent is not equivalent.
                    original = encode_posterior(vae, images)
                    flipped = encode_posterior(vae, images.flip(-1))
                    parameters = torch.stack((original, flipped), dim=1).cpu().numpy()
                    end = start + len(images)
                    if parameters.shape != (end - start, *shape[1:]) or not np.isfinite(parameters).all():
                        raise ValueError(f"Invalid VAE posterior for source indices {start}:{end}")
                    if labels.tolist() != [sample["label"] for sample in samples[start:end]]:
                        raise ValueError("Source ordering/labels changed during preprocessing")
                    array[start:end] = parameters
                    array.flush()
                    # Data becomes durable before the progress marker is committed.
                    with array_path.open("rb") as handle:
                        os.fsync(handle.fileno())
                    atomic_json(progress_path, dict(processed=end, complete=False))
                    bar.update(end - start)
                    start = end
        atomic_json(progress_path, dict(processed=start, complete=True))
        size = sum(path.stat().st_size for path in root.iterdir() if path.is_file())
        print(f"Processed {start:,}/{len(dataset):,} examples. Cache: {root}")
        print(f"Approximate disk size: {size / 1024**3:.3f} GiB (FP32, both orientations)")
        return root
    finally:
        lock.unlink()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vae", choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="Override only for small tests or a different dataset")
    parser.add_argument("--samples-per-class", type=int, help="Require this many images in every class")
    return parser


if __name__ == "__main__":
    precompute(build_parser().parse_args())
