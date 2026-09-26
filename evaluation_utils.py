"""Evaluation selection, cache identities, datasets, and standard metrics."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import pil_to_tensor

from checkpoint_utils import read_checkpoint
from latent_data import center_crop_arr
from models import SiT_VARIANTS


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_checkpoints(args):
    entries = []
    if not args.experiment_dir and (args.variants or args.steps):
        raise ValueError("--variants/--steps filter only --experiment-dir; use exact entries in a manifest")
    if args.checkpoint:
        entries = [dict(variant=v, step=int(s), checkpoint=p) for v, s, p in args.checkpoint]
    elif args.manifest:
        source = Path(args.manifest).resolve()
        entries = json.loads(source.read_text())["checkpoints"]
        entries = [dict(entry, checkpoint=str((source.parent / entry["checkpoint"]).resolve()))
                   for entry in entries]
    else:
        if not args.variants or not args.steps:
            raise ValueError("--experiment-dir requires explicit --variants and --steps")
        for folder in args.experiment_dir:
            for path in sorted(Path(folder).glob("**/checkpoints/*.pt")):
                if not path.stem.isdigit() or int(path.stem) not in args.steps:
                    continue
                checkpoint = read_checkpoint(path)
                config = checkpoint.get("args")
                variant = getattr(config, "variant", "baseline")
                if config is not None and variant in args.variants:
                    entries.append(dict(variant=variant, step=int(path.stem), checkpoint=str(path)))
                del checkpoint
        found = {(entry["variant"], entry["step"]) for entry in entries}
        missing = {(v, s) for v in args.variants for s in args.steps} - found
        if missing:
            raise ValueError(f"Requested variant/step combinations not found: {sorted(missing)}; use a manifest for uneven selections")
        if len(found) != len(entries):
            raise ValueError("Multiple experiments match a requested variant/step; use exact paths or a manifest")
    if not entries:
        raise ValueError("No checkpoints selected")
    seen = set()
    for entry in entries:
        if entry["variant"] not in SiT_VARIANTS or not isinstance(entry["step"], int) or entry["step"] < 0:
            raise ValueError(f"Invalid selection: {entry}")
        path = Path(entry["checkpoint"]).resolve()
        entry["checkpoint"] = str(path)
        key = (entry["variant"], entry["step"], str(path))
        if key in seen:
            raise ValueError(f"Duplicate selection: {key}")
        seen.add(key)
    return entries


class GeneratedImages(Dataset):
    def __init__(self, path):
        self.path = str(path)
        self.array = np.load(self.path, mmap_mode="r", allow_pickle=False)
        self.length = len(self.array)

    def __len__(self):
        return self.length

    def __getstate__(self):
        return dict(path=self.path, array=None, length=self.length)

    def __getitem__(self, index):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode="r", allow_pickle=False)
        return torch.from_numpy(np.array(self.array[index], copy=True)).permute(2, 0, 1)


class ReferenceImages(Dataset):
    """ImageFolder reference with the same deterministic ADM crop, no flips."""
    def __init__(self, root, image_size):
        root = Path(root).resolve()
        folder = ImageFolder(root)
        self.paths = [path for path, _ in folder.samples]
        self.image_size = image_size
        inventory = [(str(Path(p).relative_to(root)), Path(p).stat().st_size, Path(p).stat().st_mtime_ns)
                     for p in self.paths]
        self.identity = dict(root=str(root), image_size=image_size, transform="adm-center-crop-rgb-uint8-v1",
                             count=len(self.paths), inventory_hash=fingerprint(inventory))
        if len(self.paths) < 2:
            raise ValueError("FID requires at least two reference images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return pil_to_tensor(center_crop_arr(image.convert("RGB"), self.image_size))


def metric_version():
    try:
        version = importlib.metadata.version("torch-fidelity")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("Install metrics dependencies: python -m pip install -r requirements-eval.txt") from error
    if version != "0.4.0":
        raise ValueError("Install requirements-eval.txt to use the pinned torch-fidelity==0.4.0 protocol")
    return version


def calculate_metrics(samples, reference, args):
    import torch_fidelity
    metric_version()
    reference_key = fingerprint(dict(reference=reference.identity, backend="torch-fidelity-0.4.0",
                                     torch=torch.__version__, numpy=np.__version__,
                                     device=args.device, metric_batch_size=args.metric_batch_size,
                                     metric_tf32=False))
    # Metric feature extraction is always FP32 without TF32, independent of
    # the chosen SiT inference precision. Restore sampling flags afterward.
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        metrics = torch_fidelity.calculate_metrics(
            input1=GeneratedImages(samples), input2=reference,
            input2_cache_name="sit-reference-" + reference_key,
            cache_root=str(Path(args.metric_cache).resolve()),
            cuda=torch.device(args.device).type == "cuda", batch_size=args.metric_batch_size,
            feature_extractor="inception-v3-compat", feature_layer_fid="2048",
            feature_layer_isc="logits_unbiased",
            isc=True, fid=True, kid=False, prc=False, verbose=True,
            isc_splits=args.is_splits, rng_seed=args.seed)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
    return dict(FID=metrics["frechet_inception_distance"],
                IS_mean=metrics["inception_score_mean"], IS_std=metrics["inception_score_std"])
