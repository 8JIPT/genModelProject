"""Shared SiT image/latent conventions and a memory-mapped posterior dataset."""

import json
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# These are the conventions in the original train.py, not VAE config defaults.
LATENT_SCALE = 0.18215
FLIP_PROBABILITY = 0.5
CACHE_VERSION = 1
TRANSFORM_VERSION = "adm-center-crop-rgb-normalize-v1"


def vae_identifier(variant):
    return f"stabilityai/sd-vae-ft-{variant}"


def load_vae(variant, device):
    # Cached training without sample generation does not even import diffusers.
    from diffusers.models import AutoencoderKL
    return AutoencoderKL.from_pretrained(vae_identifier(variant)).to(device).requires_grad_(False).eval()


def center_crop_arr(pil_image, image_size):
    """Original ADM center crop from train.py, unchanged."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y:crop_y + image_size, crop_x:crop_x + image_size])


def image_transform(image_size, random_flip=True):
    steps = [transforms.Lambda(partial(center_crop_arr, image_size=image_size))]
    if random_flip:
        steps.append(transforms.RandomHorizontalFlip(p=FLIP_PROBABILITY))
    return transforms.Compose(steps + [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3, inplace=True),
    ])


@torch.no_grad()
def encode_posterior(vae, images):
    """Return unscaled FP32 [batch, mean/logvar, channel, height, width]."""
    posterior = vae.encode(images).latent_dist
    if posterior.deterministic:
        raise ValueError("This cache format expects the stochastic SiT VAE posterior")
    # Store the actual implementation's clamped logvar, not raw encoder outputs.
    return torch.stack((posterior.mean, posterior.logvar), dim=1).float()


def sample_posterior(parameters):
    mean, logvar = parameters.unbind(dim=1)
    return (mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)).mul_(LATENT_SCALE)


def validate_numeric_labels(class_to_idx):
    """Reject ImageFolder's silent reindexing of numeric class directories."""
    for name, label in class_to_idx.items():
        if name.isdecimal() and int(name) != label:
            raise ValueError(f"Class directory {name!r} would become label {label}; refusing to relabel")


class CachedLatentDataset(Dataset):
    def __init__(self, root, image_size=256, vae="ema", num_classes=1000):
        self.root = Path(root)
        with (self.root / "manifest.json").open() as handle:
            self.manifest = json.load(handle)
        m = self.manifest
        expected = dict(version=CACHE_VERSION, image_size=image_size,
                        vae=vae_identifier(vae), latent_scale=LATENT_SCALE,
                        flip_probability=FLIP_PROBABILITY, dtype="float32",
                        representation="mean_logvar", transform=TRANSFORM_VERSION,
                        num_classes=num_classes)
        for key, value in expected.items():
            if m.get(key) != value:
                raise ValueError(f"Cache {key}={m.get(key)!r}, expected {value!r}")
        self.samples = m["samples"]
        if not self.samples:
            raise ValueError("Empty latent cache")
        validate_numeric_labels(m["class_to_idx"])
        counts = [0] * num_classes
        for sample in self.samples:
            label = sample["label"]
            if not isinstance(label, int) or not 0 <= label < num_classes:
                raise ValueError(f"Invalid cached label: {label}")
            if m["class_to_idx"].get(Path(sample["path"]).parts[0]) != label:
                raise ValueError(f"Source path/label mismatch: {sample}")
            counts[label] += 1
        if counts != m["class_counts"] or not all(counts):
            raise ValueError("Cache class counts are invalid or some classes are missing")
        with (self.root / "progress.json").open() as handle:
            progress = json.load(handle)
        if not progress.get("complete") or progress["processed"] != len(self.samples):
            raise ValueError("Latent cache is incomplete; resume precompute_latents.py first")
        self.shape = (len(self.samples), 2, 2, 4, image_size // 8, image_size // 8)
        self._data = None
        self._open()

    def _open(self):
        if self._data is None:
            self._data = np.load(self.root / "posterior.npy", mmap_mode="r", allow_pickle=False)
            if self._data.shape != self.shape or self._data.dtype != np.float32:
                raise ValueError("Cache posterior array has an incompatible shape or dtype")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_data"] = None  # Reopen memmap in spawned DataLoader workers.
        return state

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        self._open()
        view = int(torch.rand(()) < FLIP_PROBABILITY)
        parameters = torch.from_numpy(np.array(self._data[index, view], copy=True))
        if not torch.isfinite(parameters).all():
            raise ValueError(f"Non-finite cache data for {self.samples[index]['path']}")
        return parameters, self.samples[index]["label"]
