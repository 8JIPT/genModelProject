"""Explicit checkpoint loading for the SiT ablation variants."""

import os

import torch

from download import find_model


def read_checkpoint(path):
    if path == "SiT-XL-2-256x256.pt" and not os.path.isfile(path):
        return find_model(path)
    # Full training checkpoints include argparse.Namespace. Retain the original
    # behavior on PyTorch >= 2.6; only load trusted local checkpoints.
    return torch.load(path, map_location="cpu", weights_only=False)


def model_weights(checkpoint, prefer_ema=False):
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["ema" if prefer_ema and "ema" in checkpoint else "model"]
    return checkpoint


def load_pretrained(model, weights, log=print):
    """Load all compatible original weights and require an exact key audit."""
    if model.variant == "baseline":
        model.load_state_dict(weights, strict=True)
        log("Pretrained load: missing keys: []; unexpected keys: []; newly initialized: []")
        return

    for index in model.linear_block_indices:
        if f"blocks.{index}.attn.qkv.weight" not in weights:
            raise ValueError(f"Expected original full attention in pretrained block {index}")
    removed = sorted(key for key in weights if any(
        key.startswith(f"blocks.{index}.attn.") for index in model.linear_block_indices))
    compatible = {key: value for key, value in weights.items() if key not in removed}
    result = model.load_state_dict(compatible, strict=False)
    expected_missing = {key for key in model.state_dict()
                        if key.startswith("skip_projections.") or any(
                            key.startswith(f"blocks.{index}.attn.")
                            for index in model.linear_block_indices)}
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError("Unexpected pretrained checkpoint mismatch: "
                           f"missing={result.missing_keys}, unexpected={result.unexpected_keys}; "
                           f"expected missing={sorted(expected_missing)}")
    log(f"Pretrained load: omitted incompatible attention keys: {removed}")
    log(f"Pretrained load: missing keys: {sorted(result.missing_keys)}")
    log(f"Pretrained load: unexpected keys: {result.unexpected_keys}")
    modules = [f"blocks.{i}.attn" for i in model.linear_block_indices]
    modules += [f"skip_projections.{i}" for i in model.skip_sources]
    log(f"Newly initialized modules: {modules}")


def infer_learn_sigma(weights, patch_size=2, in_channels=4):
    output_width = weights["final_layer.linear.weight"].shape[0]
    if output_width == patch_size ** 2 * in_channels * 2:
        return True
    if output_width == patch_size ** 2 * in_channels:
        return False
    raise ValueError(f"Unexpected final layer width: {output_width}")
