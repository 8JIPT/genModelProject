"""Smoke test all four SiT-S/2 architectures without training or downloading weights."""

import argparse

import torch
from timm.models.vision_transformer import Attention

from checkpoint_utils import load_pretrained, model_weights, read_checkpoint
from LiT_linearAttn import LinearAttention
from models import SiT_S_2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pretrained", help="Optional original SiT-S/2 checkpoint")
    parser.add_argument("--check-backward", action="store_true",
                        help="Check gradients through frozen blocks with gradient checkpointing")
    parser.add_argument("--amp", action="store_true", help="Use CUDA fp16 autocast for the checks")
    args = parser.parse_args()
    if args.amp and not args.device.startswith("cuda"):
        parser.error("--amp requires --device cuda")
    torch.manual_seed(7)
    original = SiT_S_2(input_size=32)
    torch.manual_seed(7)
    explicit = SiT_S_2(input_size=32, variant="baseline")
    assert original.state_dict().keys() == explicit.state_dict().keys()
    assert all(torch.equal(original.state_dict()[key], explicit.state_dict()[key])
               for key in original.state_dict())
    # Without a downloaded checkpoint, use a baseline state dict to exercise
    # the same strict/partial loading paths and expected key audit.
    weights = (model_weights(read_checkpoint(args.pretrained)) if args.pretrained
               else original.state_dict())
    del original, explicit

    for variant in ("baseline", "linear", "uvit", "linear_uvit"):
        model = SiT_S_2(input_size=32, variant=variant)
        load_pretrained(model, weights, log=lambda _: None)
        if variant != "baseline":
            model.freeze_backbone()
            trainable = {name for name, parameter in model.named_parameters()
                         if parameter.requires_grad}
            expected = {name for name, _ in model.named_parameters()
                        if (any(name.startswith(f"blocks.{i}.attn.")
                                for i in model.linear_block_indices)
                            or name.startswith("skip_projections."))}
            assert trainable == expected
        linear = [i for i, block in enumerate(model.blocks) if isinstance(block.attn, LinearAttention)]
        full = [i for i, block in enumerate(model.blocks) if isinstance(block.attn, Attention)]
        assert linear == (list(range(8, 12)) if "linear" in variant else [])
        assert full == [i for i in range(12) if i not in linear]
        assert model.skip_sources == ({9: 2, 10: 1} if "uvit" in variant else {})
        assert len(model.skip_projections) == (2 if "uvit" in variant else 0)
        for projection in model.skip_projections.values():
            deep, shallow = torch.randn(1, 2, 384), torch.randn(1, 2, 384)
            assert torch.equal(projection(torch.cat([deep, shallow], dim=-1)), deep)
        model = model.to(args.device).eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            output = model(torch.randn(1, 4, 32, 32, device=args.device),
                           torch.rand(1, device=args.device),
                           torch.randint(1000, (1,), device=args.device))
        assert output.shape == (1, 4, 32, 32)
        assert torch.isfinite(output).all()
        if args.check_backward and variant != "baseline":
            # A freshly initialized SiT has zero output and attention gates.
            # Give those frozen gates a signal to test the graph through them.
            with torch.no_grad():
                torch.nn.init.normal_(model.final_layer.linear.weight, std=0.02)
                for block in model.blocks:
                    block.adaLN_modulation[-1].bias[768:1152].fill_(0.1)
            model.gradient_checkpointing = True
            model.train()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                prediction = model(torch.randn(1, 4, 32, 32, device=args.device),
                                   torch.rand(1, device=args.device),
                                   torch.randint(1000, (1,), device=args.device))
            loss = prediction.float().square().mean()
            scaler = torch.cuda.amp.GradScaler(enabled=args.amp, init_scale=1024)
            scaler.scale(loss).backward()
            trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for _, p in trainable), f"Missing or invalid gradients in {variant}"
            for prefix in ([f"blocks.{i}.attn." for i in model.linear_block_indices] +
                           [f"skip_projections.{i}." for i in model.skip_sources]):
                assert any(name.startswith(prefix) and p.grad.abs().sum() > 0
                           for name, p in trainable), f"No gradient in {prefix}"
        print(model.architecture_summary())
        del model, output
    print("All four SiT-S/2 forward checks passed.")


if __name__ == "__main__":
    main()
