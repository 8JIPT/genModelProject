"""Smoke test all seven SiT-S/2 architectures without training or downloading weights."""

import argparse

import torch
from timm.models.vision_transformer import Attention

from checkpoint_utils import infer_learn_sigma, load_pretrained, model_weights, read_checkpoint
from LiT_linearAttn import LinearAttention
from models import SiT_S_2, SiT_VARIANTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pretrained", help="Optional original SiT-S/2 checkpoint")
    parser.add_argument("--check-backward", action="store_true",
                        help="Check gradients through frozen blocks with gradient checkpointing")
    parser.add_argument("--amp", action="store_true", help="Use CUDA fp16 autocast for the checks")
    parser.add_argument("--trained-checkpoint", action="append", default=[],
                        help="Repeat to check strict loading of existing trained checkpoints")
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

    for variant in SiT_VARIANTS:
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
        assert linear == (list(range(12)) if variant in ("full_linear", "full_linear_uvit")
                          else list(range(8, 12)) if "linear" in variant else [])
        assert full == [i for i in range(12) if i not in linear]
        assert model.skip_sources == ({11: 0, 10: 1, 9: 2, 8: 3, 7: 4}
                                      if variant in ("full_uvit", "full_linear_uvit")
                                      else {9: 2, 10: 1} if "uvit" in variant else {})
        assert len(model.skip_projections) == len(model.skip_sources)
        for projection in model.skip_projections.values():
            deep, shallow = torch.randn(1, 2, 384), torch.randn(1, 2, 384)
            assert torch.equal(projection(torch.cat([deep, shallow], dim=-1)), deep)
        # Check source OUTPUT -> target INPUT semantics in the real forward.
        saved_sources = {}
        handles = []
        for target, source in model.skip_sources.items():
            def save_source(module, inputs, output, source=source):
                saved_sources[source] = output
            def check_projection(module, inputs, source=source):
                assert torch.equal(inputs[0][..., 384:], saved_sources[source])
            handles.append(model.blocks[source].register_forward_hook(save_source))
            handles.append(model.skip_projections[str(target)].register_forward_pre_hook(check_projection))
        model = model.to(args.device).eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            output = model(torch.randn(1, 4, 32, 32, device=args.device),
                           torch.rand(1, device=args.device),
                           torch.randint(1000, (1,), device=args.device))
        for handle in handles:
            handle.remove()
        saved_sources.clear()
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
    for path in args.trained_checkpoint:
        checkpoint = read_checkpoint(path)
        previous = checkpoint["args"]
        weights = checkpoint["model"]
        model = SiT_S_2(input_size=previous.image_size // 8, num_classes=previous.num_classes,
                         variant=getattr(previous, "variant", "baseline"),
                         learn_sigma=infer_learn_sigma(weights))
        model.load_state_dict(weights, strict=True)
        if "ema" in checkpoint:
            model.load_state_dict(checkpoint["ema"], strict=True)
        print(f"Strict trained-checkpoint load passed: {path}")
    print("All seven SiT-S/2 forward checks passed.")


if __name__ == "__main__":
    main()
