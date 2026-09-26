# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained SiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from checkpoint_utils import infer_learn_sigma, load_pretrained, model_weights, read_checkpoint
from models import SiT_models, SiT_VARIANTS
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from sampling_utils import build_sampler, sample_latents, decode_latents
import argparse
import sys
from time import time


def main(mode, args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "SiT-XL/2", "Only SiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
        assert args.image_size == 256, "512x512 models are not yet available for auto-download." # remove this line when 512x512 models are available
    ckpt_path = args.ckpt or f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = model_weights(read_checkpoint(ckpt_path), prefer_ema=True)
    learn_sigma = infer_learn_sigma(state_dict, patch_size=int(args.model.split("/")[-1]))

    # Load model:
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
        variant=args.variant,
    ).to(device)
    has_variant_weights = any(key.startswith("skip_projections.") for key in state_dict) or any(
        key.startswith(f"blocks.{i}.attn.kv.") for i in model.linear_block_indices)
    if args.variant != "baseline" and not has_variant_weights:
        load_pretrained(model, state_dict)
        print("Warning: new variant modules are untrained; sample only after fine-tuning.")
    else:
        model.load_state_dict(state_dict, strict=True)
        print("Checkpoint load: missing keys: []; unexpected keys: []")
    print(model.architecture_summary())
    model.eval()  # important!
    sample_fn = build_sampler(mode, args)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    
    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Keep this script's original always-CFG behavior, including cfg_scale=1.
    start_time = time()
    samples, _ = sample_latents(model, sample_fn, z, y, args.cfg_scale, force_cfg=True)
    samples = decode_latents(vae, samples)
    print(f"Sampling took {time() - start_time:.2f} seconds.")

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)
    
    mode = sys.argv[1]

    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"
    
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--variant", choices=SiT_VARIANTS, default="baseline")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a SiT checkpoint (default: auto-download a pre-trained SiT-XL/2 model).")


    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE
    
    args = parser.parse_known_args()[0]
    main(mode, args)
