"""Shared official SiT sampler, CFG, decoder and sample_ddp image conversion."""

import torch
from transport import create_transport, Sampler
from latent_data import LATENT_SCALE


def build_sampler(mode, args):
    sampler = Sampler(create_transport(args.path_type, args.prediction, args.loss_weight,
                                       args.train_eps, args.sample_eps))
    if mode == "ODE":
        if getattr(args, "likelihood", False):
            if args.cfg_scale != 1:
                raise ValueError("Likelihood is incompatible with guidance")
            return sampler.sample_ode_likelihood(sampling_method=args.sampling_method,
                        num_steps=args.num_sampling_steps, atol=args.atol, rtol=args.rtol)
        return sampler.sample_ode(sampling_method=args.sampling_method,
                    num_steps=args.num_sampling_steps, atol=args.atol, rtol=args.rtol,
                    reverse=args.reverse)
    if mode == "SDE":
        return sampler.sample_sde(sampling_method=args.sampling_method,
                    diffusion_form=args.diffusion_form, diffusion_norm=args.diffusion_norm,
                    last_step=args.last_step, last_step_size=args.last_step_size,
                    num_steps=args.num_sampling_steps)
    raise ValueError(f"Unknown sampling mode: {mode}")


@torch.no_grad()
def sample_latents(model, sample_fn, z, y, cfg_scale, precision="fp32", force_cfg=False):
    using_cfg = cfg_scale > 1 or force_cfg
    if using_cfg:
        z = torch.cat([z, z], 0)
        y = torch.cat([y, torch.full_like(y, model.y_embedder.num_classes)], 0)
        kwargs = dict(y=y, cfg_scale=cfg_scale)
        model_fn = model.forward_with_cfg
    else:
        kwargs = dict(y=y)
        model_fn = model.forward
    nfe = 0

    def counted_model(*inputs, **conditioning):
        nonlocal nfe
        nfe += 1
        # Keep the integrator state and VAE in FP32; autocast the SiT only.
        with torch.autocast(device_type=z.device.type,
                            dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
                            enabled=precision != "fp32"):
            return model_fn(*inputs, **conditioning).float()

    samples = sample_fn(z, counted_model, **kwargs)[-1]
    if using_cfg:
        samples, _ = samples.chunk(2, dim=0)
    return samples, nfe


@torch.no_grad()
def decode_latents(vae, latents):
    return vae.decode(latents.float() / LATENT_SCALE).sample


def images_to_uint8(samples):
    return torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
