# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from contextlib import nullcontext
from itertools import count

from models import SiT_models
from checkpoint_utils import infer_learn_sigma, load_pretrained, model_weights, read_checkpoint
from transport import create_transport, Sampler
from latent_data import (CachedLatentDataset, LATENT_SCALE, center_crop_arr, image_transform,
                         load_vae, sample_posterior, validate_numeric_labels)
from train_utils import parse_transport_args
import wandb_utils


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if not dist.is_initialized() or dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def optimizer_step(args, model, raw_model, ema, transport, vae, opt, scaler,
                   batches, group_size, device, distributed=False):
    """The real training update, including data fetch and posterior sampling."""
    accumulated_loss = 0.0
    opt.zero_grad(set_to_none=True)
    for micro_step in range(group_size):
        x, y = next(batches)
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            x = (sample_posterior(x) if args.latent_path else
                 vae.encode(x).latent_dist.sample().mul_(LATENT_SCALE))
        sync_context = (model.no_sync() if distributed and micro_step + 1 < group_size
                        else nullcontext())
        with sync_context:
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16,
                                enabled=args.precision != "fp32"):
                loss = transport.training_losses(model, x, dict(y=y))["loss"].mean()
            scaler.scale(loss / group_size).backward()
        accumulated_loss += loss.item()
    old_scale = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    # GradScaler decreases its scale when it skipped an update due to overflow.
    updated = not scaler.is_enabled() or scaler.get_scale() >= old_scale
    opt.zero_grad(set_to_none=True)
    if updated and ema is not None:
        update_ema(ema, raw_model)
    return accumulated_loss / group_size, updated


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args, benchmark=None):
    """
    Trains a new SiT model.
    """
    validate_args(args)

    # A single GPU can train directly (including on Windows); torchrun retains DDP.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("gloo" if os.name == "nt" or args.device == "cpu" else "nccl")
        rank = dist.get_rank()
        device = torch.device("cpu" if args.device == "cpu" else f"cuda:{os.environ['LOCAL_RANK']}")
    else:
        rank = 0
        device = torch.device(args.device)
    assert args.global_batch_size % world_size == 0, "Batch size must be divisible by world size."
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    local_batch_size = int(args.global_batch_size // world_size)

    # Setup an experiment folder:
    if benchmark is not None:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logger = logging.getLogger(__name__)
    elif rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
        experiment_name = f"{experiment_index:03d}-{model_string_name}-{args.variant}-" \
                        f"{args.path_type}-{args.prediction}-{args.loss_weight}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        if args.wandb:
            wandb_utils.initialize(args, os.environ["ENTITY"], experiment_name, os.environ["PROJECT"])
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    if args.pretrained and args.ckpt:
        raise ValueError("Use --pretrained for a new run or --ckpt to resume, not both")
    if args.variant != "baseline" and not (args.pretrained or args.ckpt):
        raise ValueError("A variant requires --pretrained original SiT weights or --ckpt to resume")
    pretrained_weights = model_weights(read_checkpoint(args.pretrained)) if args.pretrained else None
    resume = read_checkpoint(args.ckpt) if args.ckpt else None
    source_weights = pretrained_weights if pretrained_weights is not None else (
        resume["model"] if resume is not None and "model" in resume else None)
    patch_size = int(args.model.split("/")[-1])
    checkpoint_learn_sigma = (infer_learn_sigma(source_weights, patch_size=patch_size)
                              if source_weights is not None else None)
    if (args.learn_sigma is not None and checkpoint_learn_sigma is not None
            and args.learn_sigma != checkpoint_learn_sigma):
        raise ValueError("--learn-sigma setting conflicts with checkpoint final layer shape")
    learn_sigma = (checkpoint_learn_sigma if checkpoint_learn_sigma is not None else
                   (args.learn_sigma if args.learn_sigma is not None else True))
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
        variant=args.variant,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    if pretrained_weights is not None:
        load_pretrained(model, pretrained_weights, logger.info)
    if resume is not None:
        if not all(key in resume for key in ("model", "opt", "args")):
            raise ValueError("--ckpt requires a full training checkpoint; use --pretrained for model weights")
        previous = resume["args"]
        if (previous.model, previous.image_size, getattr(previous, "variant", "baseline")) != (
                args.model, args.image_size, args.variant):
            raise ValueError("Checkpoint model, image size, or variant does not match this run")
        model.load_state_dict(resume["model"], strict=True)
        logger.info("Resumed model: strict load; missing keys: []; unexpected keys: []")
    freeze_backbone = args.freeze_backbone
    if freeze_backbone is None:
        freeze_backbone = args.variant != "baseline"
    if freeze_backbone:
        model.freeze_backbone()
    logger.info(model.architecture_summary())
    ema = deepcopy(model).to(device) if args.ema else None
    if ema is not None:
        requires_grad(ema, False)

    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    raw_model = model.module if distributed else model
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity; 
    transport_sampler = Sampler(transport)
    vae = None if args.latent_path else load_vae(args.vae, device)
    # Only parameters with gradients are passed to the optimizer.
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.learning_rate, weight_decay=0)
    scaler = torch.cuda.amp.GradScaler(enabled=args.precision == "fp16")
    if resume is not None:
        opt.load_state_dict(resume["opt"])
        if ema is not None:
            if "ema" not in resume:
                raise ValueError("Resume checkpoint has no EMA; use --no-ema")
            ema.load_state_dict(resume["ema"], strict=True)
        if args.precision == "fp16" and "scaler" in resume:
            scaler.load_state_dict(resume["scaler"])
    # Different variants initialize different numbers of new parameters. Reset
    # the training RNG so data order, VAE sampling, and transport noise start
    # from the same seed in each ablation run.
    torch.manual_seed(seed)

    # Setup data:
    if args.latent_path:
        dataset = CachedLatentDataset(args.latent_path, args.image_size, args.vae, args.num_classes)
    else:
        dataset = ImageFolder(args.data_path, transform=image_transform(args.image_size))
        validate_numeric_labels(dataset.class_to_idx)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True
    )
    if not len(loader):
        raise ValueError("Dataset is too small for this batch size/world size with drop_last=True")
    logger.info(f"Dataset contains {len(dataset):,} examples ({args.latent_path or args.data_path})")

    # Prepare models for training:
    if ema is not None and resume is None:
        update_ema(ema, raw_model, decay=0)
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    if ema is not None:
        ema.eval()

    # Variables for monitoring/logging purposes:
    train_steps = resume.get("step", 0) if resume is not None else 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    if args.sample_every > 0:
        # Labels to condition the model with (feel free to change):
        ys = torch.randint(args.num_classes, size=(local_batch_size,), device=device)
        use_cfg = args.cfg_scale > 1.0
        # Create sampling noise:
        n = ys.size(0)
        zs = torch.randn(n, 4, latent_size, latent_size, device=device)

        # Setup classifier-free guidance:
        if use_cfg:
            zs = torch.cat([zs, zs], 0)
            y_null = torch.tensor([args.num_classes] * n, device=device)
            ys = torch.cat([ys, y_null], 0)
            sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
            model_fn = (ema if ema is not None else raw_model).forward_with_cfg
        else:
            sample_model_kwargs = dict(y=ys)
            model_fn = (ema if ema is not None else raw_model).forward

    if benchmark is not None:
        args.max_steps = train_steps + benchmark.total_steps
    logger.info(f"Training until step {args.max_steps}" if args.max_steps else f"Training for {args.epochs} epochs...")
    if args.max_steps is not None and train_steps >= args.max_steps:
        logger.info("Checkpoint already reached the requested optimizer step budget.")
        cleanup()
        return
    skipped_updates = 0
    epochs = count() if args.max_steps is not None else range(args.epochs)
    for epoch in epochs:
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        batches = None
        for group_start in range(0, len(loader), args.grad_accum_steps):
            group_size = min(args.grad_accum_steps, len(loader) - group_start)
            if benchmark is not None:
                benchmark.begin(device)
            if batches is None:
                batches = iter(loader)
            loss, updated = optimizer_step(args, model, raw_model, ema, transport, vae,
                                           opt, scaler, batches, group_size, device, distributed)
            if benchmark is not None:
                benchmark.end(device, updated, group_size, distributed)
            if not updated:
                skipped_updates += 1
                if skipped_updates >= 100:
                    raise RuntimeError("100 consecutive AMP updates skipped; try bf16/fp32 or inspect numerical stability")
                logger.warning("AMP overflow: optimizer update skipped; step counter unchanged")
                continue
            skipped_updates = 0
            running_loss += loss
            log_steps += 1
            train_steps += 1
            if args.log_every > 0 and train_steps % args.log_every == 0:
                # Measure training speed:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                if distributed:
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        { "train loss": avg_loss, "train steps/sec": steps_per_sec },
                        step=train_steps
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save SiT checkpoint:
            if benchmark is None and ((args.ckpt_every > 0 and train_steps % args.ckpt_every == 0)
                                      or train_steps == args.max_steps):
                if rank == 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "step": train_steps,
                        "scaler": scaler.state_dict(),
                    }
                    if ema is not None:
                        checkpoint["ema"] = ema.state_dict()
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                if distributed:
                    dist.barrier()
            
            if args.sample_every > 0 and train_steps % args.sample_every == 0:
                logger.info("Generating samples...")
                if vae is None:
                    vae = load_vae(args.vae, device)
                if ema is None:
                    raw_model.eval()
                with torch.no_grad():
                    sample_fn = transport_sampler.sample_ode() # default to ode sampling
                    samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                    if distributed:
                        dist.barrier()

                    if use_cfg: #remove null samples
                        samples, _ = samples.chunk(2, dim=0)
                    samples = vae.decode(samples / LATENT_SCALE).sample
                    if distributed:
                        out_samples = torch.zeros((args.global_batch_size, 3, args.image_size, args.image_size), device=device)
                        dist.all_gather_into_tensor(out_samples, samples)
                    else:
                        out_samples = samples

                if args.wandb:
                    wandb_utils.log_image(out_samples, train_steps)
                if ema is None:
                    raw_model.train()
                logger.info("Generating samples done.")

            if args.max_steps is not None and train_steps >= args.max_steps:
                break
        if args.max_steps is not None and train_steps >= args.max_steps:
            break

    if benchmark is not None:
        benchmark.report(args, device, world_size, rank)
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


def build_parser():
    # Default args here will train SiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--data-path", type=str)
    data.add_argument("--latent-path", type=str)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu (for smoke tests)")
    parser.add_argument("--max-steps", type=int, help="Stop at this successful optimizer step count; overrides epochs")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--variant", choices=["baseline", "linear", "uvit", "linear_uvit"], default="baseline")
    freeze = parser.add_mutually_exclusive_group()
    freeze.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true",
                        help="Train only new modules")
    freeze.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.set_defaults(freeze_backbone=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--no-ema", dest="ema", action="store_false", help="Save GPU memory by omitting EMA")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Original SiT weights for a new ablation run")
    sigma = parser.add_mutually_exclusive_group()
    sigma.add_argument("--learn-sigma", dest="learn_sigma", action="store_true")
    sigma.add_argument("--no-learn-sigma", dest="learn_sigma", action="store_false")
    parser.set_defaults(learn_sigma=None)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Full training checkpoint to resume, including optimizer")

    parse_transport_args(parser)
    return parser


def validate_args(args):
    if args.grad_accum_steps < 1 or args.global_batch_size < 1 or args.epochs < 1:
        raise ValueError("Accumulation, batch size, and epochs must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: cpu, cuda, cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; check the PyTorch build/driver, or use --device cpu for smoke tests")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", device.index or 0)))
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support bf16")
    elif args.precision == "fp16":
        raise ValueError("fp16 training requires CUDA; use fp32 or bf16 on CPU")


if __name__ == "__main__":
    main(build_parser().parse_args())
