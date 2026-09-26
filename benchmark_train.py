"""Time real SiT optimizer updates; default: 200 updates, first 20 excluded."""

import json
from pathlib import Path
import statistics
from time import perf_counter

import torch
import torch.distributed as dist

from train import build_parser, main


class TrainingBenchmark:
    def __init__(self, total_steps=200, warmup_steps=20, output_json=None):
        if not 0 <= warmup_steps < total_steps:
            raise ValueError("Require 0 <= warmup steps < total steps")
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.output_json = output_json
        self.completed = 0
        self.skipped = 0
        self.pending_time = 0.0
        self.durations = []
        self.group_sizes = []

    def begin(self, device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            if self.completed == 0 and self.skipped == 0:
                torch.cuda.reset_peak_memory_stats(device)
        self.start = perf_counter()

    def end(self, device, updated, group_size, distributed=False):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - self.start
        if distributed:
            duration = torch.tensor(elapsed, device=device, dtype=torch.float64)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            elapsed = duration.item()
        self.pending_time += elapsed
        if not updated:
            self.skipped += 1
            return
        self.completed += 1
        if self.completed > self.warmup_steps:
            self.durations.append(self.pending_time)
            self.group_sizes.append(group_size)
        self.pending_time = 0.0
        if self.completed == self.warmup_steps and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def report(self, args, device, world_size, rank):
        if self.completed != self.total_steps:
            raise RuntimeError(f"Benchmark completed only {self.completed}/{self.total_steps} updates")
        average = statistics.mean(self.durations)
        summary = dict(
            variant=args.variant, model=args.model,
            device=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            precision=args.precision, global_batch_size=args.global_batch_size,
            per_rank_batch_size=args.global_batch_size // world_size, world_size=world_size,
            grad_accum_steps=args.grad_accum_steps,
            effective_batch_size=args.global_batch_size * args.grad_accum_steps,
            observed_effective_batch_min=args.global_batch_size * min(self.group_sizes),
            observed_effective_batch_max=args.global_batch_size * max(self.group_sizes),
            total_optimizer_steps=self.completed, warmup_steps=self.warmup_steps,
            timed_steps=len(self.durations), amp_skipped_updates=self.skipped,
            average_seconds_per_optimizer_step=average,
            median_seconds_per_optimizer_step=statistics.median(self.durations),
            steps_per_second=1 / average,
            estimated_hours={str(n): average * n / 3600 for n in (10000, 20000, 40000)},
            freeze_backbone=args.freeze_backbone if args.freeze_backbone is not None else args.variant != "baseline",
            ema=args.ema, gradient_checkpointing=args.gradient_checkpointing,
            learning_rate=args.learning_rate, num_workers=args.num_workers, seed=args.global_seed,
            latent_path=args.latent_path, pretrained=args.pretrained, resume_checkpoint=args.ckpt,
        )
        if device.type == "cuda":
            peaks = torch.tensor([torch.cuda.max_memory_allocated(device),
                                  torch.cuda.max_memory_reserved(device)], device=device)
            if world_size > 1:
                dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
            summary["peak_gpu_allocated_gib"], summary["peak_gpu_reserved_gib"] = (
                value / 1024**3 for value in peaks.tolist())
        if rank != 0:
            return
        print(f"\nVariant: {args.variant}\nDevice: {summary['device']}\nPrecision: {args.precision}")
        print(f"Batch size: {args.global_batch_size} global ({summary['per_rank_batch_size']} per rank)")
        print(f"Gradient accumulation: {args.grad_accum_steps}\nEffective batch size: {summary['effective_batch_size']}")
        if min(self.group_sizes) != args.grad_accum_steps:
            print(f"Partial accumulation groups at epoch boundaries: effective batch range "
                  f"{summary['observed_effective_batch_min']}–{summary['observed_effective_batch_max']}")
        print(f"Total optimizer steps: {self.completed}\nWarm-up steps excluded: {self.warmup_steps}")
        print(f"Timed steps: {len(self.durations)}\nAMP skipped updates: {self.skipped}")
        print(f"Average seconds / optimizer step: {average:.3f}")
        print(f"Median seconds / optimizer step: {summary['median_seconds_per_optimizer_step']:.3f}")
        print(f"Steps / second: {summary['steps_per_second']:.3f}")
        for n, hours in summary["estimated_hours"].items():
            print(f"Estimated time for {int(n):,} steps: {hours:.2f} h")
        if device.type == "cuda":
            print(f"Peak GPU memory allocated: {summary['peak_gpu_allocated_gib']:.3f} GiB")
            print(f"Peak GPU memory reserved: {summary['peak_gpu_reserved_gib']:.3f} GiB")
        print("Estimates = measured mean × optimizer steps; exclude setup, checkpointing, sampling, and evaluation.")
        if self.output_json:
            Path(self.output_json).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(model="SiT-S/2")
    parser.add_argument("--steps", type=int, default=200, help="Successful optimizer updates (default 200)")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--output-json", help="Optional machine-readable timing report")
    args = parser.parse_args()
    if not (args.pretrained or args.ckpt):
        parser.error("Supply --pretrained or --ckpt to benchmark the intended pretrained adaptation")
    args.sample_every = args.ckpt_every = args.log_every = 0
    args.wandb = False
    benchmark = TrainingBenchmark(args.steps, args.warmup_steps, args.output_json)
    main(args, benchmark=benchmark)
