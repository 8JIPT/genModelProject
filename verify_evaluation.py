"""Small deterministic evaluation plumbing tests; real models checked by verify_variants.py."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import evaluate
from evaluation_utils import fingerprint, select_checkpoints, ReferenceImages, calculate_metrics
from sampling_utils import build_sampler, sample_latents


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.y_embedder = argparse.Namespace(num_classes=1000)

    def forward(self, x, t, y):
        return x * 0.1 + y[:, None, None, None].float() * 1e-5

    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[:len(x) // 2]
        raw = self.forward(torch.cat([half, half]), t, y)
        eps, rest = raw[:, :3], raw[:, 3:]
        cond, uncond = eps.chunk(2)
        guided = uncond + cfg_scale * (cond - uncond)
        return torch.cat([torch.cat([guided, guided]), rest], dim=1)


class ToyVAE:
    def decode(self, z):
        return argparse.Namespace(sample=torch.nn.functional.interpolate(z[:, :3], scale_factor=8).tanh())


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "original.pt"
        self.checkpoint.write_bytes(b"checkpoint identity fixture")
        self.args = evaluate.build_parser().parse_args([
            "--checkpoint", "baseline", "0", str(self.checkpoint), "--device", "cpu",
            "--skip-metrics", "--num-samples", "3", "--batch-size", "2", "--warmup-batches", "1",
            "--num-sampling-steps", "3", "--sampling-method", "euler",
            "--sample-dir", str(self.root / "samples"), "--output-dir", str(self.root / "results")])
        self.entry = select_checkpoints(self.args)[0]

    def tearDown(self):
        self.temp.cleanup()

    def test_selection_and_identity(self):
        self.assertEqual(len(select_checkpoints(self.args)), 1)
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"checkpoints": [dict(variant="uvit", step=20000, checkpoint="original.pt")]}))
        self.args.checkpoint = None
        self.args.manifest = str(manifest)
        self.assertEqual(select_checkpoints(self.args)[0]["variant"], "uvit")
        self.assertNotEqual(fingerprint(dict(seed=0)), fingerprint(dict(seed=1)))
        self.args.manifest = None
        self.args.experiment_dir = [str(self.root)]
        with self.assertRaises(ValueError):
            select_checkpoints(self.args)

    def test_official_sampler_equivalence_and_cfg(self):
        model = ToyModel()
        z = torch.randn(2, 4, 32, 32)
        y = torch.tensor([10, 99])
        for mode, method in (("ODE", "euler"), ("SDE", "Euler")):
            self.args.sampling_method = method
            fn = build_sampler(mode, self.args)
            for cfg in (1., 4.):
                torch.manual_seed(123)
                if cfg > 1:
                    expected = fn(torch.cat([z, z]), model.forward_with_cfg,
                                  y=torch.cat([y, torch.full_like(y, 1000)]), cfg_scale=cfg)[-1].chunk(2)[0]
                else:
                    expected = fn(z, model.forward, y=y)[-1]
                torch.manual_seed(123)
                actual, nfe = sample_latents(model, fn, z, y, cfg)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertGreater(nfe, 0)

    def test_generation_resume_reuse_and_integrity(self):
        directory = self.root / "cache"
        identity = {"config": "fixture"}
        calls = []
        real_generate = evaluate.generate_batch

        def interrupt(model, vae, fn, args, count, index):
            calls.append(index)
            if index == 1:
                raise RuntimeError("simulated interruption")
            return real_generate(model, vae, fn, args, count, index)

        with patch("evaluate.load_model", return_value=ToyModel()), patch("evaluate.load_vae", return_value=ToyVAE()):
            with patch("evaluate.generate_batch", side_effect=interrupt), self.assertRaises(RuntimeError):
                evaluate.generate_samples(self.entry, self.args, identity, directory)
            self.assertEqual(json.loads((directory / "progress.json").read_text())["completed"], 2)
            path, progress, reused = evaluate.generate_samples(self.entry, self.args, identity, directory)
            self.assertEqual(progress["completed"], 3)
            self.assertEqual(len(progress["nfe"]), 2)
            saved = np.load(path).copy()
            other, _, _ = evaluate.generate_samples(self.entry, self.args, identity, self.root / "fresh")
            np.testing.assert_array_equal(saved, np.load(other))
        with patch("evaluate.load_model", side_effect=AssertionError("Should reuse complete cache")):
            _, _, reused = evaluate.generate_samples(self.entry, self.args, identity, directory)
            self.assertTrue(reused)
        with self.assertRaises(ValueError):
            evaluate.generate_samples(self.entry, self.args, {"config": "different"}, directory)
        array = np.load(path, mmap_mode="r+")
        array[0, 0, 0, 0] ^= 1
        array.flush()
        with self.assertRaises(ValueError):
            evaluate.generate_samples(self.entry, self.args, identity, directory)

    @unittest.skipUnless(os.environ.get("SIT_TEST_METRICS") == "1", "Set SIT_TEST_METRICS=1 for the real Inception/FID smoke test")
    def test_real_metrics_and_reference_cache(self):
        # Small procedural RGB fixtures exercise the actual metric library,
        # including second-run cache deserialization; these are not quality scores.
        rng = np.random.default_rng(42)
        for index in range(4):
            folder = self.root / "reference" / str(index)
            folder.mkdir(parents=True)
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(folder / "image.png")
        reference = ReferenceImages(self.root / "reference", 256)
        path = self.root / "metric_samples.npy"
        np.save(path, np.stack([reference[i].permute(1, 2, 0).numpy() for i in range(4)]))
        self.args.metric_cache = str(self.root / "metric_cache")
        self.args.metric_batch_size = 2
        self.args.is_splits = 2
        first = calculate_metrics(path, reference, self.args)
        self.assertLess(abs(first["FID"]), 0.1)
        self.assertTrue(all(np.isfinite(value) for value in first.values()))
        second = calculate_metrics(path, reference, self.args)
        self.assertEqual(first, second)

    def test_failure_preserves_success_and_resumes(self):
        self.args.checkpoint.append(["full_linear", "10000", str(self.checkpoint)])

        def load(entry, args):
            if entry["variant"] == "full_linear":
                raise RuntimeError("simulated incompatible checkpoint")
            return ToyModel()

        with patch("evaluate.load_model", side_effect=load), patch("evaluate.load_vae", return_value=ToyVAE()):
            with self.assertRaises(RuntimeError):
                evaluate.main(self.args)
        reports = list((self.root / "results").glob("*/evaluation_results.json"))
        rows = json.loads(reports[0].read_text())["results"]
        self.assertEqual([r["status"] for r in rows], ["ok", "error"])
        self.assertTrue(reports[0].with_suffix(".csv").exists())
        self.args.checkpoint.pop()
        with patch("evaluate.load_model", side_effect=AssertionError("Should reuse completed result")):
            output = evaluate.main(self.args)
        self.assertTrue(json.loads((output / "evaluation_results.json").read_text())["results"][0]["result_reused"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
