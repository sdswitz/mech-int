from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from mechint.config import SAETrainConfig
from mechint.data import deterministic_split_indices, random_batch_iterator
from mechint.sae import SparseAutoEncoder


ROOT = Path(__file__).resolve().parent.parent


class PipelineTests(unittest.TestCase):
    def test_sae_shapes(self):
        sae = SparseAutoEncoder(d=4, m=12)
        x = torch.randn(3, 4)
        xhat, f = sae(x)
        self.assertEqual(xhat.shape, (3, 4))
        self.assertEqual(f.shape, (3, 12))

    def test_decoder_renorm(self):
        sae = SparseAutoEncoder(d=4, m=12)
        with torch.no_grad():
            sae.W_dec.mul_(3.0)
        sae.renormalize_decoder()
        norms = sae.W_dec.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_deterministic_split(self):
        train_a, val_a = deterministic_split_indices(100, 0.1, 42)
        train_b, val_b = deterministic_split_indices(100, 0.1, 42)
        self.assertTrue(torch.equal(train_a, train_b))
        self.assertTrue(torch.equal(val_a, val_b))

    def test_batch_iterator_device_transfer(self):
        activations = torch.randn(100, 4)
        iterator = random_batch_iterator(activations, 8, "cpu", seed=42)
        batch = next(iterator)
        self.assertEqual(batch.shape, (8, 4))
        self.assertEqual(batch.device.type, "cpu")

    def test_train_and_eval_scripts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            activations = tmp / "acts.pt"
            torch.save(torch.randn(256, 8), activations)

            train_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "train_sae.py"),
                "--activations-path",
                str(activations),
                "--activation-dim",
                "8",
                "--expansion",
                "2",
                "--epochs",
                "4",
                "--batch-size",
                "16",
                "--eval-batch-size",
                "32",
                "--runs-root",
                str(tmp / "runs"),
                "--run-name",
                "smoke",
                "--device",
                "cpu",
            ]
            subprocess.run(train_cmd, check=True, cwd=ROOT)

            run_dir = tmp / "runs" / "smoke"
            self.assertTrue((run_dir / "checkpoint.pt").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertIn("val_mse", summary)

            eval_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "eval_sae.py"),
                "--run-dir",
                str(run_dir),
                "--device",
                "cpu",
                "--output",
                str(run_dir / "eval.json"),
            ]
            subprocess.run(eval_cmd, check=True, cwd=ROOT)
            self.assertTrue((run_dir / "eval.json").exists())

    def test_compare_expansions_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ckpt_a = tmp / "a.pt"
            ckpt_b = tmp / "b.pt"
            torch.save(SparseAutoEncoder(d=8, m=16).state_dict(), ckpt_a)
            torch.save(SparseAutoEncoder(d=8, m=16).state_dict(), ckpt_b)

            output = tmp / "compare.csv"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "compare_expansions.py"),
                "--checkpoints",
                str(ckpt_a),
                str(ckpt_b),
                "--activation-dim",
                "8",
                "--output",
                str(output),
            ]
            subprocess.run(cmd, check=True, cwd=ROOT)
            self.assertTrue(output.exists())

    def test_legacy_train_wrapper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            activations = tmp / "acts.pt"
            torch.save(torch.randn(128, 8), activations)

            cmd = [
                sys.executable,
                str(ROOT / "train.py"),
                "--activations-path",
                str(activations),
                "--activation-dim",
                "8",
                "--expansion",
                "2",
                "--epochs",
                "2",
                "--batch-size",
                "16",
                "--eval-batch-size",
                "32",
                "--device",
                "cpu",
            ]
            subprocess.run(cmd, check=True, cwd=tmp)
            self.assertTrue((tmp / "saved_models" / "sae_model_2x.pt").exists())
            self.assertTrue((tmp / "training_metrics" / "training_metrics_2x.csv").exists())


if __name__ == "__main__":
    unittest.main()
