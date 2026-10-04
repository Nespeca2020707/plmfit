import os
import unittest
from unittest.mock import patch

import torch
from lightning import Trainer

from plmfit_runs import SPLIT, THREE_CLASSES, PlmfitRunTestCase


class TestCheckpointOptions(PlmfitRunTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.head_config = cls.write_head_config("head", n_classes=3)

    def fine_tune(self, experiment_name, **arguments):
        """Fine-tune with LoRA and return the experiment directory and the paths of all the checkpoints written."""
        saved_paths = []
        save_checkpoint = Trainer.save_checkpoint

        def spy(trainer, filepath, *args, **kwargs):
            saved_paths.append(str(filepath))
            return save_checkpoint(trainer, filepath, *args, **kwargs)

        with patch.object(Trainer, "save_checkpoint", spy):
            experiment_dir = self.run_plmfit(
                experiment_name,
                function="fine_tuning",
                ft_method="lora",
                data_type=THREE_CLASSES,
                split=SPLIT,
                head_config=self.head_config,
                **arguments,
            )
        self.assertTrue(os.path.isfile(f"{experiment_dir}/{experiment_name}_metrics.json"))
        return experiment_dir, saved_paths

    def test_best_checkpoint_is_kept_by_default(self):
        experiment_dir, saved_paths = self.fine_tune("default")
        best_ckpt = f"{experiment_dir}/lightning_logs/best_model.ckpt"
        self.assertIn(best_ckpt, saved_paths)
        self.assertIn("state_dict", torch.load(best_ckpt, map_location="cpu"))

    def test_keep_checkpoint_false(self):
        experiment_dir, saved_paths = self.fine_tune("no_ckpt", keep_checkpoint="False")
        best_ckpt = f"{experiment_dir}/lightning_logs/best_model.ckpt"
        self.assertIn(best_ckpt, saved_paths)
        self.assertFalse(os.path.exists(best_ckpt))

    def test_ckpt_staging_dir(self):
        staging_dir = f"{self.workdir}/staging"
        experiment_dir, saved_paths = self.fine_tune(
            "staged", ckpt_staging_dir=staging_dir
        )
        # While training, checkpoints are only written to the staging directory...
        self.assertTrue(saved_paths)
        for path in saved_paths:
            self.assertTrue(path.startswith(f"{staging_dir}/staged_"), path)
        # ...then the best one is moved to the experiment directory
        best_ckpt = f"{experiment_dir}/lightning_logs/best_model.ckpt"
        self.assertIn("state_dict", torch.load(best_ckpt, map_location="cpu"))
        self.assertEqual(os.listdir(staging_dir), [])

    def test_ckpt_staging_dir_without_keeping_checkpoint(self):
        staging_dir = f"{self.workdir}/staging_no_ckpt"
        experiment_dir, saved_paths = self.fine_tune(
            "staged_no_ckpt", ckpt_staging_dir=staging_dir, keep_checkpoint="False"
        )
        self.assertTrue(saved_paths)
        self.assertFalse(os.path.exists(f"{experiment_dir}/lightning_logs/best_model.ckpt"))
        self.assertEqual(os.listdir(staging_dir), [])


if __name__ == "__main__":
    unittest.main()
