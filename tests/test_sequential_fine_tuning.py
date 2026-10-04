import json
import os
import unittest

import torch

from plmfit_runs import PLM, SPLIT, THREE_CLASSES, TWO_CLASSES, PlmfitRunTestCase


def lora_weights(experiment_dir):
    checkpoint = torch.load(
        f"{experiment_dir}/lightning_logs/best_model.ckpt", map_location="cpu"
    )
    return {k: v for k, v in checkpoint["state_dict"].items() if "lora_" in k}


class TestSequentialFineTuning(PlmfitRunTestCase):
    """LoRA fine-tuning on a task, starting from a model fine-tuned with LoRA on another one."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        first_head = cls.write_head_config("first_head", n_classes=3)
        # With a null learning rate, the weights saved by a run are the ones it started from
        cls.second_head = cls.write_head_config(
            "second_head", n_classes=2, learning_rate=0, epochs=1
        )
        first_run = cls.run_plmfit(
            "first_task",
            function="fine_tuning",
            ft_method="lora",
            data_type=THREE_CLASSES,
            split=SPLIT,
            head_config=first_head,
        )
        cls.first_ckpt = f"{first_run}/lightning_logs/best_model.ckpt"
        cls.first_lora = lora_weights(first_run)

    def second_task(self, experiment_name, ft_method="lora", **arguments):
        return self.run_plmfit(
            experiment_name,
            function="fine_tuning",
            ft_method=ft_method,
            data_type=TWO_CLASSES,
            split=SPLIT,
            head_config=self.second_head,
            **arguments,
        )

    def test_first_task_trains_lora(self):
        # LoRA starts as a null update (B = 0): the first run must have changed it
        lora_b = [v for k, v in self.first_lora.items() if "lora_B" in k]
        self.assertTrue(lora_b)
        self.assertTrue(all(v.abs().max() > 0 for v in lora_b))

    def test_backbone_starts_from_the_checkpoint(self):
        experiment_dir = self.second_task("second_task", model_path=self.first_ckpt)
        second_lora = lora_weights(experiment_dir)
        self.assertEqual(second_lora.keys(), self.first_lora.keys())
        for key, value in second_lora.items():
            self.assertTrue(torch.equal(value, self.first_lora[key]), key)
        self.assertTrue(os.path.isfile(f"{experiment_dir}/second_task_metrics.json"))

    def test_backbone_starts_from_the_pretrained_model_without_model_path(self):
        experiment_dir = self.second_task("second_task_from_pretrained")
        for key, value in lora_weights(experiment_dir).items():
            if "lora_B" in key:
                self.assertEqual(value.abs().max(), 0, key)

    def lora_config(self):
        with open(f"{self.workdir}/config/peft/lora_config.json") as f:
            return json.load(f)

    def test_settings_are_recorded_in_the_checkpoint(self):
        checkpoint = torch.load(self.first_ckpt, map_location="cpu")
        self.assertEqual(
            checkpoint["backbone_settings"],
            {
                "plm": PLM,
                "layer": 5,  # the last of the 6 layers of the model
                "ft_method": "lora",
                "target_layers": "all",
                "r": self.lora_config()["r"],
                "lora_alpha": self.lora_config()["lora_alpha"],
            },
        )

    def check_rejected(self, experiment_name, message, **lora_changes):
        experiment_dir = self.second_task(
            experiment_name,
            model_path=self.first_ckpt,
            lora_config_path=self.write_lora_config(experiment_name, **lora_changes),
            expect_failure=True,
        )
        self.assertIn(message, self.read_log(experiment_dir))
        self.assertFalse(os.path.exists(f"{experiment_dir}/{experiment_name}_metrics.json"))

    def test_checkpoint_from_another_lora_rank_is_rejected(self):
        rank = self.lora_config()["r"]
        self.check_rejected(
            "other_rank", f"r is {rank} in the checkpoint and {rank // 2} in this run", r=rank // 2
        )

    def test_checkpoint_from_another_lora_scaling_is_rejected(self):
        # The weights of the checkpoint fit the model, but they would be scaled differently
        alpha = self.lora_config()["lora_alpha"]
        self.check_rejected(
            "other_scaling",
            f"lora_alpha is {alpha} in the checkpoint and {2 * alpha} in this run",
            lora_alpha=2 * alpha,
        )

    def test_checkpoint_from_another_method_is_rejected(self):
        experiment_dir = self.second_task(
            "second_task_full",
            ft_method="full",
            model_path=self.first_ckpt,
            expect_failure=True,
        )
        self.assertIn(
            "ft_method is 'lora' in the checkpoint and 'full' in this run",
            self.read_log(experiment_dir),
        )


if __name__ == "__main__":
    unittest.main()
