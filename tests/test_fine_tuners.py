import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from plmfit.language_models.esm.modeling_esm import PlmfitEsmForTokenClassification
from plmfit.models.fine_tuners import LowRankAdaptationFineTuner
from plmfit.shared_utils import utils

from tiny_esm import as_plm, tiny_esm_config


def lora_ranks(py_model):
    return {p.shape[0] for name, p in py_model.named_parameters() if "lora_A" in name}


class TestLoraConfigPath(unittest.TestCase):
    def test_default_lora_config(self):
        fine_tuner = LowRankAdaptationFineTuner(logger=MagicMock())
        plm = fine_tuner.prepare_model(
            as_plm(PlmfitEsmForTokenClassification(tiny_esm_config()))
        )
        default_rank = utils.load_config("peft/lora_config.json")["r"]
        self.assertEqual(lora_ranks(plm.py_model), {default_rank})

    def test_custom_lora_config(self):
        lora_config = {
            "r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "bias": "none",
            "modules_to_save": ["classifier"],
        }
        default_config_dir = utils.config_dir
        with tempfile.TemporaryDirectory() as config_dir:
            os.makedirs(f"{config_dir}/peft")
            with open(f"{config_dir}/peft/lora_r2.json", "w") as f:
                json.dump(lora_config, f)
            utils.set_config_dir(config_dir)
            try:
                fine_tuner = LowRankAdaptationFineTuner(
                    logger=MagicMock(), lora_config_path="peft/lora_r2.json"
                )
            finally:
                utils.set_config_dir(default_config_dir)
        plm = fine_tuner.prepare_model(
            as_plm(PlmfitEsmForTokenClassification(tiny_esm_config()))
        )
        self.assertEqual(lora_ranks(plm.py_model), {2})


if __name__ == "__main__":
    unittest.main()
