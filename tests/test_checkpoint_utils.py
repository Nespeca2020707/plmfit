import os
import tempfile
import unittest
from unittest.mock import MagicMock

import torch
from torch import nn

from plmfit.language_models.esm.modeling_esm import (
    PlmfitEsmForMaskedLM,
    PlmfitEsmForSequenceClassification,
    PlmfitEsmForTokenClassification,
)
from plmfit.models.fine_tuners import LowRankAdaptationFineTuner
from plmfit.shared_utils.checkpoint_utils import load_finetuned_backbone

from tiny_esm import HIDDEN_SIZE, as_plm, tiny_esm_config


def build_model(model_class, n_outputs=None, lora=False, lora_rank=None, n_layers=2):
    """A tiny model for a task, optionally with LoRA, where every weight is random."""
    py_model = model_class(tiny_esm_config(n_layers))
    if n_outputs is not None:
        py_model.set_head(nn.Linear(HIDDEN_SIZE, n_outputs))
    if lora:
        fine_tuner = LowRankAdaptationFineTuner(logger=MagicMock())
        if lora_rank is not None:
            fine_tuner.peft_config.r = lora_rank
        py_model = fine_tuner.prepare_model(as_plm(py_model)).py_model
    # Some weights have a constant initial value (zero for the LoRA B matrices): randomize
    # them all, so that weights coming from different models can be told apart
    for parameter in py_model.parameters():
        nn.init.normal_(parameter)
    return py_model


class TestLoadFinetunedBackbone(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def save_checkpoint(self, py_model):
        """Save a model as PLMFit does: a Lightning module holding the PLM as `model`."""
        state_dict = {f"model.{key}": value for key, value in py_model.state_dict().items()}
        state_dict["loss_function.weight"] = torch.ones(2)
        path = os.path.join(self.tmp.name, "best_model.ckpt")
        torch.save({"state_dict": state_dict}, path)
        return path

    def check_transfer(self, source, target, untouched):
        """
        Load the checkpoint of `source` into `target`. The weights whose name starts with
        one of the `untouched` prefixes must keep their value, all the others must be the
        ones of `source`.
        """
        initial = {key: value.clone() for key, value in target.state_dict().items()}
        not_loaded = load_finetuned_backbone(target, self.save_checkpoint(source))

        saved = source.state_dict()
        transferred = []
        for key, value in target.state_dict().items():
            if key.startswith(untouched):
                self.assertTrue(torch.equal(value, initial[key]), f"{key} has changed")
            else:
                self.assertTrue(torch.equal(value, saved[key]), f"{key} was not loaded")
                transferred.append(key)
        self.assertEqual(
            sorted(not_loaded), sorted(k for k in initial if k.startswith(untouched))
        )
        return transferred

    def test_lora_token_classification_to_regression(self):
        source = build_model(PlmfitEsmForTokenClassification, n_outputs=3, lora=True)
        target = build_model(PlmfitEsmForSequenceClassification, n_outputs=1, lora=True)
        # The saved model has no pooler: the one of the new model is left as it is
        transferred = self.check_transfer(
            source,
            target,
            untouched=("base_model.model.classifier.", "base_model.model.esm.pooler."),
        )
        self.assertTrue(any("lora_A" in key for key in transferred))
        self.assertTrue(any("lora_B" in key for key in transferred))

    def test_lora_regression_to_token_classification(self):
        source = build_model(PlmfitEsmForSequenceClassification, n_outputs=1, lora=True)
        target = build_model(PlmfitEsmForTokenClassification, n_outputs=3, lora=True)
        # The pooler of the saved model has no place in the new model and is ignored
        self.check_transfer(source, target, untouched=("base_model.model.classifier.",))

    def test_lora_regression_to_regression(self):
        source = build_model(PlmfitEsmForSequenceClassification, n_outputs=1, lora=True)
        target = build_model(PlmfitEsmForSequenceClassification, n_outputs=1, lora=True)
        # Same head shape, but the head is not transferred. The pooler, part of both, is.
        transferred = self.check_transfer(
            source, target, untouched=("base_model.model.classifier.",)
        )
        self.assertTrue(any(".pooler." in key for key in transferred))

    def test_full_fine_tuning(self):
        # Without adapters the head is a top-level module: it must not be transferred even
        # if its shape is the same in the two models
        source = build_model(PlmfitEsmForTokenClassification, n_outputs=3)
        target = build_model(PlmfitEsmForTokenClassification, n_outputs=3)
        self.check_transfer(source, target, untouched=("classifier.",))

    def test_masked_language_model_to_token_classification(self):
        source = build_model(PlmfitEsmForMaskedLM)
        target = build_model(PlmfitEsmForTokenClassification, n_outputs=3)
        self.assertTrue(any(key.startswith("lm_head.") for key in source.state_dict()))
        self.check_transfer(source, target, untouched=("classifier.",))

    def check_mismatch(self, source, target, message):
        with self.assertRaisesRegex(RuntimeError, message):
            load_finetuned_backbone(target, self.save_checkpoint(source))

    def test_different_lora_rank(self):
        self.check_mismatch(
            build_model(PlmfitEsmForTokenClassification, n_outputs=3, lora=True),
            build_model(PlmfitEsmForTokenClassification, n_outputs=3, lora=True, lora_rank=4),
            "size mismatch",
        )

    def test_different_fine_tuning_method(self):
        self.check_mismatch(
            build_model(PlmfitEsmForTokenClassification, n_outputs=3),
            build_model(PlmfitEsmForTokenClassification, n_outputs=3, lora=True),
            "does not match the model",
        )
        self.check_mismatch(
            build_model(PlmfitEsmForTokenClassification, n_outputs=3, lora=True),
            build_model(PlmfitEsmForTokenClassification, n_outputs=3),
            "does not match the model",
        )

    def test_different_number_of_layers(self):
        self.check_mismatch(
            build_model(PlmfitEsmForTokenClassification, n_outputs=3, n_layers=2),
            build_model(PlmfitEsmForTokenClassification, n_outputs=3, n_layers=3),
            r"not in the checkpoint \([1-9]",
        )
        self.check_mismatch(
            build_model(PlmfitEsmForTokenClassification, n_outputs=3, n_layers=3),
            build_model(PlmfitEsmForTokenClassification, n_outputs=3, n_layers=2),
            r"not in the model \([1-9]",
        )

    def test_checkpoint_directory(self):
        target = build_model(PlmfitEsmForTokenClassification, n_outputs=3)
        with self.assertRaisesRegex(ValueError, "is a directory"):
            load_finetuned_backbone(target, self.tmp.name)


if __name__ == "__main__":
    unittest.main()
