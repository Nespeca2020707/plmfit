import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import torch
from lightning import Trainer
from torch import nn

import plmfit.models.downstream_heads as heads
from plmfit.functions.extract_embeddings import load_finetuned_plm
from plmfit.language_models.esm.modeling_esm import (
    PlmfitEsmForEmbdeddingsExtraction,
    PlmfitEsmForTokenClassification,
)
from plmfit.models.fine_tuners import LowRankAdaptationFineTuner
from plmfit.shared_utils import utils

from plmfit_runs import PLM, SPLIT, THREE_CLASSES, PlmfitRunTestCase
from tiny_esm import HIDDEN_SIZE, as_plm, tiny_esm_config


class TestLoadFinetunedPlm(unittest.TestCase):
    """The model used to extract embeddings gets the weights of the fine-tuned one."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.input_ids = torch.randint(4, 24, (3, 12))

    def fine_tuned_model(self, lora):
        """A tiny model for token classification that differs from its initialization as after fine-tuning."""
        py_model = PlmfitEsmForTokenClassification(tiny_esm_config())
        py_model.set_head(nn.Linear(HIDDEN_SIZE, 3))
        if lora:
            fine_tuner = LowRankAdaptationFineTuner(logger=MagicMock())
            py_model = fine_tuner.prepare_model(as_plm(py_model)).py_model
            for name, parameter in py_model.named_parameters():
                if "lora_" in name:
                    nn.init.normal_(parameter, std=0.1)
        return py_model.eval()

    def extraction_model(self, source, ft_method):
        path = os.path.join(self.tmp.name, "best_model.ckpt")
        state_dict = {f"model.{key}": value for key, value in source.state_dict().items()}
        torch.save({"state_dict": state_dict}, path)
        args = types.SimpleNamespace(
            plm="tiny_esm",
            model_path=path,
            ft_method=ft_method,
            target_layers="all",
            lora_config_path="peft/lora_config.json",
        )
        plm = as_plm(PlmfitEsmForEmbdeddingsExtraction(tiny_esm_config()))
        plm = load_finetuned_plm(plm, args, MagicMock())
        # Whatever the fine-tuning method, embeddings are extracted with a plain model
        self.assertIs(type(plm.py_model), PlmfitEsmForEmbdeddingsExtraction)
        return plm.py_model.eval()

    def last_hidden_state(self, py_model):
        with torch.no_grad():
            return py_model(self.input_ids).hidden_states[-1]

    def test_lora(self):
        source = self.fine_tuned_model(lora=True)
        model = self.extraction_model(source, "lora")

        # Weights: those of the backbone, plus the low-rank update where LoRA is applied
        lora_config = utils.load_config("peft/lora_config.json")
        scaling = lora_config["lora_alpha"] / lora_config["r"]
        saved = {
            key.removeprefix("base_model.model."): value
            for key, value in source.state_dict().items()
        }
        n_merged = 0
        for key, value in model.state_dict().items():
            module, parameter = key.rsplit(".", 1)
            if key.startswith("esm.pooler."):
                continue  # the fine-tuned model has no pooler
            if f"{module}.lora_A.default.weight" not in saved:
                self.assertTrue(torch.equal(value, saved[key]), key)
                continue
            expected = saved[f"{module}.base_layer.{parameter}"]
            if parameter == "weight":
                lora_a = saved[f"{module}.lora_A.default.weight"]
                lora_b = saved[f"{module}.lora_B.default.weight"]
                expected = expected + scaling * lora_b @ lora_a
                n_merged += 1
            self.assertTrue(torch.allclose(value, expected, atol=1e-6), key)
        self.assertEqual(n_merged, 3 * model.config.num_hidden_layers)  # query, key, value

        # Embeddings: those of the fine-tuned model, not those of the model before LoRA
        embeddings = self.last_hidden_state(model)
        self.assertTrue(
            torch.allclose(embeddings, self.last_hidden_state(source), atol=1e-4)
        )
        with source.disable_adapter():
            before_lora = self.last_hidden_state(source)
        self.assertFalse(torch.allclose(embeddings, before_lora, atol=1e-2))

    def test_full_fine_tuning(self):
        source = self.fine_tuned_model(lora=False)
        model = self.extraction_model(source, "full")
        saved = source.state_dict()
        for key, value in model.state_dict().items():
            if not key.startswith("esm.pooler."):
                self.assertTrue(torch.equal(value, saved[key]), key)
        self.assertTrue(
            torch.equal(self.last_hidden_state(model), self.last_hidden_state(source))
        )

    def test_unsupported_method(self):
        source = self.fine_tuned_model(lora=False)
        for ft_method in ("bottleneck_adapters", "feature_extraction"):
            with self.assertRaisesRegex(ValueError, "--ft_method"):
                self.extraction_model(source, ft_method)

    def test_wrong_method(self):
        # A checkpoint with LoRA cannot be read as a fully fine-tuned model, and vice versa
        with self.assertRaisesRegex(RuntimeError, "does not match the model"):
            self.extraction_model(self.fine_tuned_model(lora=True), "full")
        with self.assertRaisesRegex(RuntimeError, "does not match the model"):
            self.extraction_model(self.fine_tuned_model(lora=False), "lora")


class TestExtractFinetunedEmbeddings(PlmfitRunTestCase):
    """Embeddings extracted from a fine-tuned checkpoint through the command line."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.head_config = cls.write_head_config("head", n_classes=3, epochs=10)
        run = cls.run_plmfit(
            "fine_tuned",
            function="fine_tuning",
            ft_method="lora",
            data_type=THREE_CLASSES,
            split=SPLIT,
            head_config=cls.head_config,
        )
        cls.ckpt = f"{run}/lightning_logs/best_model.ckpt"

    def extract(self, experiment_name, **arguments):
        # Embeddings are extracted in half precision: use single precision here, so that
        # they can be compared exactly with the ones computed in the test
        module = sys.modules["plmfit.functions.extract_embeddings"]
        single_precision = lambda **kwargs: Trainer(**{**kwargs, "precision": 32})
        with patch.object(module, "Trainer", single_precision):
            experiment_dir = self.run_plmfit(
                experiment_name,
                function="extract_embeddings",
                data_type=THREE_CLASSES,
                reduction="mean",
                **arguments,
            )
        return torch.load(f"{experiment_dir}/{experiment_name}.pt")

    def reference_embeddings(self):
        """Embeddings computed with the fine-tuned model itself, rebuilt as in the fine-tuning run."""
        plm = utils.init_plm(PLM, None, task="token_classification")
        head_config = utils.load_config(f"training/{self.head_config}")
        plm.py_model.set_head(heads.init_head(head_config, plm.emb_layers_dim))
        plm = LowRankAdaptationFineTuner(logger=MagicMock()).prepare_model(plm)
        state_dict = torch.load(self.ckpt, map_location="cpu")["state_dict"]
        plm.py_model.load_state_dict(
            {k.removeprefix("model."): v for k, v in state_dict.items() if k.startswith("model.")}
        )
        tokens = plm.categorical_encode(utils.load_dataset(THREE_CLASSES))
        with torch.no_grad():
            return plm.py_model.eval()(tokens).hidden_states[-1].mean(dim=1)

    def test_without_checkpoint_embeddings_are_those_of_the_pretrained_model(self):
        plm = utils.init_plm(PLM, None, task="extract_embeddings")
        tokens = plm.categorical_encode(utils.load_dataset(THREE_CLASSES))
        with torch.no_grad():
            reference = plm.py_model.eval()(tokens).hidden_states[-1].mean(dim=1)
        pretrained = self.extract("pretrained_only_embs")
        self.assertTrue(torch.allclose(pretrained, reference, atol=1e-4))

    def test_embeddings_are_those_of_the_fine_tuned_model(self):
        fine_tuned = self.extract("fine_tuned_embs", model_path=self.ckpt, ft_method="lora")
        pretrained = self.extract("pretrained_embs")
        reference = self.reference_embeddings()
        self.assertEqual(fine_tuned.shape, reference.shape)
        self.assertTrue(torch.allclose(fine_tuned, reference, atol=1e-4))
        # Fine-tuning did change the embeddings: those of the pretrained model are different
        self.assertGreater((pretrained - reference).abs().max(), 0.1)

    def test_another_lora_scaling_is_rejected(self):
        # Merging the LoRA weights with another scaling factor would give other embeddings
        with open(f"{self.workdir}/config/peft/lora_config.json") as f:
            alpha = json.load(f)["lora_alpha"]
        experiment_dir = self.run_plmfit(
            "other_scaling",
            function="extract_embeddings",
            data_type=THREE_CLASSES,
            model_path=self.ckpt,
            ft_method="lora",
            lora_config_path=self.write_lora_config("other_scaling", lora_alpha=2 * alpha),
            expect_failure=True,
        )
        self.assertIn(
            f"lora_alpha is {alpha} in the checkpoint and {2 * alpha} in this run",
            self.read_log(experiment_dir),
        )
        self.assertFalse(os.path.exists(f"{experiment_dir}/other_scaling.pt"))

    def test_fine_tuning_method_is_required(self):
        experiment_dir = self.run_plmfit(
            "no_method",
            function="extract_embeddings",
            data_type=THREE_CLASSES,
            model_path=self.ckpt,
            expect_failure=True,
        )
        self.assertIn("--ft_method", self.read_log(experiment_dir))
        self.assertFalse(os.path.exists(f"{experiment_dir}/no_method.pt"))


if __name__ == "__main__":
    unittest.main()
