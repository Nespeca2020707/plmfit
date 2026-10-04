import json
import unittest

import torch

from plmfit.models.lightning_model import token_class_probabilities

from plmfit_runs import SPLIT, THREE_CLASSES, PlmfitRunTestCase


class TestTokenClassProbabilities(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = 3 * torch.randn(2, 3, 5)  # (sequences, classes, length)
        self.probabilities = torch.softmax(self.logits, dim=1).permute(0, 2, 1)

    def test_logits(self):
        probabilities = token_class_probabilities(self.logits)
        self.assertEqual(probabilities.shape, (2, 5, 3))
        self.assertTrue(torch.allclose(probabilities, self.probabilities))

    def test_outputs_of_a_head_with_softmax_activation(self):
        # They are probabilities already and must not go through another softmax
        outputs = torch.softmax(self.logits, dim=1)
        self.assertTrue(
            torch.allclose(token_class_probabilities(outputs), self.probabilities)
        )

    def test_half_precision(self):
        for outputs in (self.logits, torch.softmax(self.logits, dim=1)):
            probabilities = token_class_probabilities(outputs.half())
            self.assertTrue(torch.allclose(probabilities, self.probabilities, atol=1e-2))


class TestSavedProbabilities(PlmfitRunTestCase):
    def predictions(self, name, **training_parameters):
        head_config = self.write_head_config(name, n_classes=3, **training_parameters)
        experiment_dir = self.run_plmfit(
            name,
            function="fine_tuning",
            ft_method="lora",
            data_type=THREE_CLASSES,
            split=SPLIT,
            head_config=head_config,
        )
        with open(f"{experiment_dir}/{name}_metrics.json") as f:
            return json.load(f)["pred_data"]

    def test_probabilities_are_saved_on_request(self):
        predictions = self.predictions("with_probabilities", save_probabilities=True)
        preds = torch.tensor(predictions["preds"])
        probs = torch.tensor(predictions["probs"])
        # One probability per class for each token, consistent with the predicted class
        self.assertEqual(probs.shape, (*preds.shape, 3))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(preds.shape)))
        self.assertTrue(torch.equal(probs.argmax(dim=-1), preds))

    def test_probabilities_are_not_saved_by_default(self):
        self.assertNotIn("probs", self.predictions("without_probabilities"))


if __name__ == "__main__":
    unittest.main()
