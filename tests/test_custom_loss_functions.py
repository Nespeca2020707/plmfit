import os
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from plmfit.models.lightning_model import LightningModel
from plmfit.shared_utils.custom_loss_functions import (
    MaskedFocalSoftmaxLoss,
    MaskedWeightedCrossEntropyLoss,
)

from plmfit_runs import SPLIT, TWO_CLASSES, PlmfitRunTestCase


class TokenLossTestCase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(4, 3, 10)  # (sequences, classes, length)
        self.targets = torch.randint(0, 3, (4, 10))
        self.targets[:, 7:] = -100  # padding
        self.valid = self.targets != -100

        # Quantities the losses are defined from, computed token by token
        probabilities = torch.softmax(self.logits, dim=1)
        true_class = self.targets.clamp(min=0).unsqueeze(1)
        self.p_t = probabilities.gather(1, true_class).squeeze(1)[self.valid]
        self.classes = self.targets[self.valid]

    def assert_close(self, value, expected):
        self.assertTrue(torch.allclose(value, expected, atol=1e-6), (value, expected))

    def check_padding_is_ignored(self, loss_function):
        logits = self.logits.clone()
        logits[:, :, 7:] += 5 * torch.randn(4, 3, 3)
        self.assert_close(
            loss_function(logits, self.targets), loss_function(self.logits, self.targets)
        )


class TestMaskedWeightedCrossEntropyLoss(TokenLossTestCase):
    def test_without_weights(self):
        loss = MaskedWeightedCrossEntropyLoss()(self.logits, self.targets)
        self.assert_close(loss, -torch.log(self.p_t).mean())

    def test_class_weights(self):
        weights = torch.tensor([1.0, 30.0, 5.0])
        loss = MaskedWeightedCrossEntropyLoss(class_weights=weights.tolist())(
            self.logits, self.targets
        )
        # Weighted average of the cross-entropy of the tokens, as torch.nn.CrossEntropyLoss
        token_weights = weights[self.classes]
        expected = (token_weights * -torch.log(self.p_t)).sum() / token_weights.sum()
        self.assert_close(loss, expected)

    def test_padding_is_ignored(self):
        self.check_padding_is_ignored(MaskedWeightedCrossEntropyLoss([1.0, 30.0, 5.0]))


class TestMaskedFocalSoftmaxLoss(TokenLossTestCase):
    def focal(self, gamma):
        return (1 - self.p_t) ** gamma * -torch.log(self.p_t)

    def test_gamma_zero_is_cross_entropy(self):
        loss = MaskedFocalSoftmaxLoss(gamma=0)(self.logits, self.targets)
        self.assert_close(loss, -torch.log(self.p_t).mean())

    def test_focal_term(self):
        loss = MaskedFocalSoftmaxLoss(gamma=2.0)(self.logits, self.targets)
        self.assert_close(loss, self.focal(2.0).mean())

    def test_alpha_as_weight_of_class_1(self):
        loss = MaskedFocalSoftmaxLoss(gamma=2.0, alpha=0.75)(self.logits, self.targets)
        alpha_t = torch.where(self.classes == 1, 0.75, 0.25)
        self.assert_close(loss, (alpha_t * self.focal(2.0)).mean())

    def test_alpha_per_class(self):
        alpha = torch.tensor([0.2, 1.0, 3.0])
        loss = MaskedFocalSoftmaxLoss(gamma=1.0, alpha=alpha.tolist())(
            self.logits, self.targets
        )
        self.assert_close(loss, (alpha[self.classes] * self.focal(1.0)).mean())

    def test_negative_alpha_disables_class_weighting(self):
        loss = MaskedFocalSoftmaxLoss(gamma=2.0, alpha=-1)(self.logits, self.targets)
        self.assert_close(loss, self.focal(2.0).mean())

    def test_padding_is_ignored(self):
        self.check_padding_is_ignored(MaskedFocalSoftmaxLoss(alpha=0.75))
        self.check_padding_is_ignored(MaskedFocalSoftmaxLoss(alpha=[0.2, 1.0, 3.0]))

    def test_reductions(self):
        per_token = MaskedFocalSoftmaxLoss(reduction="none")(self.logits, self.targets)
        self.assertEqual(per_token.shape, self.targets.shape)
        self.assertEqual(per_token[~self.valid].abs().max(), 0)
        self.assert_close(
            MaskedFocalSoftmaxLoss(reduction="sum")(self.logits, self.targets),
            per_token.sum(),
        )

    def test_no_valid_token(self):
        logits = self.logits.clone().requires_grad_()
        loss = MaskedFocalSoftmaxLoss()(logits, torch.full((4, 10), -100))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_saturated_predictions_have_finite_gradients(self):
        logits = (100 * self.logits).requires_grad_()
        loss = MaskedFocalSoftmaxLoss(alpha=0.25)(logits, self.targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())


class TestLossConfiguration(unittest.TestCase):
    """The losses are selected and configured from the training parameters of the head."""

    def loss_function(self, **training_parameters):
        model = nn.Linear(1, 1)
        model.task = "token_classification"
        return LightningModel(model, {"no_classes": 2, **training_parameters}).loss_function

    def test_masked_weighted_ce(self):
        loss = self.loss_function(loss_f="masked_weighted_ce", class_weights=[1.0, 30.0])
        self.assertIsInstance(loss, MaskedWeightedCrossEntropyLoss)
        self.assertEqual(loss.weight.tolist(), [1.0, 30.0])
        self.assertIsNone(self.loss_function(loss_f="masked_weighted_ce").weight)

    def test_masked_focal_softmax(self):
        loss = self.loss_function(loss_f="masked_focal_softmax", gamma=1.5, alpha=0.75)
        self.assertIsInstance(loss, MaskedFocalSoftmaxLoss)
        self.assertEqual((loss.gamma, loss.alpha), (1.5, 0.75))
        loss = self.loss_function(loss_f="masked_focal_softmax", alpha=[1.0, 30.0])
        self.assertEqual(loss.class_weights.tolist(), [1.0, 30.0])
        loss = self.loss_function(loss_f="masked_focal_softmax")
        self.assertEqual((loss.gamma, loss.alpha, loss.class_weights), (2.0, None, None))


class TestFineTuningWithTokenLosses(PlmfitRunTestCase):
    def fine_tune(self, name, **training_parameters):
        head_config = self.write_head_config(name, n_classes=2, **training_parameters)
        experiment_dir = self.run_plmfit(
            name,
            function="fine_tuning",
            ft_method="lora",
            data_type=TWO_CLASSES,
            split=SPLIT,
            head_config=head_config,
        )
        self.assertTrue(os.path.isfile(f"{experiment_dir}/{name}_metrics.json"))

    def test_masked_weighted_ce(self):
        self.fine_tune("weighted_ce", loss_f="masked_weighted_ce", class_weights=[1.0, 3.0])

    def test_masked_focal_softmax(self):
        self.fine_tune("focal", loss_f="masked_focal_softmax", gamma=2.0, alpha=0.75)


if __name__ == "__main__":
    unittest.main()
