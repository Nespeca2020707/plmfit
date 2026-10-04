import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedBCELoss, self).__init__()
        self.weights = weights

    def forward(self, pred, targets):
        EPS = 1e-12
        loss = -2 * (self.weights[0] * (targets * torch.log(pred + EPS)) + self.weights[1] * (1 - targets) * torch.log(1 - pred +EPS))
        return loss.mean()


class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Custom BCE loss that applies a sigmoid internally and ignores targets == -100.
    Use this if your model outputs raw logits (no sigmoid applied).
    """

    def __init__(self, reduction="mean", ignore_index=-100, pos_weight=None):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float)
            # Use register_buffer so this tensor is moved along with the model
            self.register_buffer("pos_weight", pos_weight_tensor)
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        """
        logits:   shape [batch_size, num_labels]
        targets:  shape [batch_size, num_labels], with ignore_index in some places
        """
        mask = targets != self.ignore_index

        # We must clamp "ignored" targets to something valid for BCE (e.g., 0)
        # so that the BCE won't produce NaNs.
        clamped_targets = targets.clone()
        clamped_targets[~mask] = 0.0  # or 1.0, doesn't matter, because they'll be masked out
        
        element_loss = F.binary_cross_entropy_with_logits(
            logits, clamped_targets,
            pos_weight=self.pos_weight,
            reduction="none"  # shape [batch_size, num_labels]
        )

        # Zero out ignored labels
        masked_loss = element_loss * mask 

        # 3) Reduce over the valid elements only
        if self.reduction == "mean":
            loss = masked_loss.sum() / mask.sum()
        elif self.reduction == "sum":
            loss = masked_loss.sum()
        else:
            # e.g., "none" – return the full matrix
            loss = masked_loss

        return loss

class MaskedFocalWithLogitsLoss(nn.Module):
    """
    Focal loss extended for use with missing labels. Ignores targets == ignore_index.
    Reference (original focal loss): https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2, reduction="mean", ignore_index=-100):
        """
        Args:
            alpha (float): Weighting factor for positives in [0, 1]. 
                           If alpha >= 0, alpha is applied to positive examples, 
                           (1 - alpha) to negatives.
            gamma (float): Exponent of the modulating factor (1 - p_t)**gamma to reduce 
                           the relative loss for well-classified examples.
            reduction (str): 'none' | 'mean' | 'sum'
            ignore_index (int): Specifies a target value that is ignored and does not 
                                contribute to the gradient.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (N, ...) Tensor of raw predictions (before sigmoid).
            targets: (N, ...) Tensor of same shape as logits, 
                     with 0 or 1 for valid labels, and ignore_index for missing.

        Returns:
            Loss tensor of shape (), or the same shape as inputs if reduction='none'.
        """
        # Build the mask for valid labels (those != ignore_index)
        mask = (targets != self.ignore_index)

        # Clone targets and clamp ignored entries to some valid label (0 or 1).
        # We'll use 0 here, but since they are masked out, it won't matter.
        clamped_targets = targets.clone()
        clamped_targets[~mask] = 0.0

        # Compute the standard binary cross-entropy per element, no reduction
        ce_loss = F.binary_cross_entropy_with_logits(logits, clamped_targets, reduction="none")

        # Compute p = sigmoid(logits)
        p = torch.sigmoid(logits)

        # p_t = p * t + (1 - p) * (1 - t)
        #    This is the probability assigned to the *true* class per element
        p_t = p * clamped_targets + (1 - p) * (1 - clamped_targets)

        # Apply the focal term: (1 - p_t)**gamma
        focal_term = (1.0 - p_t) ** self.gamma
        loss = ce_loss * focal_term

        # If alpha >= 0, compute alpha_t = alpha for positives, (1-alpha) for negatives
        if self.alpha >= 0:
            alpha_t = self.alpha * clamped_targets + (1.0 - self.alpha) * (1.0 - clamped_targets)
            loss = alpha_t * loss

        # Mask out the ignored entries
        loss = loss * mask

        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            # Average over the valid (non-ignored) entries
            valid_count = mask.sum()
            if valid_count == 0:
                # If there are no valid entries, return zero (or you could raise a warning).
                return loss.sum() * 0.0
            else:
                return loss.sum() / valid_count
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(
                f"Invalid 'reduction' mode: {self.reduction}. "
                "Supported: 'none', 'mean', 'sum'."
            )


class MaskedWeightedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy with class weights for token classification. Ignores targets == ignore_index.

    A class with a larger weight contributes more to the loss, which compensates for
    classes that are rare among the residues (binding or catalytic sites, for instance).
    """

    def __init__(self, class_weights=None, reduction="mean", ignore_index=-100):
        """
        Args:
            class_weights (list of float): Weight of each class. If None, all classes
                                           have the same weight.
            reduction (str): 'none' | 'mean' | 'sum'
            ignore_index (int): Specifies a target value that is ignored and does not
                                contribute to the gradient.
        """
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        if class_weights is not None:
            # Use register_buffer so this tensor is moved along with the model
            self.register_buffer("weight", torch.tensor(class_weights, dtype=torch.float))
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (N, C, L) Tensor of raw predictions (before softmax).
            targets: (N, L) Tensor of class indices, with ignore_index for missing labels.
        """
        return F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )


class MaskedFocalSoftmaxLoss(nn.Module):
    """
    Focal loss for token classification with two or more classes, where the model outputs
    one logit per class. Ignores targets == ignore_index.
    Reference (original focal loss): https://arxiv.org/abs/1708.02002
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean", ignore_index=-100):
        """
        Args:
            gamma (float): Exponent of the modulating factor (1 - p_t)**gamma to reduce
                           the relative loss for well-classified examples.
            alpha (float or list of float): Weighting of the classes. A list gives the
                           weight of each class. A float in [0, 1] is the weight of
                           class 1, the other classes having weight (1 - alpha): note
                           that a value below 0.5 down-weights class 1. If None (or
                           negative), all classes have the same weight.
            reduction (str): 'none' | 'mean' | 'sum'
            ignore_index (int): Specifies a target value that is ignored and does not
                                contribute to the gradient.
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        if isinstance(alpha, (list, tuple)):
            # Use register_buffer so this tensor is moved along with the model
            self.register_buffer("class_weights", torch.tensor(alpha, dtype=torch.float))
            self.alpha = None
        else:
            self.class_weights = None
            self.alpha = alpha if alpha is not None and alpha >= 0 else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (N, C, L) Tensor of raw predictions (before softmax).
            targets: (N, L) Tensor of class indices, with ignore_index for missing labels.

        Returns:
            Loss tensor of shape (), or (N, L) if reduction='none'.
        """
        mask = targets != self.ignore_index

        # Cross-entropy per token, no reduction: it is -log(p_t), where p_t is the
        # probability assigned to the true class, and 0 for the ignored tokens
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", ignore_index=self.ignore_index
        )
        p_t = torch.exp(-ce_loss)

        # Apply the focal term: (1 - p_t)**gamma
        loss = (1.0 - p_t) ** self.gamma * ce_loss

        if self.class_weights is not None:
            # Ignored tokens are given the weight of class 0, but they are masked out
            loss = self.class_weights[targets * mask] * loss
        elif self.alpha is not None:
            is_class_1 = (targets == 1).to(loss.dtype)
            loss = (self.alpha * is_class_1 + (1.0 - self.alpha) * (1.0 - is_class_1)) * loss

        # Mask out the ignored entries
        loss = loss * mask

        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            # Average over the valid (non-ignored) entries
            valid_count = mask.sum()
            if valid_count == 0:
                return loss.sum() * 0.0
            else:
                return loss.sum() / valid_count
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(
                f"Invalid 'reduction' mode: {self.reduction}. "
                "Supported: 'none', 'mean', 'sum'."
            )
