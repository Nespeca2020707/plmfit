"""Reuse the weights of a checkpoint saved by a PLMFit fine-tuning run."""

import os

import torch

# Modules holding the prediction head, which is specific to the task a model was fine-tuned
# on. The same names are listed as `modules_to_save` in the PEFT configuration files.
HEAD_MODULES = ("classifier", "mlm", "lm_head")

# Modules and buffers that only exist for some tasks or library versions (token
# classification models have no pooler, for instance): they may be on one side only.
OPTIONAL_MODULES = ("pooler", "position_ids")

MISMATCH_HINT = (
    "The model has to be built with the same --plm, --layer, --ft_method, --target_layers "
    "and LoRA or adapter configuration as the run that saved the checkpoint."
)


def _in_modules(key, module_names):
    """Whether the weight named `key` belongs to a module with one of the given names."""
    return any(name in module_names for name in key.split("."))


def _preview(keys, limit=5):
    return ", ".join(keys[:limit]) + (", ..." if len(keys) > limit else "")


def load_model_state_dict(ckpt_path):
    """
    Read the weights of the fine-tuned PLM from a PLMFit checkpoint.

    The checkpoint is the file saved in the experiment directory by a fine-tuning run. The
    weights are returned with the names they have in the PyTorch model, i.e. without the
    prefix of the Lightning module that wraps it.
    """
    if os.path.isdir(ckpt_path):
        raise ValueError(
            f"{ckpt_path} is a directory, while a checkpoint file is expected, such as "
            "the 'best_model.ckpt' saved in the experiment directory of a fine-tuning run."
        )
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    prefix = "model."
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def load_finetuned_backbone(py_model, ckpt_path, logger=None):
    """
    Initialize `py_model` with the fine-tuned backbone stored in a checkpoint.

    Every weight of the checkpoint is loaded except those of the prediction head: `py_model`
    gets the fine-tuned backbone, with its LoRA or bottleneck adapters if any, and keeps its
    own head.

    The backbone of `py_model` must be the same as the one that was saved: same PLM and
    layers, fine-tuning method, target layers and adapter configuration. If a weight of the
    backbone is missing from the checkpoint, has no counterpart in `py_model` or has a
    different shape, a RuntimeError is raised, so that a run never continues from weights
    other than the ones requested.

    Returns the names of the weights of `py_model` that are not in the checkpoint and are
    therefore left as they were (the head, and the pooler if the saved model had none).
    """
    state_dict = load_model_state_dict(ckpt_path)
    backbone = {
        key: value
        for key, value in state_dict.items()
        if not _in_modules(key, HEAD_MODULES)
    }

    try:
        not_loaded, unexpected = py_model.load_state_dict(backbone, strict=False)
    except RuntimeError as error:  # raised by PyTorch when the shape of a weight differs
        raise RuntimeError(f"{error}\n{MISMATCH_HINT}") from error

    missing = [
        key for key in not_loaded if not _in_modules(key, HEAD_MODULES + OPTIONAL_MODULES)
    ]
    unexpected = [key for key in unexpected if not _in_modules(key, OPTIONAL_MODULES)]
    if missing or unexpected:
        raise RuntimeError(
            f"The checkpoint {ckpt_path} does not match the model.\n"
            f"Weights of the model that are not in the checkpoint ({len(missing)}): "
            f"{_preview(missing)}\n"
            f"Weights of the checkpoint that are not in the model ({len(unexpected)}): "
            f"{_preview(unexpected)}\n"
            f"{MISMATCH_HINT}"
        )

    if logger is not None:
        logger.log(
            f"Loaded {len(py_model.state_dict()) - len(not_loaded)} tensors from "
            f"{ckpt_path}; not in the checkpoint and left as initialized: {not_loaded}"
        )
    return list(not_loaded)
