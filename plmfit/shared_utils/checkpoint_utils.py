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


def _check_settings(saved, current, ckpt_path, logger):
    """Compare the settings recorded in a checkpoint with those of the current run."""
    if saved is None:
        if logger is not None:
            logger.log(
                f"{ckpt_path} does not record the settings of the run that saved it: "
                f"make sure they were the same as in this run ({current})"
            )
        return
    differences = [
        f"{name} is {saved.get(name)!r} in the checkpoint and {value!r} in this run"
        for name, value in current.items()
        if saved.get(name) != value
    ]
    if differences:
        raise RuntimeError(
            f"The checkpoint {ckpt_path} was saved by a run with different settings: "
            f"{'; '.join(differences)}.\n{MISMATCH_HINT}"
        )


def load_finetuned_backbone(py_model, ckpt_path, logger=None, settings=None):
    """
    Initialize `py_model` with the fine-tuned backbone stored in a checkpoint.

    The checkpoint is the file saved in the experiment directory by a fine-tuning run.
    Every weight it contains is loaded except those of the prediction head: `py_model`
    gets the fine-tuned backbone, with its LoRA or bottleneck adapters if any, and keeps
    its own head.

    The backbone of `py_model` must be the same as the one that was saved: same PLM and
    layers, fine-tuning method, target layers and adapter configuration. A RuntimeError is
    raised, so that a run never continues from weights other than the ones requested, if
    - a weight of the backbone is missing from the checkpoint, has no counterpart in
      `py_model` or has a different shape;
    - one of the `settings` of the current run (a dictionary) differs from the value
      recorded in the checkpoint by the run that saved it. This catches what the weights
      cannot reveal, such as a different LoRA scaling factor.

    Returns the names of the weights of `py_model` that are not in the checkpoint and are
    therefore left as they were (the head, and the pooler if the saved model had none).
    """
    if os.path.isdir(ckpt_path):
        raise ValueError(
            f"{ckpt_path} is a directory, while a checkpoint file is expected, such as "
            "the 'best_model.ckpt' saved in the experiment directory of a fine-tuning run."
        )
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if settings is not None:
        _check_settings(checkpoint.get("backbone_settings"), settings, ckpt_path, logger)

    # The weights of the PLM are those of the `model` attribute of the Lightning module
    prefix = "model."
    backbone = {
        key[len(prefix) :]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix) and not _in_modules(key, HEAD_MODULES)
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
