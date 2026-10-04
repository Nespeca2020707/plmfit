"""
Tiny, randomly initialized ESM models for the tests.

They are instances of the model classes PLMFit uses for ESM-2, so they have the same
structure and parameter names as the real models, but nothing has to be downloaded and
they run in milliseconds on a CPU.
"""

import types

from transformers import EsmConfig

N_LAYERS = 2
HIDDEN_SIZE = 16


def tiny_esm_config(n_layers=N_LAYERS):
    return EsmConfig(
        vocab_size=33,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=n_layers,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        pad_token_id=1,
        mask_token_id=32,
        position_embedding_type="rotary",
        output_hidden_states=True,
    )


def as_plm(py_model):
    """Wrap a PyTorch model with the attributes of a PLMFit model that fine-tuners rely on."""
    n_layers = py_model.config.num_hidden_layers
    return types.SimpleNamespace(py_model=py_model, layer_to_use=n_layers - 1)
