"""
DeepSpeed is an optional dependency of PLMFit.

It is only needed to train on GPUs with the DeepSpeed ZeRO strategy. When the package
cannot be imported (it does not build on machines without the CUDA toolkit, for instance)
PLMFit falls back to the default Lightning strategy on a single device.
"""

import torch

try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
    from deepspeed.runtime.zero.stage3 import (
        estimate_zero3_model_states_mem_needs_all_live,
    )

    DEEPSPEED_AVAILABLE = True
except ImportError:
    DeepSpeedCPUAdam = None
    FlopsProfiler = None
    estimate_zero3_model_states_mem_needs_all_live = None
    DEEPSPEED_AVAILABLE = False


def use_deepspeed():
    """DeepSpeed is used whenever it is installed and a GPU is available."""
    return DEEPSPEED_AVAILABLE and torch.cuda.is_available()
