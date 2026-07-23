"""Work around vLLM 0.25.1 Qwen warm-up with CPU-offloaded tensors.

The stock warm-up invokes Triton directly with parameters that the UVA
offloader keeps on CPU. Real model execution goes through the offloader. This
module is intended only as a bind-mounted replacement for the warm-up module
during the bounded local feasibility probe.
"""


def qwen_triton_warmup(runner, model_config) -> None:
    return None
