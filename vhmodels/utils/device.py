"""Shared device selection for PyTorch-backed models."""


def resolve_torch_device(torch_module, requested_device="auto"):
    """Return a torch device, resolving ``auto`` to CUDA or CPU."""
    if requested_device == "auto":
        requested_device = (
            "cuda:0" if torch_module.cuda.is_available() else "cpu"
        )
    return torch_module.device(requested_device)
