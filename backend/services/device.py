import torch


def resolve_device(setting):
    """'auto' picks CUDA when available; an explicit 'cuda' fails loudly if CUDA is missing."""
    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if setting == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("BT_DEVICE=cuda but CUDA is not available; use BT_DEVICE=cpu or auto")
    if setting not in ("cuda", "cpu"):
        raise RuntimeError(f"BT_DEVICE must be auto, cuda or cpu, got {setting!r}")
    return torch.device(setting)
