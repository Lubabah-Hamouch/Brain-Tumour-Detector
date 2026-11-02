"""Strip the served checkpoints down to what inference needs, for publishing as release assets.

Training checkpoints also hold optimizer, scheduler and AMP scaler state for resuming, which is
about two thirds of their size. The exported files keep the weights and the metadata the API
reads (training config, epoch, validation scores), so they load exactly like the originals. They
contain only tensors and plain Python values, so torch.load(..., weights_only=True) accepts them.

Usage:
    python -m src.export_weights          # writes models/release/ and prints SHA-256 hashes

The hashes printed here go into WEIGHTS in src/download_weights.py.
"""
import hashlib

import numpy as np
import torch

from src.config import PROJECT_ROOT
from src.download_weights import WEIGHTS

KEEP = ("model", "config", "epoch", "best_dice", "fold", "metrics", "classes")
OUT_DIR = PROJECT_ROOT / "models" / "release"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(value):
    """NumPy scalars -> Python numbers, so the file loads with torch.load(weights_only=True)."""
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(plain(v) for v in value)
    return value.item() if isinstance(value, np.generic) else value


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for asset, (target, _) in WEIGHTS.items():
        state = torch.load(PROJECT_ROOT / target, map_location="cpu", weights_only=False)
        slim = {k: state[k] if k == "model" else plain(state[k]) for k in KEEP if k in state}
        out = OUT_DIR / asset
        torch.save(slim, out)
        digest = sha256(out)
        lines.append(f"{digest}  {asset}")
        print(f"{asset}: {out.stat().st_size / 1e6:.1f} MB  sha256 {digest}")
    (OUT_DIR / "SHA256SUMS").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
