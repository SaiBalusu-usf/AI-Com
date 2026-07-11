"""Central seeding so every experiment is reproducible (§2 rule 3).

Torch is optional here on purpose: the core pipeline must run on machines
where the ML stack is not installed (see requirements-ml.txt).
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def set_seed(seed: int) -> int:
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


def rng_for(seed: int, *key: object) -> np.random.Generator:
    """Deterministic per-item generator, e.g. rng_for(seed, match_id, over, ball).

    Lets per-ball choices (template variant picks, sampling) stay stable even
    if iteration order changes.
    """
    material = f"{seed}|" + "|".join(str(k) for k in key)
    # sha256, not hash(): str hashing is salted per-process and would silently
    # break cross-run reproducibility.
    digest = hashlib.sha256(material.encode()).digest()
    child = np.random.SeedSequence(list(digest[:16]))
    return np.random.default_rng(child)
