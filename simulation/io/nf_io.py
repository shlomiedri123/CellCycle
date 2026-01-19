"""Load deterministic Nf(t) vectors for RNAP limitation."""

from __future__ import annotations

import pathlib

import numpy as np


def load_nf_vector(path: str | pathlib.Path) -> np.ndarray:
    path = pathlib.Path(path)
    if path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix == ".csv":
        data = np.loadtxt(path, delimiter=",")
    else:
        data = np.loadtxt(path)

    vec = np.asarray(data, dtype=float).squeeze()
    if vec.ndim != 1:
        raise ValueError(f"Nf vector must be 1D; got shape {vec.shape} from {path}")
    if vec.size == 0:
        raise ValueError(f"Nf vector is empty: {path}")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"Nf vector contains NaN/inf values: {path}")
    if np.any(vec <= 0):
        raise ValueError(f"Nf vector must be positive: {path}")
    return vec
