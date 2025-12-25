from __future__ import annotations

import importlib
from typing import Iterable

import numpy as np


def _load_kernel():
    try:
        return importlib.import_module("simulation.kernels.tau_kernel")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "C++ tau-leaping extension not built. "
            "Build with `python simulation/kernels/setup.py build_ext --inplace`."
        ) from exc


def tau_leap_step(
    dt: float,
    age: float,
    mrna: Iterable[float],
    t_rep: Iterable[float],
    trans_prop: Iterable[float],
    deg_prop: Iterable[float],
    gamma_deg: Iterable[float],
    max_mrna_per_gene: int,
    rng_seed: int
) -> np.ndarray:
    kernel = _load_kernel()

    m_in = np.asarray(mrna, dtype=np.float64, order="C")
    t_rep_arr = np.asarray(t_rep, dtype=np.float64, order="C")
    trans_prop_arr = np.asarray(trans_prop, dtype=np.float64, order="C")
    deg_prop_arr = np.asarray(deg_prop, dtype=np.float64, order="C")
    gamma_deg_arr = np.asarray(gamma_deg, dtype=np.float64, order="C")
    if not (len(m_in) == len(t_rep_arr) == len(trans_prop_arr) == len(deg_prop_arr)== len(gamma_deg_arr)):
        raise ValueError("All gene-specific arrays must have the same length")
    m_out = np.empty_like(m_in)

    kernel.tau_leap_step(
        float(dt),
        float(age),
        m_in,
        m_out,
        t_rep_arr,
        trans_prop_arr,
        deg_prop_arr,
        gamma_deg_arr,
        m_in.size,
        int(max_mrna_per_gene),
        np.uint64(rng_seed)
    )
    return m_out.astype(np.int64, copy=False)
