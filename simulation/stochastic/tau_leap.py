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


def tau_leap_batch(
    dt: float,
    ages: Iterable[float],
    mrna: Iterable[Iterable[float]],
    t_rep: Iterable[float],
    trans_prop: Iterable[Iterable[float]],
    deg_prop: Iterable[Iterable[float]],
    gamma_deg: Iterable[float],
    max_mrna_per_gene: int,
    rng_seeds: Iterable[int],
) -> np.ndarray:
    kernel = _load_kernel()

    ages_arr = np.asarray(ages, dtype=np.float64, order="C")
    m_in = np.asarray(mrna, dtype=np.float64, order="C")
    t_rep_arr = np.asarray(t_rep, dtype=np.float64, order="C")
    trans_prop_arr = np.asarray(trans_prop, dtype=np.float64, order="C")
    deg_prop_arr = np.asarray(deg_prop, dtype=np.float64, order="C")
    gamma_deg_arr = np.asarray(gamma_deg, dtype=np.float64, order="C")
    rng_seeds_arr = np.asarray(rng_seeds, dtype=np.uint64, order="C")

    if m_in.ndim != 2:
        raise ValueError("mRNA input must be a 2-D (n_cells, n_genes) array")
    n_cells, G = m_in.shape
    if ages_arr.ndim != 1 or ages_arr.shape[0] != n_cells:
        raise ValueError("ages must be a 1-D array of length n_cells")
    if trans_prop_arr.shape != (n_cells, G) or deg_prop_arr.shape != (n_cells, G):
        raise ValueError("propensity arrays must have shape (n_cells, n_genes)")
    if t_rep_arr.ndim != 1 or t_rep_arr.shape[0] != G:
        raise ValueError("t_rep must be a 1-D array of length n_genes")
    if gamma_deg_arr.ndim != 1 or gamma_deg_arr.shape[0] != G:
        raise ValueError("gamma_deg must be a 1-D array of length n_genes")
    if rng_seeds_arr.ndim != 1 or rng_seeds_arr.shape[0] != n_cells:
        raise ValueError("rng_seeds must be a 1-D array of length n_cells")

    m_out = np.empty_like(m_in)
    kernel.tau_leap_step_batch(
        float(dt),
        ages_arr,
        m_in,
        m_out,
        t_rep_arr,
        trans_prop_arr,
        deg_prop_arr,
        gamma_deg_arr,
        int(max_mrna_per_gene),
        rng_seeds_arr,
    )
    return m_out.astype(np.int64, copy=False)
