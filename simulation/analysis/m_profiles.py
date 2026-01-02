"""Exact solution helpers for mRNA profiles using provided Nf(t) vectors.

All rates/times are in minutes and per-minute units. Nf is provided as a
single-cell-cycle vector and indexed by integer time steps (t = k * dt), with
periodic wrap-around. Rounding, when enabled, applies abs then half-up to
integers to match plotting conventions.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _round_int(arr: np.ndarray) -> np.ndarray:
    arr = np.abs(arr)
    return np.floor(arr + 0.5).astype(int)


def _nf_at_times(nf_vec: np.ndarray, t: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0:
        raise ValueError("dt must be positive")
    nf_vec = np.asarray(nf_vec, dtype=float)
    if nf_vec.ndim != 1 or nf_vec.size == 0:
        raise ValueError("Nf vector must be a non-empty 1D array")
    t = np.asarray(t, dtype=float)
    if np.any(t < 0):
        raise ValueError("Time values must be non-negative")
    idx = np.floor(t / dt).astype(int)
    if idx.size == 0:
        return np.array([], dtype=float)
    idx = idx % nf_vec.size
    return nf_vec[idx]


def solve_mRNA_exact(
    t: np.ndarray,
    m0: float,
    gamma: float,
    Gamma: float,
    A: float,
    nf_vec: np.ndarray,
    dt: float,
    g_func: Callable[[float], float] | None = None,
    round_int: bool = False,
) -> np.ndarray:
    """
    Exact solution for time-dependent Nf(t):
        dm/dt = g(t)*Gamma * Nf(t)/(Nf(t)+A) - gamma*m

    Integrating factor:
        m(t) = exp(-gamma*t) * [ m0 + ∫_0^t exp(gamma*s) * g(s)*Gamma*Nf(s)/(Nf(s)+A) ds ]
    """
    t = np.asarray(t, dtype=float)
    if g_func is None:
        g_func = lambda _t: 1.0

    nf_vals = _nf_at_times(nf_vec, t, dt)
    g_vals = np.array([float(g_func(float(ts))) for ts in t], dtype=float)
    integrand = np.exp(gamma * t) * g_vals * Gamma * nf_vals / (nf_vals + A)
    if len(t) >= 2:
        dt = np.diff(t)
        mid = 0.5 * (integrand[1:] + integrand[:-1])
        integral = np.concatenate(([0.0], np.cumsum(mid * dt)))
    else:
        integral = np.zeros_like(t)
    m_t = np.exp(-gamma * t) * (m0 + integral)
    if round_int:
        return _round_int(m_t)
    return m_t
