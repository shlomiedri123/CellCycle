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


def _cumulative_trapz(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    t = np.asarray(t, dtype=float)
    if values.size == 0:
        return np.array([], dtype=float)
    if values.shape != t.shape:
        raise ValueError("values and t must have the same shape")
    if values.size < 2:
        return np.zeros_like(values)
    dt = np.diff(t)
    mid = 0.5 * (values[1:] + values[:-1])
    return np.concatenate(([0.0], np.cumsum(mid * dt)))


def _cycle_integral(
    gamma: float,
    nf_vec: np.ndarray,
    dt: float,
    g_func: Callable[[float], float],
    use_nf: bool,
) -> tuple[float, float]:
    if dt <= 0:
        raise ValueError("dt must be positive")
    nf_vec = np.asarray(nf_vec, dtype=float)
    if nf_vec.ndim != 1 or nf_vec.size == 0:
        raise ValueError("Nf vector must be a non-empty 1D array")
    tau = float(nf_vec.size) * float(dt)
    t_cycle = np.arange(nf_vec.size, dtype=float) * float(dt)
    g_cycle = np.array([float(g_func(float(ts))) for ts in t_cycle], dtype=float)
    g_end = float(g_func(0.0))
    if use_nf:
        integrand = np.exp(gamma * t_cycle) * g_cycle * nf_vec
        end_val = np.exp(gamma * tau) * g_end * float(nf_vec[0])
    else:
        integrand = np.exp(gamma * t_cycle) * g_cycle
        end_val = np.exp(gamma * tau) * g_end
    integrand = np.concatenate([integrand, [end_val]])
    return float(np.trapz(integrand, dx=float(dt))), tau


def _solve_regime_I(
    t: np.ndarray,
    gamma: float,
    Gamma: float,
    A: float,
    nf_vec: np.ndarray,
    dt: float,
    g_func: Callable[[float], float],
) -> np.ndarray:
    if A <= 0:
        raise ValueError("A must be positive for regime I")
    cycle_integral, tau = _cycle_integral(gamma, nf_vec, dt, g_func, use_nf=True)
    denom = 2.0 * np.exp(gamma * tau) - 1.0
    nf_vals = _nf_at_times(nf_vec, t, dt)
    g_vals = np.array([float(g_func(float(ts))) for ts in t], dtype=float)
    g_integrand = np.exp(gamma * t) * g_vals
    integral_to_t = _cumulative_trapz(g_integrand, t)
    prefactor = Gamma / A
    return prefactor * np.exp(-gamma * t) * (cycle_integral / denom + nf_vals * integral_to_t)


def _solve_regime_II(
    t: np.ndarray,
    gamma: float,
    Gamma: float,
    nf_vec: np.ndarray,
    dt: float,
    g_func: Callable[[float], float],
) -> np.ndarray:
    cycle_integral, tau = _cycle_integral(gamma, nf_vec, dt, g_func, use_nf=False)
    denom = 2.0 * np.exp(gamma * tau) - 1.0
    g_vals = np.array([float(g_func(float(ts))) for ts in t], dtype=float)
    integrand = np.exp(gamma * t) * g_vals
    integral_to_t = _cumulative_trapz(integrand, t)
    return Gamma * np.exp(-gamma * t) * (cycle_integral / denom + integral_to_t)


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
    regime: str | None = None,
) -> np.ndarray:
    """
    Exact periodic solution for mRNA profiles in regime I/II.

    Uses tau = len(nf_vec) * dt and the closed-form periodic integrals shown in
    the regime formulas; m0 is ignored for these periodic solutions.
    """
    t = np.asarray(t, dtype=float)
    if np.any(t < 0):
        raise ValueError("Time values must be non-negative")
    if g_func is None:
        g_func = lambda _t: 1.0
    regime_key = str(regime).strip().upper() if regime is not None else ""
    if regime_key in ("I", "1"):
        m_t = _solve_regime_I(t, gamma, Gamma, A, nf_vec, dt, g_func)
    elif regime_key in ("II", "2"):
        m_t = _solve_regime_II(t, gamma, Gamma, nf_vec, dt, g_func)
    else:
        raise ValueError("regime must be 'I' or 'II'")
    if round_int:
        return _round_int(m_t)
    return m_t
