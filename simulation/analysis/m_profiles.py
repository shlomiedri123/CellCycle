"""Shared helpers for mRNA steady-state and time-dependent profiles.

All rates/times are in minutes and per-minute units. Nf is clamped to a small
positive floor to avoid non-physical values. Rounding, when enabled, applies
abs then half-up to integers to match plotting conventions.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

FLOOR_NF = 1e-9


def clamp_nf(val: float) -> float:
    return max(float(val), FLOOR_NF)


def make_nf_getter(nf_global) -> Callable[[float], float]:
    if callable(nf_global):
        return lambda t: clamp_nf(nf_global(float(t)))
    const_nf = clamp_nf(float(nf_global))
    return lambda _t, _c=const_nf: _c


def _round_int(arr: np.ndarray) -> np.ndarray:
    arr = np.abs(arr)
    return np.floor(arr + 0.5).astype(int)


def steady_mean_constant_nf(gene, copies: int, nf: float, round_int: bool = False) -> float | int:
    nf = clamp_nf(nf)
    denom = 1.0 + (gene.k_off_rnap + gene.Gamma_esc) / (gene.k_on_rnap * nf)
    mean_m = (copies * (gene.Gamma_esc / gene.gamma_deg)) / denom
    return int(np.floor(abs(mean_m) + 0.5)) if round_int else float(mean_m)


def steady_profile_constant_nf(
    centers: np.ndarray,
    gene,
    sim_config,
    nf_getter: Callable[[float], float],
    round_int: bool = True,
) -> np.ndarray:
    """Compute steady-state profile across theta centers using per-center Nf."""
    ages = (centers / (2.0 * np.pi)) * sim_config.T_div
    gene_copies = np.where(ages > gene.t_rep, 2.0, 1.0)
    Nf = np.array([nf_getter(float(t)) for t in ages], dtype=float)
    denom = 1.0 + (gene.k_off_rnap + gene.Gamma_esc) / (gene.k_on_rnap * Nf)
    mean_m = (gene_copies * (gene.Gamma_esc / gene.gamma_deg)) / denom
    if round_int:
        return _round_int(mean_m)
    return mean_m


def solve_mRNA_exact(
    t: np.ndarray,
    m0: float,
    gamma: float,
    Gamma: float,
    A: float,
    Nf_func: Callable[[float], float],
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

    nf_vals = np.array([clamp_nf(Nf_func(float(ts))) for ts in t], dtype=float)
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


def self_check() -> None:
    """Basic consistency check between utility functions."""

    class _G:
        def __init__(self):
            self.k_on_rnap = 0.2
            self.k_off_rnap = 0.0
            self.Gamma_esc = 1.0
            self.gamma_deg = 0.3
            self.t_rep = 20.0

    gene = _G()
    nf_get = make_nf_getter(2.0)
    centers = np.linspace(0, 2 * np.pi, 5)
    prof = steady_profile_constant_nf(
        centers, gene, type("SC", (), {"T_div": 40.0})(), nf_get, round_int=False
    )
    direct = []
    for age in (centers / (2 * np.pi)) * 40.0:
        copies = 2.0 if age > gene.t_rep else 1.0
        direct.append(steady_mean_constant_nf(gene, copies=int(copies), nf=nf_get(age), round_int=False))
    assert np.allclose(prof, direct), "steady_profile_constant_nf mismatch"


if __name__ == "__main__":
    self_check()
