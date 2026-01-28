"""Analysis utilities for simulation outputs.

Provides functions for computing age distributions, mRNA profiles, and plotting.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from Simulation.config import SimulationConfig


# -----------------------------------------------------------------------------
# Age distribution analysis
# -----------------------------------------------------------------------------

def age_array(rows: Iterable[Mapping[str, object]]) -> np.ndarray:
    """Extract cell ages from snapshot rows."""
    return np.array([float(r["age"]) for r in rows], dtype=float)


def age_density(sim_config: SimulationConfig, ages: np.ndarray, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Compute age density histogram."""
    hist, edges = np.histogram(ages, bins=bins, range=(0.0, sim_config.T_div), density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, hist


def expected_exponential(sim_config: SimulationConfig, ages: np.ndarray) -> np.ndarray:
    """Compute expected exponential age distribution."""
    T = sim_config.T_div
    return (np.log(2.0) / T) * np.power(2.0, 1.0 - ages / T)


def plot_age_distribution(rows: Iterable[Mapping[str, object]], sim_config: SimulationConfig, bins: int = 20):
    """Plot age distribution comparing simulated to expected exponential."""
    import matplotlib.pyplot as plt

    ages = age_array(rows)
    centers, hist = age_density(sim_config, ages, bins=bins)
    dense_ages = np.linspace(0.0, sim_config.T_div, 200)
    expected = expected_exponential(sim_config, dense_ages)

    fig, ax_age = plt.subplots()

    ax_age.bar(centers, hist, width=centers[1] - centers[0], alpha=0.6, label="snapshot")
    ax_age.plot(dense_ages, expected, color="red", label="expected exp")
    ax_age.set_xlabel("Cell-cycle age")
    ax_age.set_ylabel("Density (log scale)")
    ax_age.set_yscale("log")
    ax_age.legend()

    fig.tight_layout()
    return fig, ax_age


# -----------------------------------------------------------------------------
# mRNA profile analysis
# -----------------------------------------------------------------------------

def _nf_at_times(nf_vec: np.ndarray, t: np.ndarray, dt: float) -> np.ndarray:
    """Get Nf values at given times with periodic wrap-around."""
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
    """Compute cumulative trapezoidal integral."""
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
    """Compute integral over one cell cycle."""
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


def solve_mRNA_exact(
    t: np.ndarray,
    gamma: float,
    Gamma: float,
    A: float,
    nf_vec: np.ndarray,
    dt: float,
    g_func: Callable[[float], float] | None = None,
    regime: str | None = None,
) -> np.ndarray:
    """Compute exact periodic solution for mRNA profiles in regime I/II.

    Uses tau = len(nf_vec) * dt and the closed-form periodic integrals.
    """
    t = np.asarray(t, dtype=float)
    if np.any(t < 0):
        raise ValueError("Time values must be non-negative")
    if g_func is None:
        g_func = lambda _t: 1.0
    regime_key = str(regime).strip().upper() if regime is not None else ""

    if regime_key in ("I", "1"):
        if A <= 0:
            raise ValueError("A must be positive for regime I")
        cycle_integral, tau = _cycle_integral(gamma, nf_vec, dt, g_func, use_nf=True)
        denom = 2.0 * np.exp(gamma * tau) - 1.0
        nf_vals = _nf_at_times(nf_vec, t, dt)
        g_vals = np.array([float(g_func(float(ts))) for ts in t], dtype=float)
        integrand = np.exp(gamma * t) * g_vals * nf_vals
        integral_to_t = _cumulative_trapz(integrand, t)
        prefactor = Gamma / A
        return prefactor * np.exp(-gamma * t) * (cycle_integral / denom + integral_to_t)
    elif regime_key in ("II", "2"):
        cycle_integral, tau = _cycle_integral(gamma, nf_vec, dt, g_func, use_nf=False)
        denom = 2.0 * np.exp(gamma * tau) - 1.0
        g_vals = np.array([float(g_func(float(ts))) for ts in t], dtype=float)
        integrand = np.exp(gamma * t) * g_vals
        integral_to_t = _cumulative_trapz(integrand, t)
        return Gamma * np.exp(-gamma * t) * (cycle_integral / denom + integral_to_t)
    else:
        raise ValueError("regime must be 'I' or 'II'")


# -----------------------------------------------------------------------------
# TRIP profile utilities
# -----------------------------------------------------------------------------

def _theta_array(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    """Extract theta_rad values from snapshot rows."""
    if not rows:
        return np.array([], dtype=float)
    return np.array([float(r["theta_rad"]) for r in rows], dtype=float)


def mean_expression_by_bin(
    rows: Sequence[Mapping[str, object]],
    gene_ids: Sequence[str],
    nbins: int = 25,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute mean gene expression per theta bin."""
    theta = _theta_array(rows)
    edges = np.linspace(0.0, 2.0 * np.pi, nbins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    bin_idx = np.digitize(theta, edges) - 1
    bin_idx = np.clip(bin_idx, 0, nbins - 1)

    if not gene_ids:
        return {}

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for gene in gene_ids:
        values = np.array([float(r[gene]) for r in rows], dtype=float)
        sums = np.zeros(nbins, dtype=float)
        counts = np.zeros(nbins, dtype=float)
        for idx, value in zip(bin_idx, values):
            sums[idx] += value
            counts[idx] += 1.0
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        result[gene] = (centers, np.abs(means))
    return result
