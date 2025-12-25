from __future__ import annotations

import numpy as np
from typing import Iterable, Mapping

from simulation.config.simulation_config import SimulationConfig


def age_array(rows: Iterable[Mapping[str, object]]) -> np.ndarray:
    return np.array([float(r["age"]) for r in rows], dtype=float)


def age_density(sim_config: SimulationConfig, ages: np.ndarray, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(ages, bins=bins, range=(0.0, sim_config.T_div), density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, hist


def expected_exponential(sim_config: SimulationConfig, ages: np.ndarray) -> np.ndarray:
    T = sim_config.T_div
    return (np.log(2.0) / T) * np.power(2.0, 1.0 - ages / T)


def plot_age_distribution(rows: Iterable[Mapping[str, object]], sim_config: SimulationConfig, bins: int = 20):
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
