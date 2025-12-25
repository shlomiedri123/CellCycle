from __future__ import annotations

import math
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _theta_array(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    if not rows:
        return np.array([], dtype=float)
    return np.array([float(r["theta_rad"]) for r in rows], dtype=float)


def _bin_theta(theta: np.ndarray, nbins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if nbins <= 0:
        raise ValueError("nbins must be positive")
    edges = np.linspace(0.0, 2.0 * np.pi, nbins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    bin_idx = np.digitize(theta, edges) - 1
    bin_idx = np.clip(bin_idx, 0, nbins - 1)
    return edges, centers, bin_idx


def mean_expression_by_bin(
    rows: Sequence[Mapping[str, object]],
    gene_ids: Sequence[str],
    nbins: int = 25,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    theta = _theta_array(rows)
    _, centers, bin_idx = _bin_theta(theta, nbins)
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


def mean_adjusted_trip(
    rows: Sequence[Mapping[str, object]],
    gene_ids: Sequence[str],
    nbins: int = 25,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return mean_expression_by_bin(rows, gene_ids, nbins=nbins)


def plot_trip_profiles_grid(
    profiles: Mapping[str, tuple[np.ndarray, np.ndarray]],
    gene_ids: Sequence[str],
    ncols: int = 5,
    title: str | None = None,
    rep_marks: Mapping[str, float] | None = None,
    phase_spans: tuple[float, float, float] | None = None,
    angle_shift: float = np.pi,
    division_angle: float | None = None,
    steady_profiles: Mapping[str, np.ndarray] | None = None,
):
    two_pi = 2.0 * np.pi

    if ncols <= 0:
        raise ValueError("ncols must be positive")

    nrows = math.ceil(len(gene_ids) / ncols) if gene_ids else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2), squeeze=False)

    for idx, gene_id in enumerate(gene_ids):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        centers, values = profiles[gene_id]
        shifted_centers = (centers + angle_shift) % two_pi
        ax.plot(shifted_centers, values, marker="o")
        if steady_profiles and gene_id in steady_profiles:
            steady_vals = steady_profiles[gene_id]
            ax.plot(
                shifted_centers,
                steady_vals,
                color="black",
                linestyle="--",
                linewidth=1,
                label="m_profile",
            )
        if rep_marks and gene_id in rep_marks:
            rep_angle = (rep_marks[gene_id] + angle_shift) % two_pi
            ax.axvline(rep_angle, color="red", linestyle=":", linewidth=1)

        if phase_spans:
            b_end, c_end, total = phase_spans
            spans = [
                (0.0, b_end, "red", 0.08),
                (b_end, c_end, "green", 0.06),
                (c_end, total, "blue", 0.06),
            ]
            for start, end, color, alpha in spans:
                s = (start + angle_shift) % two_pi
                e = (end + angle_shift) % two_pi
                if s < e:
                    ax.axvspan(s, e, color=color, alpha=alpha)
                else:
                    ax.axvspan(s, two_pi, color=color, alpha=alpha)
                    ax.axvspan(0.0, e, color=color, alpha=alpha)
        ax.set_title(gene_id)
        ax.set_xlim(0.0, two_pi)
        if steady_profiles and gene_id in steady_profiles:
            ax.legend(loc="upper right", fontsize="x-small")

    for idx in range(len(gene_ids), nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis("off")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    return fig, axes
