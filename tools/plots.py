from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from simulation.analysis.age_distribution import plot_age_distribution as _plot_age_distribution
from simulation.analysis.m_profiles import solve_mRNA_exact
from simulation.analysis.trip_profile import mean_adjusted_trip, plot_trip_profiles_grid
from simulation.io.config_io import load_simulation_config
from simulation.io.gene_io import load_gene_table
from simulation.io.nf_io import load_nf_vector
from simulation.io.output_io import load_snapshot_csv
from simulation.models.replication import build_genes


def plot_age_distribution(
    snapshots_path: str | Path,
    config_path: str | Path,
    out_path: str | Path,
    bins: int = 20,
):
    rows = load_snapshot_csv(snapshots_path)
    sim_config = load_simulation_config(config_path)
    fig, ax = _plot_age_distribution(rows, sim_config, bins=bins)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.clf()
    return fig, ax


def plot_Nf_timecourse(
    config_path: str | Path,
    out_path: str | Path,
    points: int = 500,
    step: bool = False,
):
    if points <= 1:
        raise ValueError("points must be > 1")

    sim_config = load_simulation_config(config_path)
    nf_vec = load_nf_vector(sim_config.nf_vector_path)

    total_time = sim_config.dt * nf_vec.size
    ages = np.arange(nf_vec.size, dtype=float) * sim_config.dt
    values = nf_vec
    if points < nf_vec.size:
        sample_idx = np.linspace(0, nf_vec.size - 1, points).astype(int)
        ages = ages[sample_idx]
        values = values[sample_idx]

    fig, ax = plt.subplots()
    if step:
        ax.step(ages, values, where="post")
    else:
        ax.plot(ages, values)
    ax.set_xlabel("age")
    ax.set_ylabel("Nf(t)")
    ax.set_title(f"Nf(t) (T_div={total_time:g}, dt={sim_config.dt:g})")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.clf()
    return fig, ax


def plot_TRIP_report_style_expression(
    snapshots_path: str | Path,
    config_path: str | Path,
    out_path: str | Path,
    genes_path: str | Path | None = None,
    max_genes: int = 100,
    bins: int = 25,
    ncols: int = 10,
    angle_shift: float = 0.0,
    seed: int | None = None,
    title: str = "Mean expression per gene",
):
    sim_config = load_simulation_config(config_path)
    gene_path = genes_path if genes_path is not None else sim_config.genes_path
    gene_cfgs = load_gene_table(gene_path)
    genes = build_genes(gene_cfgs, sim_config)
    rows = load_snapshot_csv(snapshots_path)
    nf_vec = load_nf_vector(sim_config.nf_vector_path)

    gene_ids = [g.gene_id for g in genes]
    missing = [gid for gid in gene_ids if gid not in rows[0]]
    if missing:
        raise ValueError(f"Snapshot file is missing gene columns: {', '.join(missing)}")

    rng_seed = seed if seed is not None else sim_config.random_seed
    plot_rng = np.random.default_rng(rng_seed)
    plot_gene_ids = _select_plot_genes(gene_ids, max_genes, plot_rng)
    if len(plot_gene_ids) < len(gene_ids):
        logging.info("Plotting %d of %d genes (random subset).", len(plot_gene_ids), len(gene_ids))

    profiles = mean_adjusted_trip(rows, plot_gene_ids, nbins=bins)
    m_profiles = {}
    gene_lookup = {g.gene_id: g for g in genes}
    for gene_id in plot_gene_ids:
        gene = gene_lookup[gene_id]
        centers, _ = profiles[gene_id]
        ages = (centers / (2.0 * np.pi)) * sim_config.T_div
        denom = 1.0 + (gene.k_off_rnap + gene.Gamma_esc) / (gene.k_on_rnap * nf_vec[0])
        m0 = (gene.Gamma_esc / gene.gamma_deg) / denom
        A = (gene.k_off_rnap + gene.Gamma_esc) / gene.k_on_rnap

        def _g_func(t: float, trep: float = gene.t_rep) -> float:
            return 2.0 if t > trep else 1.0

        m_vals = solve_mRNA_exact(
            ages,
            m0,
            gamma=gene.gamma_deg,
            Gamma=gene.Gamma_esc,
            A=A,
            nf_vec=nf_vec,
            dt=sim_config.dt,
            g_func=_g_func,
            round_int=False,
        )
        m_profiles[gene_id] = m_vals

    rep_marks = {
        g.gene_id: (2.0 * np.pi * g.t_rep) / sim_config.T_div
        for g in genes
        if g.gene_id in plot_gene_ids
    }
    b_end = 2.0 * np.pi * (sim_config.B_period / sim_config.T_div)
    c_end = 2.0 * np.pi * ((sim_config.B_period + sim_config.C_period) / sim_config.T_div)
    phase_spans = (b_end, c_end, 2.0 * np.pi)

    fig, axes = plot_trip_profiles_grid(
        profiles,
        plot_gene_ids,
        ncols=ncols,
        title=title,
        rep_marks=rep_marks,
        phase_spans=phase_spans,
        angle_shift=angle_shift,
        steady_profiles=m_profiles,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.clf()
    return fig, axes


def _select_plot_genes(gene_ids: list[str], max_genes: int, rng: np.random.Generator) -> list[str]:
    if max_genes <= 0 or len(gene_ids) <= max_genes:
        return list(gene_ids)
    return list(rng.choice(gene_ids, size=max_genes, replace=False))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plotting utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    age_parser = subparsers.add_parser("age", help="Plot age distribution from snapshots")
    age_parser.add_argument("--snapshots", required=True, help="Path to snapshots CSV")
    age_parser.add_argument("--config", required=True, help="Path to simulation YAML config")
    age_parser.add_argument("--out", required=True, help="Output path for the PNG")
    age_parser.add_argument("--bins", type=int, default=20, help="Number of age bins")

    nf_parser = subparsers.add_parser("nf", help="Plot Nf(t) from a YAML config and Nf vector")
    nf_parser.add_argument("--config", required=True, help="Path to simulation YAML config")
    nf_parser.add_argument("--out", required=True, help="Output path for the PNG")
    nf_parser.add_argument("--points", type=int, default=500, help="Number of points to plot")
    nf_parser.add_argument(
        "--step",
        action="store_true",
        help="Use a step plot (useful to visualize copy-number jumps).",
    )

    trip_parser = subparsers.add_parser("trip", help="Plot TRIP profile grid from snapshots")
    trip_parser.add_argument("--snapshots", required=True, help="Path to snapshots CSV")
    trip_parser.add_argument("--genes", default=None, help="Path to gene CSV (default: use config)")
    trip_parser.add_argument("--config", required=True, help="Path to simulation YAML config")
    trip_parser.add_argument("--out", required=True, help="Output path for the PNG")
    trip_parser.add_argument("--max_genes", type=int, default=100, help="Max genes to plot (random subset)")
    trip_parser.add_argument("--bins", type=int, default=25, help="Number of theta bins")
    trip_parser.add_argument("--ncols", type=int, default=10, help="Grid columns")
    trip_parser.add_argument("--angle_shift", type=float, default=0.0, help="Cycle shift in radians")
    trip_parser.add_argument("--seed", type=int, default=None, help="Random seed for gene subset")
    trip_parser.add_argument("--title", default="Mean expression per gene", help="Plot title")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    if args.command == "age":
        plot_age_distribution(
            snapshots_path=args.snapshots,
            config_path=args.config,
            out_path=args.out,
            bins=args.bins,
        )
    elif args.command == "nf":
        plot_Nf_timecourse(
            config_path=args.config,
            out_path=args.out,
            points=args.points,
            step=args.step,
        )
    else:
        plot_TRIP_report_style_expression(
            snapshots_path=args.snapshots,
            config_path=args.config,
            out_path=args.out,
            genes_path=args.genes,
            max_genes=args.max_genes,
            bins=args.bins,
            ncols=args.ncols,
            angle_shift=args.angle_shift,
            seed=args.seed,
            title=args.title,
        )


if __name__ == "__main__":
    main()
