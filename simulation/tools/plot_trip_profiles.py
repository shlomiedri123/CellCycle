from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from simulation.analysis.m_profiles import (
    solve_mRNA_exact,
    steady_mean_constant_nf,
    steady_profile_constant_nf,
)
from simulation.analysis.trip_profile import mean_adjusted_trip, plot_trip_profiles_grid
from simulation.io.config_io import load_simulation_config
from simulation.io.gene_io import load_gene_table
from simulation.io.output_io import load_snapshots_csv
from simulation.lineage.lineage_simulator import build_nf_getter
from simulation.models.replication import build_genes


def _select_plot_genes(gene_ids: list[str], max_genes: int, rng: np.random.Generator) -> list[str]:
    if max_genes <= 0 or len(gene_ids) <= max_genes:
        return list(gene_ids)
    return list(rng.choice(gene_ids, size=max_genes, replace=False))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TRIP profile grid from snapshots.")
    parser.add_argument("--snapshots", required=True, help="Path to snapshots CSV")
    parser.add_argument("--genes", required=True, help="Path to gene CSV")
    parser.add_argument("--config", required=True, help="Path to simulation YAML config")
    parser.add_argument("--out", required=True, help="Output path for the PNG")
    parser.add_argument("--max_genes", type=int, default=100, help="Max genes to plot (random subset)")
    parser.add_argument("--bins", type=int, default=25, help="Number of theta bins")
    parser.add_argument("--ncols", type=int, default=10, help="Grid columns")
    parser.add_argument("--angle_shift", type=float, default=0.0, help="Cycle shift in radians")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for gene subset")
    parser.add_argument("--title", default="Mean expression per gene", help="Plot title")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    sim_config = load_simulation_config(args.config)
    gene_cfgs = load_gene_table(args.genes)
    genes = build_genes(gene_cfgs, sim_config)
    rows = load_snapshots_csv(args.snapshots)

    gene_ids = [g.gene_id for g in genes]
    missing = [gid for gid in gene_ids if gid not in rows[0]]
    if missing:
        raise ValueError(f"Snapshot file is missing gene columns: {', '.join(missing)}")

    rng_seed = args.seed if args.seed is not None else sim_config.random_seed
    plot_rng = np.random.default_rng(rng_seed)
    plot_gene_ids = _select_plot_genes(gene_ids, args.max_genes, plot_rng)
    if len(plot_gene_ids) < len(gene_ids):
        logging.info("Plotting %d of %d genes (random subset).", len(plot_gene_ids), len(gene_ids))

    profiles = mean_adjusted_trip(rows, plot_gene_ids, nbins=args.bins)
    nf_getter = build_nf_getter(sim_config, genes)
    time_dependent = sim_config.Nf_mode == "operon_scaled" or type(sim_config.Nf_global) != float
    m_profiles = {}
    gene_lookup = {g.gene_id: g for g in genes}
    for gene_id in plot_gene_ids:
        gene = gene_lookup[gene_id]
        centers, _ = profiles[gene_id]
        if time_dependent:
            ages = (centers / (2.0 * np.pi)) * sim_config.T_div
            m0 = steady_mean_constant_nf(gene, copies=1, nf=nf_getter(0.0), round_int=False)
            A = (gene.k_off_rnap + gene.Gamma_esc) / gene.k_on_rnap
            print(f"Computing time-dependent profile for gene {gene_id} with m0={m0:.2f} and A={A:.2f}")
            def _g_func(t: float, trep=gene.t_rep) -> float:
                return 2.0 if t > trep else 1.0

            m_vals = solve_mRNA_exact(
                ages,
                m0,
                gamma=gene.gamma_deg,
                Gamma=gene.Gamma_esc,
                A=A,
                Nf_func=nf_getter,
                g_func=_g_func,
                round_int=False,
            )
        else:
            m_vals = steady_profile_constant_nf(
                centers,
                gene,
                sim_config,
                nf_getter,
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

    fig, _ = plot_trip_profiles_grid(
        profiles,
        plot_gene_ids,
        ncols=args.ncols,
        title=args.title,
        rep_marks=rep_marks,
        phase_spans=phase_spans,
        angle_shift=args.angle_shift,
        steady_profiles=m_profiles,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.clf()
    logging.info("Wrote TRIP profile grid to %s", out_path)


if __name__ == "__main__":
    main()
