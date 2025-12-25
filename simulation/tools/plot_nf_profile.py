from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from simulation.io.config_io import load_simulation_config
from simulation.io.gene_io import load_gene_table
from simulation.lineage.lineage_simulator import build_nf_getter
from simulation.models.replication import build_genes


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Nf(t) from config and gene table.")
    parser.add_argument("--config", required=True, help="Path to simulation YAML config")
    parser.add_argument("--genes", required=True, help="Path to gene CSV")
    parser.add_argument("--out", required=True, help="Output path for the PNG")
    parser.add_argument("--points", type=int, default=500, help="Number of points along the cycle")
    parser.add_argument(
        "--step",
        action="store_true",
        help="Use a step plot (useful to visualize copy-number jumps).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    if args.points <= 1:
        raise ValueError("--points must be > 1")

    sim_config = load_simulation_config(args.config)
    gene_cfgs = load_gene_table(args.genes)
    genes = build_genes(gene_cfgs, sim_config)
    nf_getter = build_nf_getter(sim_config, genes)
    mode = sim_config.Nf_mode

    ages = np.linspace(0.0, sim_config.T_div, args.points)
    values = np.array([nf_getter(age) for age in ages], dtype=float)

    fig, ax = plt.subplots()
    if args.step:
        ax.step(ages, values, where="post")
    else:
        ax.plot(ages, values)
    ax.set_xlabel("age")
    ax.set_ylabel("Nf(t)")
    ax.set_title(f"Nf(t) ({mode})")
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.clf()
    logging.info("Wrote Nf(t) plot to %s", out_path)


if __name__ == "__main__":
    main()
