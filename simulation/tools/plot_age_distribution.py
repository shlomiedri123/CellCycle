from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from simulation.analysis.age_distribution import plot_age_distribution
from simulation.io.config_io import load_simulation_config
from simulation.io.output_io import load_snapshots_csv


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot age distribution from a snapshots CSV.")
    parser.add_argument("--snapshots", required=True, help="Path to snapshots CSV")
    parser.add_argument("--config", required=True, help="Path to simulation YAML config")
    parser.add_argument("--out", required=True, help="Output path for the PNG")
    parser.add_argument("--bins", type=int, default=20, help="Number of age bins")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    rows = load_snapshots_csv(args.snapshots)
    sim_config = load_simulation_config(args.config)
    fig, _ = plot_age_distribution(rows, sim_config, bins=args.bins)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.clf()
    logging.info("Wrote age distribution plot to %s", out_path)


if __name__ == "__main__":
    main()
