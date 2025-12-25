from __future__ import annotations

import argparse
import dataclasses
import pathlib
from typing import Sequence

from simulation.config.simulation_config import SimulationConfig
from simulation.io.config_io import load_simulation_config
from simulation.io.gene_io import load_gene_table
from simulation.io.output_io import apply_measurement_model, load_measured_mrna_distribution, save_snapshots_csv
from simulation.lineage.lineage_simulator import LineageSimulator
from simulation.models.replication import build_genes


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RNAP-limited lineage simulation.")
    parser.add_argument("--config", required=True, help="Path to simulation YAML config")
    parser.add_argument("--genes", required=True, help="Path to gene CSV table")
    parser.add_argument("--out", required=True, help="Output CSV path for snapshots")
    parser.add_argument(
        "--measured_dist",
        type=pathlib.Path,
        default=None,
        help="Path to measured mRNA distribution JSON for parsed snapshots.",
    )
    parser.add_argument(
        "--parsed_out",
        type=pathlib.Path,
        default=None,
        help="Output CSV path for parsed snapshots (default: add _parsed suffix to --out).",
    )
    parser.add_argument("--n_samples", type=int, default=None, help="Override N_target_samples")
    parser.add_argument("--snapshot_interval", type=int, default=10, help="Snapshot interval in steps")
    return parser.parse_args(argv)


def maybe_override_samples(config: SimulationConfig, n_samples: int | None) -> SimulationConfig:
    if n_samples is None:
        return config
    if n_samples <= 0:
        raise ValueError("n_samples override must be positive")
    return dataclasses.replace(config, N_target_samples=n_samples)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    sim_config = load_simulation_config(args.config)
    sim_config = maybe_override_samples(sim_config, args.n_samples)
    gene_configs = load_gene_table(args.genes) # Create the config for each gene. 
    genes = build_genes(gene_configs, sim_config) # Build Gene objects from the configs.

    simulator = LineageSimulator(sim_config, genes, snapshot_interval_steps=args.snapshot_interval)
    snapshots = simulator.run()
    save_snapshots_csv(snapshots, args.out)
    if args.measured_dist is not None:
        mu, sigma = load_measured_mrna_distribution(args.measured_dist)
        gene_ids = [g.gene_id for g in genes]
        parsed_rows = apply_measurement_model(
            snapshots,
            gene_ids,
            mu=mu,
            sigma=sigma,
            seed=sim_config.random_seed,
        )
        out_path = pathlib.Path(args.out)
        parsed_path = args.parsed_out
        if parsed_path is None:
            suffix = out_path.suffix or ".csv"
            parsed_path = out_path.with_name(out_path.stem + "_parsed" + suffix)
        save_snapshots_csv(parsed_rows, parsed_path)
        print(f"Wrote parsed snapshots to {parsed_path}")
    
    print(f"Wrote {len(snapshots)} samples to {args.out}")


if __name__ == "__main__":
    main()
