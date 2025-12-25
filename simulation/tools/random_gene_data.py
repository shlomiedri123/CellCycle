"""Utility to generate random gene parameters and configs for simulation.

Assumed units: minutes for time and per-minute for rates (dt, T_div, k_on, k_off,
Gamma_esc, gamma_deg). Keep everything consistent so steady-state estimates are meaningful.

Outputs:
- gene CSV compatible with run_simulation.py
- sim config YAML (with placeholder Nf_global; real callable described in hidden params)
- hidden metadata JSON (true Nf(t) params, gene phases/params)
"""

from __future__ import annotations

import argparse
import json
import dataclasses
from pathlib import Path
from typing import Iterable, List

import numpy as np
import yaml

from simulation.config.gene_config import GeneConfig
from simulation.config.simulation_config import SimulationConfig


def _random_nffunc(rng: np.random.Generator, period: float = 40.0) -> tuple[callable, dict]:
    """Return a slowly varying Nf_global(t) callable and its parameters."""
    base = float(rng.uniform(1.0, 2.0))
    amp = float(rng.uniform(0.0, 0.25))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))

    def _nf(t: float) -> float:
        return base + amp * np.sin((2.0 * np.pi * t) / period + phase)

    return _nf, {"base": base, "amp": amp, "phase": phase, "period": period}

def _draw_gene(idx: int, total: int, rng: np.random.Generator, chrom_length: float) -> tuple[GeneConfig, dict]:
    gene_id = f"gene_{idx}"
    # Stratify along the chromosome to avoid clustering
    span = chrom_length / float(total)
    base = (idx - 1) * span
    jitter = float(rng.uniform(0.1 * span, 0.9 * span))
    chrom_pos_bp = base + jitter
    phase_two = rng.random() < 0.5  # coin flip: phase II vs phase I

    # Degradation ~1/3 min^-1, varied within roughly one order of magnitude (clamped)
    gamma_deg = float(rng.lognormal(mean=np.log(1.0 / 3.0), sigma=0.5))
    gamma_deg = float(np.clip(gamma_deg, 0.05, 1.0))

    # Michael: k_off ≈ 0. Occupancy differences come from k_on vs Gamma_esc.
    k_off = float(rng.uniform(0.0, 1e-3))

    if phase_two:
        # Phase II: occupancy ~1 (high k_on, moderate Gamma_esc)
        k_on = float(rng.uniform(0.1, 0.3))
        Gamma_esc = float(rng.uniform(0.2, 3.0))
    else:
        # Phase I: lower occupancy (smaller k_on and/or larger Gamma_esc)
        k_on = float(rng.uniform(0.01, 0.08))
        Gamma_esc = float(rng.uniform(0.05, 1.5))

    meta = {"phase": "II" if phase_two else "I"}

    return (
        GeneConfig(
            gene_id=gene_id,
            chrom_pos_bp=chrom_pos_bp,
            k_on_rnap=k_on,
            k_off_rnap=k_off,
            Gamma_esc=Gamma_esc,
            gamma_deg=gamma_deg,
        ),
        meta,
    )


def generate_genes(n_genes: int, chrom_length: float, seed: int | None = None) -> tuple[List[GeneConfig], List[dict]]:
    rng = np.random.default_rng(seed)
    genes: List[GeneConfig] = []
    meta: List[dict] = []
    for i in range(1, n_genes + 1):
        g, m = _draw_gene(i, n_genes, rng, chrom_length)
        g.validate()
        genes.append(g)
        meta.append(m)
    return genes, meta


def _occupancy(k_on: float, k_off: float, Gamma_esc: float, N_f: float) -> float:
    denom = 1.0 + (k_off + Gamma_esc) / (k_on * N_f)
    return 1.0 / denom


def _write_gene_csv(path: Path, genes: Iterable[GeneConfig]) -> None:
    import csv
    from dataclasses import asdict

    fields = ["gene_id", "chrom_pos_bp", "k_on_rnap", "k_off_rnap", "Gamma_esc", "gamma_deg"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for g in genes:
            writer.writerow(asdict(g))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random genes and configs.")
    parser.add_argument("--n_genes", type=int, required=True, help="Number of genes to generate")
    parser.add_argument("--n_samples", type=int, required=True, help="Number of samples target (for config)")
    parser.add_argument("--chrom_length", type=float, default=4_600_000, help="Chromosome length (bp)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--out_dir", type=Path, default=Path("simulation/test_data"), help="Output directory")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    genes, gene_meta = generate_genes(args.n_genes, args.chrom_length, seed=args.seed)
    nf_func, nf_params = _random_nffunc(np.random.default_rng(args.seed), period=40.0)  # Nf(t) global, slowly varying in [~1, ~2]
    sim_config_runtime = SimulationConfig(
        B_period=10.0,
        C_period=20.0,
        D_period=10.0,
        dt=0.1,
        N_target_cells=1000,
        N_target_samples=args.n_samples,
        random_seed=args.seed or 123,
        chromosome_length_bp=args.chrom_length,
        Nf_global=nf_func,  # callable for simulation runtime
        MAX_MRNA_PER_GENE=10_000,
    )

    _write_gene_csv(args.out_dir / "random_genes.csv", genes)

    # Save a YAML-friendly config with Nf parameters; loader will rebuild callable
    sim_config_yaml = dataclasses.replace(sim_config_runtime, Nf_global=nf_params["base"])
    cfg_dict = dataclasses.asdict(sim_config_yaml)
    cfg_dict.update(
        {
            "Nf_type": "sine",
            "Nf_base": nf_params["base"],
            "Nf_amp": nf_params["amp"],
            "Nf_phase": nf_params["phase"],
            "Nf_period": nf_params["period"],
            "Nf_mode": "provided",
        }
    )
    with (args.out_dir / "random_sim_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=True)

    hidden = {
        "nf_params": nf_params,
        "genes": [
            {
                "gene_id": g.gene_id,
                "phase": m["phase"],
                "k_on_rnap": g.k_on_rnap,
                "k_off_rnap": g.k_off_rnap,
                "Gamma_esc": g.Gamma_esc,
                "gamma_deg": g.gamma_deg,
                "chrom_pos_bp": g.chrom_pos_bp,
            }
            for g, m in zip(genes, gene_meta)
        ],
    }
    with (args.out_dir / "random_hidden_params.json").open("w", encoding="utf-8") as f:
        json.dump(hidden, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
