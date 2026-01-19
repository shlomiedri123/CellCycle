"""Utility to generate random gene parameters and configs for simulation.

Assumed units: minutes for time and per-minute for rates (dt, T_div, k_on, k_off,
Gamma_esc, gamma_deg). Keep everything consistent so steady-state estimates are meaningful.

Outputs:
- gene CSV compatible with run_simulation.py
- Nf vector file (nf_vector.npy)
- sim config YAML referencing the gene table and Nf vector
- hidden metadata JSON (true Nf(t) params, gene phases/params)

This is a data-generation helper, not part of the core simulator dynamics.
"""

from __future__ import annotations

import argparse
import json
import dataclasses
import math
from pathlib import Path
from typing import Iterable, List

import numpy as np
import yaml

from simulation.config.gene_config import GeneConfig
from simulation.config.simulation_config import SimulationConfig


def _random_nffunc(rng: np.random.Generator, period: float = 40.0) -> tuple[callable, dict]:
    """Return a slowly varying Nf(t) callable and its parameters."""
    base = float(rng.uniform(1.0, 2.0))
    amp = float(rng.uniform(0.0, 0.25))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))

    def _nf(t: float) -> float:
        return base + amp * np.sin((2.0 * np.pi * t) / period + phase)

    return _nf, {"base": base, "amp": amp, "phase": phase, "period": period}

def _draw_gene(
    idx: int,
    total: int,
    rng: np.random.Generator,
    chrom_length: float,
    nf_ref: float,
    occ_low: float,
    occ_high: float,
    max_attempts: int,
) -> tuple[GeneConfig, dict]:
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

    if nf_ref <= 0:
        raise ValueError("nf_ref must be positive")

    if phase_two:
        # Phase II: high occupancy (A << Nf)
        k_on_range = (0.3, 3.0)
        Gamma_range = (0.05, 0.8)
        occ_target = occ_high
    else:
        # Phase I: low occupancy (A >> Nf)
        k_on_range = (0.005, 0.08)
        Gamma_range = (0.3, 3.0)
        occ_target = occ_low

    for _ in range(max_attempts):
        k_on = float(rng.uniform(*k_on_range))
        Gamma_esc = float(rng.uniform(*Gamma_range))
        occ = _occupancy(k_on, k_off, Gamma_esc, nf_ref)
        if phase_two and occ >= occ_target:
            break
        if not phase_two and occ <= occ_target:
            break
    else:
        phase_label = "II" if phase_two else "I"
        raise RuntimeError(
            f"Failed to sample parameters for regime {phase_label} with nf_ref={nf_ref:.3f}"
        )

    phase = "II" if phase_two else "I"
    meta = {"phase": phase}

    return (
        GeneConfig(
            gene_id=gene_id,
            chrom_pos_bp=chrom_pos_bp,
            k_on_rnap=k_on,
            k_off_rnap=k_off,
            Gamma_esc=Gamma_esc,
            gamma_deg=gamma_deg,
            phase=phase,
        ),
        meta,
    )


def generate_genes(
    n_genes: int,
    chrom_length: float,
    nf_ref: float,
    seed: int | None = None,
    occ_low: float = 0.2,
    occ_high: float = 0.8,
    max_attempts: int = 1000,
) -> tuple[List[GeneConfig], List[dict]]:
    rng = np.random.default_rng(seed)
    genes: List[GeneConfig] = []
    meta: List[dict] = []
    for i in range(1, n_genes + 1):
        g, m = _draw_gene(
            i,
            n_genes,
            rng,
            chrom_length,
            nf_ref=nf_ref,
            occ_low=occ_low,
            occ_high=occ_high,
            max_attempts=max_attempts,
        )
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

    fields = [
        "gene_id",
        "chrom_pos_bp",
        "k_on_rnap",
        "k_off_rnap",
        "Gamma_esc",
        "gamma_deg",
        "phase",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for g in genes:
            writer.writerow(asdict(g))


def _suggest_total_time(n_samples: int, initial_cells: int, t_div: float) -> float:
    if n_samples <= initial_cells:
        return t_div
    cycles = math.ceil(math.log2(n_samples / float(initial_cells)))
    return (cycles + 1) * t_div


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random genes and configs.")
    parser.add_argument("--n_genes", type=int, required=True, help="Number of genes to generate")
    parser.add_argument("--n_samples", type=int, required=True, help="Number of samples target (for config)")
    parser.add_argument("--chrom_length", type=float, default=4_600_000, help="Chromosome length (bp)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--out_dir", type=Path, default=Path("simulation/test_data"), help="Output directory")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    nf_func, nf_params = _random_nffunc(np.random.default_rng(args.seed), period=40.0)  # Nf(t) slowly varying in [~1, ~2]
    t_div = 40.0
    initial_cells = 3
    t_total = _suggest_total_time(args.n_samples, initial_cells, t_div)
    dt = 0.1
    n_steps = int(round(t_div / dt))
    time_grid = np.arange(n_steps, dtype=float) * dt
    nf_vec = np.array([nf_func(t) for t in time_grid], dtype=float)
    nf_ref = float(np.mean(nf_vec))
    genes, gene_meta = generate_genes(
        args.n_genes,
        args.chrom_length,
        nf_ref=nf_ref,
        seed=args.seed,
    )
    nf_path = args.out_dir / "nf_vector.npy"
    np.save(nf_path, nf_vec)

    sim_config_runtime = SimulationConfig(
        B_period=10.0,
        C_period=20.0,
        D_period=10.0,
        T_total=t_total,
        dt=dt,
        N_target_samples=args.n_samples,
        random_seed=args.seed or 123,
        chromosome_length_bp=args.chrom_length,
        MAX_MRNA_PER_GENE=10_000,
        genes_path=str(args.out_dir / "random_genes.csv"),
        nf_vector_path=str(nf_path),
        out_path=str(args.out_dir / "snapshots.csv"),
        initial_cell_count=initial_cells,
    )

    _write_gene_csv(args.out_dir / "random_genes.csv", genes)

    # Save a YAML-friendly config referencing the Nf vector file.
    sim_config_yaml = sim_config_runtime
    cfg_dict = dataclasses.asdict(sim_config_yaml)
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
