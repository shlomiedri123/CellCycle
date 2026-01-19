"""Read per-gene kinetic parameters used by the simulator."""

from __future__ import annotations

import csv
import pathlib
from typing import List

from simulation.config.gene_config import GeneConfig


def load_gene_table(path: str | pathlib.Path) -> List[GeneConfig]:
    genes: List[GeneConfig] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"gene_id", "chrom_pos_bp", "k_on_rnap", "k_off_rnap", "Gamma_esc", "gamma_deg"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in gene CSV: {missing}")
        for row in reader:
            phase_val = row.get("phase", "") or row.get("regime", "")
            phase_norm = None
            if phase_val:
                phase_key = str(phase_val).strip().upper()
                if phase_key in {"1", "I"}:
                    phase_norm = "I"
                elif phase_key in {"2", "II"}:
                    phase_norm = "II"
                else:
                    raise ValueError(f"phase must be 'I' or 'II' for {row.get('gene_id', '')}")
            genes.append(
                GeneConfig(
                    gene_id=row["gene_id"],
                    chrom_pos_bp=float(row["chrom_pos_bp"]),
                    k_on_rnap=float(row["k_on_rnap"]),
                    k_off_rnap=float(row["k_off_rnap"]),
                    Gamma_esc=float(row["Gamma_esc"]),
                    gamma_deg=float(row["gamma_deg"]),
                    phase=phase_norm,
                )
            )
    return genes
