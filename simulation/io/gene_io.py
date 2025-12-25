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
            genes.append(
                GeneConfig(
                    gene_id=row["gene_id"],
                    chrom_pos_bp=float(row["chrom_pos_bp"]),
                    k_on_rnap=float(row["k_on_rnap"]),
                    k_off_rnap=float(row["k_off_rnap"]),
                    Gamma_esc=float(row["Gamma_esc"]),
                    gamma_deg=float(row["gamma_deg"]),
                )
            )
    return genes
