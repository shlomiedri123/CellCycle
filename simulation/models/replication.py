"""Replication timing utilities for gene dosage changes.

Computes per-gene replication times from chromosomal position and builds
Gene objects used by the stochastic lineage simulator.
"""

from __future__ import annotations

from typing import Iterable, List

from simulation.config.gene_config import GeneConfig
from simulation.config.simulation_config import SimulationConfig
from simulation.models.gene import Gene


def compute_t_rep(chrom_pos_bp: float, sim_config: SimulationConfig) -> float:
    x_g = chrom_pos_bp / sim_config.chromosome_length_bp
    if x_g < 0 or x_g > 1:
        raise ValueError(f"chrom_pos_bp out of range: {chrom_pos_bp}")
    return sim_config.B_period + sim_config.C_period * x_g


def build_genes(gene_configs: Iterable[GeneConfig], sim_config: SimulationConfig) -> List[Gene]:
    genes: List[Gene] = []
    for cfg in gene_configs:
        cfg.validate()
        t_rep = compute_t_rep(cfg.chrom_pos_bp, sim_config)
        genes.append(
            Gene(
                gene_id=cfg.gene_id,
                chrom_pos_bp=cfg.chrom_pos_bp,
                phase=cfg.phase,
                gamma_deg=cfg.gamma_deg,
                t_rep=t_rep,
                Gamma_esc=cfg.Gamma_esc,
                k_on_rnap=cfg.k_on_rnap,
                k_off_rnap=cfg.k_off_rnap
            )
        )
    return genes
