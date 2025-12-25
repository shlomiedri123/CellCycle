from __future__ import annotations

import numpy as np

from simulation.config.simulation_config import SimulationConfig
from simulation.lineage.lineage_simulator import LineageSimulator
from simulation.models.gene import Gene


def _make_genes() -> list[Gene]:
    return [
        Gene(
            gene_id="g1",
            chrom_pos_bp=1.0,
            gamma_deg=0.1,
            Gamma_esc=1.0,
            t_rep=10.0,
            k_on_rnap=0.1,
            k_off_rnap=0.01,
        ),
        Gene(
            gene_id="g2",
            chrom_pos_bp=2.0,
            gamma_deg=0.1,
            Gamma_esc=1.0,
            t_rep=20.0,
            k_on_rnap=0.1,
            k_off_rnap=0.01,
        ),
    ]


def _base_config() -> SimulationConfig:
    return SimulationConfig(
        B_period=10.0,
        C_period=20.0,
        D_period=10.0,
        dt=1.0,
        N_target_cells=10,
        N_target_samples=10,
        random_seed=123,
        chromosome_length_bp=4_600_000,
        Nf_global=None,
        Nf_mode="operon_scaled",
        Nf_birth=1.5,
        Nf_min=1e-9,
        Nf_max=None,
        MAX_MRNA_PER_GENE=1000,
    )


def test_operon_scaled_nf_lineage() -> None:
    genes = _make_genes()
    cfg = _base_config()
    sim = LineageSimulator(cfg, genes, snapshot_interval_steps=1, initial_cell_count=1)

    def expected(age: float) -> float:
        operons0 = 2.0
        copies = np.where(age > np.array([10.0, 20.0]), 2.0, 1.0)
        operons = float(copies.sum())
        growth = 2.0 ** (age / cfg.T_div)
        return cfg.Nf_birth * growth / (operons / operons0)

    for age in (0.0, 5.0, 15.0, 25.0):
        got = sim._Nf_getter(age)
        assert np.isclose(got, expected(age))


def test_provided_nf_constant() -> None:
    genes = _make_genes()
    cfg = SimulationConfig(
        B_period=10.0,
        C_period=20.0,
        D_period=10.0,
        dt=1.0,
        N_target_cells=10,
        N_target_samples=10,
        random_seed=123,
        chromosome_length_bp=4_600_000,
        Nf_global=2.5,
        Nf_mode="provided",
        MAX_MRNA_PER_GENE=1000,
    )
    sim = LineageSimulator(cfg, genes, snapshot_interval_steps=1, initial_cell_count=1)
    assert np.isclose(sim._Nf_getter(0.0), 2.5)
    assert np.isclose(sim._Nf_getter(12.3), 2.5)
