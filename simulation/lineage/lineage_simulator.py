from __future__ import annotations

from typing import Callable, List

import numpy as np

from simulation.config.simulation_config import SimulationConfig
from simulation.models.cell import Cell
from simulation.models.gene import Gene
from simulation.lineage.partitioning import partition_mrna
from simulation.stochastic.tau_leap import tau_leap_step


def build_nf_getter(sim_config: SimulationConfig, genes: List[Gene]) -> Callable[[float], float]:
    min_nf = float(sim_config.Nf_min)
    if sim_config.Nf_mode == "operon_scaled":
        t_rep = np.array([g.t_rep for g in genes], dtype=np.float64)
        operons0 = float(len(genes))
        T_div = float(sim_config.T_div)
        Nf_birth = float(sim_config.Nf_birth)
        Nf_max = sim_config.Nf_max

        def nf(t: float) -> float:
            # Map to cell-cycle age to keep Nf periodic across divisions.
            age = float(t) % T_div
            copies = np.where(age > t_rep, 2.0, 1.0)
            operons = float(np.sum(copies))
            growth = 2.0 ** (age / T_div)
            val = Nf_birth * growth / (operons / operons0)
            val = max(val, min_nf)
            if Nf_max is not None:
                val = min(val, float(Nf_max))
            return val

        return nf

    if callable(sim_config.Nf_global):
        return lambda t: max(float(sim_config.Nf_global(t)), min_nf)
    Nf_const = max(float(sim_config.Nf_global), min_nf)
    return lambda _t, _c=Nf_const: _c


class LineageSimulator:
    def __init__(
        self,
        sim_config: SimulationConfig,
        genes: List[Gene],
        snapshot_interval_steps: int = 10,
        initial_cell_count: int = 3,
        division_jitter_cv: float = 0.05,
    ) -> None:
        if snapshot_interval_steps <= 0:
            raise ValueError("snapshot_interval_steps must be positive")
        if initial_cell_count <= 0:
            raise ValueError("initial_cell_count must be positive")
        
        self.sim_config = sim_config
        self.genes = genes
        self.snapshot_interval_steps = snapshot_interval_steps
        self.initial_cell_count = int(initial_cell_count)
        self.division_jitter_cv = max(0.0, division_jitter_cv) # Prevent negative jitter
        self.t_rep = np.array([g.t_rep for g in genes], dtype=np.float64) # Replication times of each gene dna part
        self.gamma_deg = np.array([g.gamma_deg for g in genes], dtype=np.float64)
        self.k_on_rnap = np.array([g.k_on_rnap for g in genes], dtype=np.float64)
        self.k_off_rnap = np.array([g.k_off_rnap for g in genes], dtype=np.float64)
        self.Gamma = np.array([g.Gamma_esc for g in genes], dtype=np.float64)
        self._Nf_getter = build_nf_getter(sim_config, genes)

    def compute_promoter_occ(self, N_f: float) -> np.ndarray:
        """Compute per-gene promoter occupancy for a given RNAP pool."""
        #  Computing O_i, the promoter occupancy for each gene i since we want stochastically 
        # And not a steady state. 
        denominator = 1.0 + (self.k_off_rnap + self.Gamma) / (self.k_on_rnap * N_f)
        return 1 / denominator

    def _transcription_propensity(self, cell: Cell, promoter_occ: np.ndarray) -> np.ndarray:
        """Compute per-gene transcription propensities using precomputed occupancy."""
        gene_copies = np.where(cell.age > self.t_rep, 2, 1)
        return gene_copies * self.Gamma * promoter_occ

    def _degradation_propensity(self, cell:Cell) -> np.ndarray:
        """Compute per-gene degradation propensities."""
        return cell.mrna * self.gamma_deg
    
    def _compute_mRNA_dynamic(self, N_f: float) -> np.ndarray:
        """Compute per-gene effective initiation rates for a given RNAP pool."""
        denominator = 1.0 + (self.k_off_rnap * self.Gamma) / (self.k_on_rnap * N_f)
        return (N_f * self.k_on_rnap * self.Gamma) / denominator

    def _calculate_cell_phase(self, age: float) -> str:
        """Determine cell phase based on age and division time."""
        if age < self.sim_config.B_period:
            return "B"
        elif age < self.sim_config.B_period + self.sim_config.C_period:
            return "C"
        else: 
            return "D"
        
    def _draw_division_time(self, rng: np.random.Generator) -> float:
        """Draw a cell-specific division time around the mean T_div."""
        base = self.sim_config.T_div
        if self.division_jitter_cv == 0.0:
            return base
        sigma = base * self.division_jitter_cv
        draw = rng.normal(loc=base, scale=sigma)
        # Prevent pathological, gives a 10 percent variation in T_div
        return float(np.clip(draw, 0.9 * base, 1.1 * base))
    
    def _make_row(self, cell: Cell) -> dict:
        cell_phase = self._calculate_cell_phase(cell.age)
        theta_rad = (cell.age / cell.division_time) * (2 * np.pi)
        row = {
            "cell_id": cell.cell_id,
            "parent_id": cell.parent_id,
            "generation": cell.generation,
            "age": cell.age,
            "phase": cell_phase,
            "theta_rad": theta_rad,
            }
        for idx, gene in enumerate(self.genes):
            row[gene.gene_id] = int(cell.mrna[idx])
        return row

    def _make_split_cell(self, updated_cells,rng: np.random.Generator, cell: Cell, next_cell_id: int) -> tuple[Cell, Cell, int]:
        d1, d2 = partition_mrna(cell.mrna, rng)
        updated_cells.append(
            self._make_cell(rng,
            cell_id=next_cell_id,
            parent_id=cell.cell_id,
            generation=cell.generation + 1,
            age=0.0,
            mrna=d1,
            ))
        next_cell_id += 1
        updated_cells.append(self._make_cell(
        rng,
        cell_id=next_cell_id,
        parent_id=cell.cell_id,
        generation=cell.generation + 1,
        age=0.0,
        mrna=d2,
        ))
        next_cell_id += 1
        return updated_cells ,next_cell_id

    def _make_cell(
        self,
        rng: np.random.Generator,
        cell_id: int,
        parent_id: int | None,
        generation: int,
        age: float,
        mrna: np.ndarray,
    ) -> Cell:
        return Cell(
            cell_id=cell_id,
            parent_id=parent_id,
            generation=generation,
            age=age,
            division_time=self._draw_division_time(rng),
            mrna=mrna,
        )

    def _init_cells(self, rng: np.random.Generator) -> list[Cell]:
        gene_copy = 1
        m0 = gene_copy * (self.Gamma / self.gamma_deg) * self._compute_mRNA_dynamic(self._Nf_getter(0))
        if self.initial_cell_count == 1:
            return [self._make_cell(rng, cell_id=0, parent_id=None, generation=0, age=0.0, mrna=m0)]

        cells: list[Cell] = []
        for cid in range(self.initial_cell_count):
            # Spread starting ages uniformly across the cell cycle
            age = float(rng.uniform(0.0, self.sim_config.T_div))
            cells.append(self._make_cell(rng, cell_id=cid, parent_id=None, generation=0, age=age, mrna=m0.copy()))
        return cells

    def run(self) -> list[dict]:
        rng = np.random.default_rng(self.sim_config.random_seed)
        cells: list[Cell] = self._init_cells(rng)
        next_cell_id = len(cells)
        snapshots: list[dict] = []
        step = 0
        while len(snapshots) < self.sim_config.N_target_samples:
            step += 1
            N_f = self._Nf_getter(step * self.sim_config.dt)
            promoter_occ = self.compute_promoter_occ(N_f)
            updated_cells: list[Cell] = []
            for cell in cells:
                seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
                new_mrna = tau_leap_step(
                    dt=self.sim_config.dt,
                    age=cell.age,
                    mrna=cell.mrna,
                    t_rep=self.t_rep,
                    trans_prop=self._transcription_propensity(cell, promoter_occ),
                    deg_prop=self._degradation_propensity(cell),
                    gamma_deg=self.gamma_deg,
                    max_mrna_per_gene=self.sim_config.MAX_MRNA_PER_GENE,
                    rng_seed=seed,
                )
                cell.mrna = new_mrna
                cell.step_age(self.sim_config.dt)

                if cell.age >= cell.division_time:
                    updated_cells, next_cell_id = self._make_split_cell(updated_cells, rng, cell, next_cell_id)
                else:
                    updated_cells.append(cell)
            cells = updated_cells

            if len(cells) >= self.sim_config.N_target_samples:
                for cell in cells:
                    row = self._make_row(cell)
                    snapshots.append(row)
                break # Exit if we have enough samples
        return snapshots
