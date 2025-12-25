# Mathematical Model Overview

This document summarizes the full stochastic model implemented in the `simulation` package: Cooper–Helmstetter B–C–D structure, RNAP-limited initiation, τ-leaping birth–death dynamics, and lineage growth with binomial partitioning.

## Cell-Cycle Timing (Cooper–Helmstetter)
- Global periods: `B_period`, `C_period`, `D_period`, `T_div = B + C + D`.
- Single replication round per cycle.
- Genome length: `chromosome_length_bp`.
- For gene *g* at position `chrom_pos_bp[g]`:
  - Fractional position: `x_g = chrom_pos_bp[g] / chromosome_length_bp`.
  - Replication time: `t_rep[g] = B_period + C_period * x_g`.
- Copy number:
  - `copies_g(a) = 1` if `age < t_rep[g]`, else `2`.
  - After division (age = 0), all genes start with one copy.

## RNAP-Limited Initiation
Per gene parameters: `k_on_rnap[g]`, `k_off_rnap[g]`, `Gamma[g]`, `gamma_mrna[g]`.

Shared pool: `Nf_global` (free RNAP).

Effective initiation per copy:
```
gamma_eff[g] = (Nf_global * k_on_rnap[g] * Gamma[g]) /
               (1 + (k_off_rnap[g] * Gamma[g]) / (k_on_rnap[g] * Nf_global))
```
Birth rate for a cell of age `a` (gamma_eff is recomputed each step from the current free RNAP `N_f`):
```
lambda_birth_g(a) = copies_g(a) * gamma_eff[g]
```

## τ-Leaping Birth–Death Update
For timestep `dt`, gene *g*, and current mRNA `m_g`:
- Births: `Poisson(lambda_birth_g(a) * dt)`
- Deaths: `Poisson(gamma_mrna[g] * m_g * dt)`
- Clamp: `deaths ≤ m_g`, `0 ≤ m_g ≤ MAX_MRNA_PER_GENE`
- Update: `m_g ← m_g + births - deaths`

## Lineage Dynamics
Initialization:
- One founder cell: `age = 0`, `mRNA[g] = 0`, `generation = 0`.

Per global step:
1. Run τ-leaping for all genes.
2. Increment age: `age += dt`.
3. If `age ≥ T_div`, divide:
   - Partition each gene binomially: `daughter1 ~ Binomial(m_parent, 0.5)`, `daughter2 = m_parent - daughter1`.
   - Reset ages to 0, increment generation, assign new cell IDs.

Stopping rule:
- Continue until alive cells `>= N_target_cells`.
- Take population snapshots every `snapshot_interval_steps` until total stored rows `>= N_target_samples`.

## Theta and Phase Annotation
- `theta_raw = (age / T_div) * 2π`
- `theta_rad = (theta_raw + π) mod 2π` (wrap to `[0, 2π)` with a π shift)
- Phase by age:
  - `B` if `age < B_period`
  - `C` if `B_period ≤ age < B_period + C_period`
  - `D` otherwise

## Snapshot Dataset
Each snapshot row contains:
- `cell_id`, `parent_id`, `generation`, `age`, `theta_rad`, `phase`
- Per-gene integer mRNA counts.

## Assumptions and Exclusions
- No promoter ON/OFF telegraph model; initiation limited solely by RNAP pool.
- One replication round per cell cycle; copy number step-change at `t_rep`.
- Partitioning is symmetric binomial (p = 0.5) at division.
- τ-leaping, replication-dependent copy number, and lineage partitioning occur every step.***
