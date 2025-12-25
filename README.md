# TRIP / scRNA-seq Bacterial Simulator

This repository simulates lineage-resolved bacterial gene expression with RNAP-limited initiation, stochastic tau-leaping updates, and Cooper-Helmstetter B/C/D timing. It also includes utilities to generate random gene/config tables and to learn the empirical distribution of measured mRNA counts per cell, P(S), from experimental data.

## Install

Recommended dependencies:

- numpy
- scipy
- pandas
- matplotlib
- pyyaml

For the stochastic kernel, build the C++ extension once per environment:

```bash
python simulation/kernels/setup.py build_ext --inplace
```

## Generate Random Config + Genes

```bash
python -m tools.random_gene_data   --n_genes 1000   --n_samples 100000   --out_dir out/configs   --seed 1
```

Outputs `random_genes.csv`, `random_sim_config.yaml`, and `random_hidden_params.json` in `out/configs`.

## Run Simulation

```bash
python -m lineage.lineage_simulator   --config out/configs/random_sim_config.yaml   --genes out/configs/random_genes.csv   --out out/sim_run/snapshots.csv
```

Optional parsed snapshots (measurement model):

```bash
python -m lineage.lineage_simulator   --config out/configs/random_sim_config.yaml   --genes out/configs/random_genes.csv   --out out/sim_run/snapshots.csv   --measured_dist out/measured_mrna_dist/measured_mrna_distribution.json
```

## Fit Measured mRNA Distribution P(S)

```bash
python -m tools.fit_measured_mrna_distribution   --input data/ecoli_counts.csv   --outdir out/measured_mrna_dist   --min_counts 1   --bootstrap 500   --seed 42
```

## Plot Age Distribution

```bash
python -m tools.plot_age_distribution   --snapshots out/sim_run/snapshots.csv   --config out/configs/random_sim_config.yaml   --out out/sim_run/age_distribution.png
```

## Plot TRIP Profiles (Grid)

```bash
python -m tools.plot_trip_profiles   --snapshots out/sim_run/snapshots.csv   --genes out/configs/random_genes.csv   --config out/configs/random_sim_config.yaml   --out out/sim_run/trip_profiles_grid.png   --max_genes 100
```

## Plot Nf(t)

```bash
python -m tools.plot_nf_profile   --config out/configs/random_sim_config.yaml   --genes out/configs/random_genes.csv   --out out/sim_run/nf_profile.png
```

## Nf(t) Modes

`Nf_mode: provided` (default) uses a constant or YAML-defined sine function.

```yaml
Nf_mode: provided
Nf_type: sine
Nf_base: 1.5
Nf_amp: 0.2
Nf_phase: 0.0
Nf_period: 40.0
```

`Nf_mode: operon_scaled` generates Nf(t) from growth and operon copy number:

```yaml
Nf_mode: operon_scaled
Nf_birth: 1.5
Nf_min: 1e-9
Nf_max: 10.0
```
