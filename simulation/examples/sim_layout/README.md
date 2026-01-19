Minimal simulation layout

This folder is a lightweight example of the files needed to run the simulator:
- sim_config.yaml
- random_genes.csv
- nf_vector.npy
- S_params.json

Generate outputs by running:
python -m simulation.run_simulation --config simulation/examples/sim_layout/sim_config.yaml

To regenerate these inputs with the random data generator:
python -m simulation.tools.random_gene_data --n_genes 20 --n_samples 100000 --out_dir simulation/examples/sim_layout
The generator writes `random_sim_config.yaml`; rename it to `sim_config.yaml`
if you want to keep the same entrypoint name.
