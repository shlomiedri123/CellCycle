from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.tools.fit_measured_mrna_distribution import (
    bootstrap_lognormal,
    compute_measured_mrna,
    fit_lognormal,
    load_counts_csv,
    summarize_bootstrap,
)


def _write_counts_csv(path: Path, counts: np.ndarray, with_cell_ids: bool = False) -> None:
    n_cells, n_genes = counts.shape
    columns = [f"gene_{i}" for i in range(n_genes)]
    df = pd.DataFrame(counts, columns=columns)
    if with_cell_ids:
        df.insert(0, "cell_id", [f"cell_{i}" for i in range(n_cells)])
    df.to_csv(path, index=False)


def test_fit_recovers_log_normal_params(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    mu_true = 2.0
    sigma_true = 0.4
    s_vals = rng.lognormal(mean=mu_true, sigma=sigma_true, size=3000)
    s_counts = np.maximum(1, np.rint(s_vals)).astype(int)
    counts = s_counts.reshape(-1, 1)
    path = tmp_path / "counts.csv"
    _write_counts_csv(path, counts)

    counts_loaded, has_ids = load_counts_csv(path)
    assert not has_ids
    s_measured = compute_measured_mrna(counts_loaded, min_counts=1)
    mu_hat, sigma_hat = fit_lognormal(s_measured)

    assert abs(mu_hat - mu_true) < 0.2
    assert abs(sigma_hat - sigma_true) < 0.15


def test_load_counts_csv_detects_cell_ids(tmp_path: Path) -> None:
    counts = np.array([[1, 2], [3, 4]], dtype=int)
    path = tmp_path / "counts_ids.csv"
    _write_counts_csv(path, counts, with_cell_ids=True)

    counts_loaded, has_ids = load_counts_csv(path)
    assert has_ids
    np.testing.assert_array_equal(counts_loaded, counts)


def test_bootstrap_shapes_and_bounds() -> None:
    rng = np.random.default_rng(42)
    s_vals = rng.lognormal(mean=1.5, sigma=0.6, size=200)
    mu_samples, sigma_samples = bootstrap_lognormal(s_vals, 50, rng)

    assert mu_samples.shape == (50,)
    assert sigma_samples.shape == (50,)
    summary = summarize_bootstrap(mu_samples, sigma_samples)
    mu_ci = summary["mu"]["ci_95"]
    sigma_ci = summary["sigma"]["ci_95"]

    assert mu_ci[0] <= summary["mu"]["median"] <= mu_ci[1]
    assert sigma_ci[0] <= summary["sigma"]["median"] <= sigma_ci[1]
    assert mu_ci[0] >= float(mu_samples.min()) - 1e-8
    assert mu_ci[1] <= float(mu_samples.max()) + 1e-8
    assert sigma_ci[0] >= float(sigma_samples.min()) - 1e-8
    assert sigma_ci[1] <= float(sigma_samples.max()) + 1e-8


def test_cli_creates_outputs(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    s_vals = rng.lognormal(mean=1.8, sigma=0.5, size=250)
    s_counts = np.maximum(1, np.rint(s_vals)).astype(int)
    counts = s_counts.reshape(-1, 1)
    input_path = tmp_path / "counts.csv"
    _write_counts_csv(input_path, counts, with_cell_ids=True)
    outdir = tmp_path / "out"

    root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        "-m",
        "tools.fit_measured_mrna_distribution",
        "--input",
        str(input_path),
        "--outdir",
        str(outdir),
        "--bootstrap",
        "25",
        "--seed",
        "123",
    ]
    subprocess.check_call(cmd, cwd=root)

    params_path = outdir / "measured_mrna_distribution.json"
    assert params_path.exists()
    plots_dir = outdir / "plots"
    assert (plots_dir / "measured_mrna_hist_fit.png").exists()
    assert (plots_dir / "measured_mrna_qq.png").exists()
    assert (plots_dir / "measured_mrna_cdf.png").exists()

    with params_path.open("r", encoding="utf-8") as f:
        params = json.load(f)
    for key in ("mu", "sigma", "bootstrap", "n_cells", "min_counts", "seed"):
        assert key in params
