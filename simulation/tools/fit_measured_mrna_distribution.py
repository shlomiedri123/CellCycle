"""Fit empirical distribution P(S) of measured mRNA counts per cell.

Scientific meaning:
S_c = sum_g N_{cg}, where N_{cg} is the observed count matrix (cells x genes).
S_c is the measured number of mRNA molecules in cell c.

In the CellCycleNonlinear model, S = N(t) * D. Here we integrate out hidden
variables (cell age, growth, depth) and infer P(S) directly from data as a
log-normal distribution. P(S) is later used to draw the number of molecules
captured per simulated cell and controls sparsity / measurement noise.

This module is row-only and does not perform any gene-level or kinetic inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

DEFAULT_MIN_COUNTS = 1
DEFAULT_BOOTSTRAP = 500
DEFAULT_CHUNK_SIZE = 2000
LARGE_FILE_BYTES = 200 * 1024 * 1024


def _is_numeric_series(series: pd.Series) -> bool:
    coerced = pd.to_numeric(series, errors="coerce")
    return not coerced.isna().any()


def _detect_separator(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        sample = ""
        for line in handle:
            if line.strip():
                sample = line
                break
    if not sample:
        raise ValueError(f"Input CSV is empty: {path}")

    candidates = [",", "\t", ";"]
    counts = {sep: sample.count(sep) for sep in candidates}
    best_sep = max(counts, key=counts.get)
    if counts[best_sep] == 0:
        if " " in sample:
            return r"\s+"
        return ","
    return best_sep


def _read_counts_dataframe(path: Path, separator: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=separator,
        header=0,
        encoding="utf-8",
        encoding_errors="replace",
        na_values=["", "NA", "NaN"],
        keep_default_na=True,
        engine="python",
        on_bad_lines="skip",
        quoting=csv.QUOTE_NONE,
    )


def _should_chunk(path: Path, chunksize: int | None) -> bool:
    if chunksize is not None:
        return True
    try:
        size = path.stat().st_size
    except OSError:
        return False
    return size >= LARGE_FILE_BYTES


def load_counts_csv(path: Path, separator: str | None = None) -> Tuple[np.ndarray, bool]:
    """Load a cells x genes count matrix, detecting optional cell ID column."""
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    if separator is None:
        separator = _detect_separator(path)
        logger.info("Auto-detected separator %r.", separator)
    else:
        logger.info("Using provided separator %r.", separator)

    df = _read_counts_dataframe(path, separator)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {path}")
    if df.shape[1] == 0:
        raise ValueError(f"Input CSV has no columns: {path}")
    has_cell_ids = not _is_numeric_series(df.iloc[:, 0])
    if has_cell_ids:
        logger.info("Detected non-numeric first column; treating as cell IDs.")
        df_counts = df.iloc[:, 1:]
    else:
        logger.info("First column appears numeric; treating as gene counts.")
        df_counts = df

    if df_counts.shape[1] == 0:
        raise ValueError("No gene columns found after removing cell IDs.")

    df_counts = df_counts.apply(pd.to_numeric, errors="raise")
    counts = df_counts.to_numpy()
    if not np.isfinite(counts).all():
        raise ValueError("Counts contain NaN or inf values.")
    if np.any(counts < 0):
        raise ValueError("Counts must be non-negative.")
    if not np.all(np.isclose(counts, np.round(counts))):
        raise ValueError("Counts must be integers.")
    return counts, has_cell_ids


def compute_measured_mrna_from_csv(
    path: Path,
    min_counts: int,
    separator: str | None = None,
    chunksize: int = DEFAULT_CHUNK_SIZE,
    log_every_chunks: int = 10,
) -> Tuple[np.ndarray, bool]:
    """Stream CSV in chunks and return filtered S_c values and cell ID flag."""
    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")
    if min_counts < 0:
        raise ValueError("min_counts must be non-negative.")
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    if separator is None:
        separator = _detect_separator(path)
        logger.info("Auto-detected separator %r.", separator)
    else:
        logger.info("Using provided separator %r.", separator)

    reader = pd.read_csv(
        path,
        sep=separator,
        header=0,
        encoding="utf-8",
        encoding_errors="replace",
        na_values=["", "NA", "NaN"],
        keep_default_na=True,
        engine="python",
        on_bad_lines="skip",
        quoting=csv.QUOTE_NONE,
        chunksize=chunksize,
    )

    s_chunks: list[np.ndarray] = []
    total_rows = 0
    kept_rows = 0
    has_cell_ids: bool | None = None
    start = time.monotonic()

    for idx, chunk in enumerate(reader, start=1):
        if chunk.empty:
            continue
        if has_cell_ids is None:
            has_cell_ids = not _is_numeric_series(chunk.iloc[:, 0])
            if has_cell_ids:
                logger.info("Detected non-numeric first column; treating as cell IDs.")
            else:
                logger.info("First column appears numeric; treating as gene counts.")

        df_counts = chunk.iloc[:, 1:] if has_cell_ids else chunk
        if df_counts.shape[1] == 0:
            raise ValueError("No gene columns found after removing cell IDs.")

        df_counts = df_counts.apply(pd.to_numeric, errors="coerce")
        values = df_counts.to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("Counts contain NaN or inf values.")
        if np.any(values < 0):
            raise ValueError("Counts must be non-negative.")
        if not np.all(np.isclose(values, np.round(values))):
            raise ValueError("Counts must be integers.")

        s_chunk = values.sum(axis=1)
        total_rows += values.shape[0]
        s_chunk = s_chunk[s_chunk >= min_counts]
        s_chunk = s_chunk[s_chunk > 0]
        kept_rows += s_chunk.size
        s_chunks.append(s_chunk)

        if log_every_chunks and idx % log_every_chunks == 0:
            elapsed = time.monotonic() - start
            logger.info(
                "Loaded %d rows, kept %d cells after filtering (%.1fs)",
                total_rows,
                kept_rows,
                elapsed,
            )

    if has_cell_ids is None:
        raise ValueError(f"Input CSV is empty: {path}")

    s_values = np.concatenate(s_chunks) if s_chunks else np.array([], dtype=float)
    logger.info("Finished loading %d rows; kept %d cells.", total_rows, kept_rows)
    return s_values, has_cell_ids


def compute_measured_mrna(counts: np.ndarray, min_counts: int) -> np.ndarray:
    """Compute S_c per cell and apply min_counts filtering."""
    if min_counts < 0:
        raise ValueError("min_counts must be non-negative.")
    s_values = counts.sum(axis=1)
    s_filtered = s_values[s_values >= min_counts]
    s_filtered = s_filtered[s_filtered > 0]
    return s_filtered


def fit_lognormal(s_values: np.ndarray) -> Tuple[float, float]:
    """Fit log-normal parameters to positive S values (floc=0)."""
    if s_values.size == 0:
        raise ValueError("No positive S values available for fitting.")
    shape, _, scale = stats.lognorm.fit(s_values, floc=0)
    sigma = float(shape)
    mu = float(np.log(scale))
    return mu, sigma


def bootstrap_lognormal(
    s_values: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    log_every: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap log-normal fits by resampling S values with replacement."""
    if n_boot <= 0:
        raise ValueError("bootstrap iterations must be positive.")
    n = s_values.size
    if n == 0:
        raise ValueError("Cannot bootstrap with zero samples.")
    mu_samples = np.empty(n_boot, dtype=float)
    sigma_samples = np.empty(n_boot, dtype=float)
    start = time.monotonic()
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = s_values[idx]
        mu_i, sigma_i = fit_lognormal(sample)
        mu_samples[i] = mu_i
        sigma_samples[i] = sigma_i
        if log_every and (i + 1) % log_every == 0:
            elapsed = time.monotonic() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            remaining = (n_boot - (i + 1)) / rate if rate > 0 else float("inf")
            logger.info(
                "Bootstrap %d/%d (%.1f%%) elapsed %.1fs, est. remaining %.1fs",
                i + 1,
                n_boot,
                100.0 * (i + 1) / n_boot,
                elapsed,
                remaining,
            )
    return mu_samples, sigma_samples


def summarize_bootstrap(mu_samples: np.ndarray, sigma_samples: np.ndarray) -> Dict[str, Dict[str, object]]:
    """Summarize bootstrap samples with median and 95% interval."""
    mu_q = np.quantile(mu_samples, [0.025, 0.5, 0.975])
    sigma_q = np.quantile(sigma_samples, [0.025, 0.5, 0.975])
    return {
        "mu": {"median": float(mu_q[1]), "ci_95": [float(mu_q[0]), float(mu_q[2])]},
        "sigma": {"median": float(sigma_q[1]), "ci_95": [float(sigma_q[0]), float(sigma_q[2])]},
    }


def _logspace(values: np.ndarray, num: int = 200) -> np.ndarray:
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if min_val <= 0:
        raise ValueError("Log-space plotting requires positive values.")
    if min_val == max_val:
        min_val *= 0.9
        max_val *= 1.1
    return np.logspace(np.log10(min_val), np.log10(max_val), num=num)


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def plot_hist_fit(s_values: np.ndarray, mu: float, sigma: float, outpath: Path) -> None:
    """Plot histogram with fitted log-normal PDF overlay."""
    plt = _get_pyplot()
    x_vals = _logspace(s_values, num=300)
    bins = _logspace(s_values, num=40)
    pdf_vals = stats.lognorm.pdf(x_vals, s=sigma, scale=np.exp(mu))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(s_values, bins=bins, density=True, alpha=0.6, color="C0")
    ax.plot(x_vals, pdf_vals, color="C1", linewidth=2.0, label="Log-normal fit")
    ax.set_xscale("log")
    ax.set_xlabel("Measured mRNA per cell (S_c)")
    ax.set_ylabel("Density")
    ax.set_title("Measured mRNA distribution with log-normal fit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_qq(s_values: np.ndarray, mu: float, sigma: float, outpath: Path) -> None:
    """Plot QQ comparison between empirical and fitted log-normal."""
    plt = _get_pyplot()
    sorted_vals = np.sort(s_values)
    n = sorted_vals.size
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = stats.lognorm.ppf(probs, s=sigma, scale=np.exp(mu))

    min_val = float(min(theo.min(), sorted_vals.min()))
    max_val = float(max(theo.max(), sorted_vals.max()))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(theo, sorted_vals, s=10, alpha=0.6, color="C0")
    ax.plot([min_val, max_val], [min_val, max_val], color="C1", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Theoretical quantiles (log-normal)")
    ax.set_ylabel("Empirical quantiles")
    ax.set_title("QQ plot: empirical vs fitted log-normal")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_cdf(s_values: np.ndarray, mu: float, sigma: float, outpath: Path) -> None:
    """Plot empirical CDF vs fitted log-normal CDF (log x-axis)."""
    plt = _get_pyplot()
    sorted_vals = np.sort(s_values)
    n = sorted_vals.size
    empirical = np.arange(1, n + 1) / n

    x_vals = _logspace(sorted_vals, num=300)
    fitted = stats.lognorm.cdf(x_vals, s=sigma, scale=np.exp(mu))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(sorted_vals, empirical, drawstyle="steps-post", label="Empirical", color="C0")
    ax.plot(x_vals, fitted, label="Log-normal fit", color="C1", linewidth=2.0)
    ax.set_xscale("log")
    ax.set_xlabel("Measured mRNA per cell (S_c)")
    ax.set_ylabel("CDF")
    ax.set_title("Empirical vs fitted CDF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def fit_measured_mrna_distribution(
    input_path: Path,
    outdir: Path,
    min_counts: int = DEFAULT_MIN_COUNTS,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    separator: str | None = None,
    chunksize: int | None = None,
    seed: int | None = None,
) -> Dict[str, object]:
    """Fit P(S) from an experimental counts CSV and write params/plots to disk."""
    if _should_chunk(input_path, chunksize):
        if chunksize is None:
            chunksize = DEFAULT_CHUNK_SIZE
            logger.info("Large input detected; streaming with chunksize=%d.", chunksize)
        s_values, has_cell_ids = compute_measured_mrna_from_csv(
            input_path,
            min_counts=min_counts,
            separator=separator,
            chunksize=chunksize,
        )
    else:
        counts, has_cell_ids = load_counts_csv(input_path, separator=separator)
        logger.info("Loaded counts matrix with shape %s", counts.shape)
        if has_cell_ids:
            logger.info("Cell ID column removed; %d gene columns retained.", counts.shape[1])
        s_values = compute_measured_mrna(counts, min_counts=min_counts)

    if s_values.size == 0:
        raise ValueError("No cells remain after filtering; adjust min_counts or input data.")
    logger.info("Computed S_c for %d cells after filtering.", s_values.size)

    mu, sigma = fit_lognormal(s_values)
    logger.info("Fitted log-normal parameters: mu=%.4f, sigma=%.4f", mu, sigma)

    rng = np.random.default_rng(seed)
    log_every = max(1, min(50, bootstrap // 10)) if bootstrap >= 10 else 0
    mu_samples, sigma_samples = bootstrap_lognormal(s_values, bootstrap, rng, log_every=log_every)
    bootstrap_summary = summarize_bootstrap(mu_samples, sigma_samples)
    logger.info(
        "Bootstrap median mu=%.4f (%.4f, %.4f), sigma=%.4f (%.4f, %.4f)",
        bootstrap_summary["mu"]["median"],
        bootstrap_summary["mu"]["ci_95"][0],
        bootstrap_summary["mu"]["ci_95"][1],
        bootstrap_summary["sigma"]["median"],
        bootstrap_summary["sigma"]["ci_95"][0],
        bootstrap_summary["sigma"]["ci_95"][1],
    )

    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_hist_fit(s_values, mu, sigma, plots_dir / "measured_mrna_hist_fit.png")
    plot_qq(s_values, mu, sigma, plots_dir / "measured_mrna_qq.png")
    plot_cdf(s_values, mu, sigma, plots_dir / "measured_mrna_cdf.png")

    params = {
        "mu": float(mu),
        "sigma": float(sigma),
        "bootstrap": {
            "iterations": int(bootstrap),
            "mu": bootstrap_summary["mu"],
            "sigma": bootstrap_summary["sigma"],
        },
        "n_cells": int(s_values.size),
        "min_counts": int(min_counts),
        "seed": seed,
    }
    params_path = outdir / "measured_mrna_distribution.json"
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, sort_keys=True)

    logger.info("Wrote parameters to %s", params_path)
    logger.info("Saved plots to %s", plots_dir)
    return params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit empirical log-normal P(S) of measured mRNA per cell from a CSV count matrix."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input counts CSV (cells x genes).")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for parameters and plots.")
    parser.add_argument("--min_counts", type=int, default=DEFAULT_MIN_COUNTS, help="Minimum S_c per cell.")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP, help="Bootstrap iterations.")
    parser.add_argument(
        "--separator",
        type=str,
        default=None,
        help="Field separator (default: auto-detect from header row).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Rows per chunk for streaming large CSVs (default: auto for large files).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for bootstrap sampling.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    fit_measured_mrna_distribution(
        input_path=args.input,
        outdir=args.outdir,
        min_counts=args.min_counts,
        bootstrap=args.bootstrap,
        separator=args.separator,
        chunksize=args.chunksize,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
