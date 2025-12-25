from __future__ import annotations

import csv
import json
import pathlib
from typing import List, Mapping, Sequence

import numpy as np


def save_snapshots_csv(rows: Sequence[Mapping[str, object]], path: str | pathlib.Path) -> None:
    if not rows:
        raise ValueError("No snapshot rows to write")
    base_fields = ["cell_id", "parent_id", "generation", "age", "theta_rad", "phase"]
    gene_fields: List[str] = []
    for key in rows[0].keys():
        if key not in base_fields:
            gene_fields.append(str(key))
    fieldnames = base_fields + gene_fields

    path_obj = pathlib.Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_snapshots_csv(path: str | pathlib.Path) -> list[dict[str, object]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No snapshot rows found in {path}")
    return rows


def load_measured_mrna_distribution(path: str | pathlib.Path) -> tuple[float, float]:
    """Load log-normal parameters (mu, sigma) for measured mRNA distribution."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "mu" not in payload or "sigma" not in payload:
        raise ValueError("Measured mRNA distribution JSON must contain 'mu' and 'sigma'.")
    return float(payload["mu"]), float(payload["sigma"])


def apply_measurement_model(
    rows: Sequence[Mapping[str, object]],
    gene_ids: Sequence[str],
    mu: float,
    sigma: float,
    seed: int,
) -> list[dict]:
    """Generate parsed snapshots by subsampling counts per cell from log-normal S."""
    rng = np.random.default_rng(seed)
    base_fields = ["cell_id", "parent_id", "generation", "age", "theta_rad", "phase"]
    parsed_rows: list[dict] = []

    for row in rows:
        base = {field: row[field] for field in base_fields}
        counts = np.array([float(row[gid]) for gid in gene_ids], dtype=float)
        if np.any(counts < 0):
            raise ValueError("Snapshot counts must be non-negative.")
        counts = np.nan_to_num(counts, nan=0.0)
        if not np.all(np.isclose(counts, np.round(counts))):
            raise ValueError("Snapshot counts must be integers.")
        counts_int = counts.astype(int)
        total_int = int(np.sum(counts_int))

        if total_int <= 0:
            measured_counts = np.zeros_like(counts, dtype=int)
        else:
            s_draw = rng.lognormal(mean=mu, sigma=sigma)
            s_int = int(np.floor(s_draw + 0.5))
            if s_int < 0:
                s_int = 0
            if s_int > total_int:
                s_int = total_int
            if s_int == 0:
                measured_counts = np.zeros_like(counts_int, dtype=int)
            else:
                measured_counts = _sample_without_replacement(counts_int, s_int, rng)

        for gid, val in zip(gene_ids, measured_counts):
            base[gid] = int(val)
        parsed_rows.append(base)
    return parsed_rows


def _sample_without_replacement(
    counts: np.ndarray,
    n_sample: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample without replacement from per-gene counts (multivariate hypergeometric)."""
    if n_sample < 0:
        raise ValueError("n_sample must be non-negative.")
    if counts.ndim != 1:
        raise ValueError("counts must be a 1D array.")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative.")
    total = int(np.sum(counts))
    if total == 0 or n_sample == 0:
        return np.zeros_like(counts, dtype=int)
    if n_sample > total:
        n_sample = total

    remaining = total
    remaining_sample = n_sample
    out = np.zeros_like(counts, dtype=int)
    for idx, k in enumerate(counts):
        k = int(k)
        if remaining_sample <= 0:
            break
        if k <= 0:
            remaining -= k
            continue
        draw = rng.hypergeometric(ngood=k, nbad=remaining - k, nsample=remaining_sample)
        out[idx] = draw
        remaining_sample -= draw
        remaining -= k
    return out
