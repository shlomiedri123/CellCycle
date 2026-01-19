"""Binomial partitioning of mRNA at cell division.

Each transcript is independently assigned to one of the two daughter cells.
"""

from __future__ import annotations

import numpy as np


def partition_mrna(mrna: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Partition mRNA counts binomially between two daughters."""
    #Spliting the transcription profile with binomial distribution between the two daughter cells
    mrna = mrna.astype(np.int64, copy=False)
    draws = rng.binomial(mrna, 0.5)
    daughter1 = draws.astype(np.int64, copy=False)
    daughter2 = (mrna - draws).astype(np.int64, copy=False)
    return daughter1, daughter2
