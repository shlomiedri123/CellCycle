from __future__ import annotations

from dataclasses import dataclass
# Should we add i

@dataclass(frozen=True)
class Gene:
    gene_id: str
    chrom_pos_bp: float
    gamma_deg: float
    Gamma_esc: float
    t_rep: float
    k_on_rnap: float
    k_off_rnap: float


