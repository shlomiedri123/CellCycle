"""Gene-level parameters for RNAP-limited transcription.

Stores per-gene kinetic rates that drive promoter occupancy and mRNA decay.
These parameters feed the stochastic birth-death process in the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

# Need to find the area of valid parameters so when we generate random number we won't leave the 
# valid area 
@dataclass(frozen=True)
class GeneConfig:
    gene_id: str # Tracking gene by its unique identifier
    chrom_pos_bp: float # Chromosomal position in base pairs -> Maybe it is irrelevant. 
    k_on_rnap: float # Rate of RNAP binding
    k_off_rnap: float # Rate of RNAP unbinding
    Gamma_esc: float #Promoter escape rate 
    gamma_deg: float  # Degradation rate of mRNA
    phase: str | None = None

    def validate(self) -> None:
        if self.chrom_pos_bp < 0:
            raise ValueError(f"chrom_pos_bp must be non-negative for {self.gene_id}")
        if min(self.k_on_rnap, self.k_off_rnap, self.Gamma_esc, self.gamma_deg) < 0:
            raise ValueError(f"Rates must be positive for {self.gene_id}")
        if self.phase:
            phase_key = str(self.phase).strip().upper()
            if phase_key not in {"I", "II"}:
                raise ValueError(f"phase must be 'I' or 'II' for {self.gene_id}")
