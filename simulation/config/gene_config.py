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

    def validate(self) -> None:
        if self.chrom_pos_bp < 0:
            raise ValueError(f"chrom_pos_bp must be non-negative for {self.gene_id}")
        if min(self.k_on_rnap, self.k_off_rnap, self.Gamma_esc, self.gamma_deg) < 0:
            raise ValueError(f"Rates must be positive for {self.gene_id}")

