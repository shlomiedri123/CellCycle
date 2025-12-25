from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

@dataclass(frozen=True)
class SimulationConfig:
    B_period: float
    C_period: float
    D_period: float
    dt: float
    N_target_cells: int
    N_target_samples: int
    random_seed: int
    chromosome_length_bp: float
    Nf_global: float | Callable[[float], float] | None  # Provided Nf(t) for Nf_mode="provided"
    MAX_MRNA_PER_GENE: int # to avoid infinite mRNA number in case of bugs
    Nf_mode: str = "provided"
    Nf_birth: float = 1.0
    Nf_min: float = 1e-9
    Nf_max: float | None = None
    T_div: float | None = field(default=None)
    mode: str = field(default="stochastic")

    def __post_init__(self) -> None:
        computed_t_div = self.B_period + self.C_period + self.D_period
        object.__setattr__(self, "T_div", computed_t_div if self.T_div is None else self.T_div)
        if self.T_div is not None and abs(self.T_div - computed_t_div) > 1e-9:
            raise ValueError("T_div must equal B_period + C_period + D_period")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.N_target_cells <= 0:
            raise ValueError("N_target_cells must be positive")
        if self.N_target_samples <= 0:
            raise ValueError("N_target_samples must be positive")
        if self.chromosome_length_bp <= 0:
            raise ValueError("chromosome_length_bp must be positive")
        if self.Nf_mode not in ("provided", "operon_scaled"):
            raise ValueError("Nf_mode must be 'provided' or 'operon_scaled'")
        if self.Nf_mode == "provided":
            if not (isinstance(self.Nf_global, (int, float)) or callable(self.Nf_global)):
                raise ValueError("Nf_global must be a number or a callable(age)->float when Nf_mode='provided'")
            if isinstance(self.Nf_global, (int, float)):
                if self.Nf_global <= 0:
                    raise ValueError("Nf_global must be positive")
        else:
            if self.Nf_birth <= 0:
                raise ValueError("Nf_birth must be positive")
            if self.Nf_min <= 0:
                raise ValueError("Nf_min must be positive")
            if self.Nf_max is not None and float(self.Nf_max) < float(self.Nf_min):
                raise ValueError("Nf_max must be >= Nf_min")
        if self.MAX_MRNA_PER_GENE <= 0:
            raise ValueError("MAX_MRNA_PER_GENE must be positive")
        if self.mode != "stochastic":
            raise ValueError("mode must be 'stochastic'")
