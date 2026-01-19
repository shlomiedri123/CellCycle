"""Configuration dataclasses for simulation runtime and gene parameters.

These dataclasses define the inputs required by the stochastic simulator.
"""

from .simulation_config import SimulationConfig
from .gene_config import GeneConfig

__all__ = ["SimulationConfig", "GeneConfig"]
