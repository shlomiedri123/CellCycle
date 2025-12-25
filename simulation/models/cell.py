from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Question : Should we add an age limit for the cell? to make sure none of them explode and we do have 
# Some limit for the age 

@dataclass
class Cell:
    cell_id: int
    parent_id: Optional[int]
    generation: int
    age: float
    division_time: float
    mrna: np.ndarray = field(repr=False)
    
    def step_age(self, dt: float) -> None:
        self.age += dt
    
    def snapshot(self) -> dict:
        """Return lightweight snapshot dictionary for serialization."""
        return {
            "cell_id": self.cell_id,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "age": self.age,
            "mrna": self.mrna.copy(),
        }
