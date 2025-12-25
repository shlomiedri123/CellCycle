from __future__ import annotations

import pathlib
import math
import yaml

from simulation.config.simulation_config import SimulationConfig


def load_simulation_config(path: str | pathlib.Path) -> SimulationConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    def _to_float(val):
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            stripped = val.strip()
            if stripped == "":
                return None
            return float(stripped)
        raise TypeError(f"Expected numeric value or None, got {type(val)}")

    nf_mode = raw.get("Nf_mode", "provided")
    # Rebuild Nf_global callable if specified parametrically
    nf_type = raw.pop("Nf_type", None)
    if nf_mode == "provided" and nf_type == "sine":
        base = float(raw.pop("Nf_base", 1.0))
        amp = float(raw.pop("Nf_amp", 0.0))
        phase = float(raw.pop("Nf_phase", 0.0))
        period = float(raw.pop("Nf_period", raw.get("T_div", 1.0)))
        floor = float(raw.get("Nf_min", 1e-9))  # prevent non-physical negative/zero free RNAP

        def nf_func(t: float, *, _b=base, _a=amp, _p=phase, _per=period, _floor=floor) -> float:
            val = _b + _a * math.sin((2.0 * math.pi * t) / _per + _p)
            return max(val, _floor)

        raw["Nf_global"] = nf_func
    else:
        raw.pop("Nf_base", None)
        raw.pop("Nf_amp", None)
        raw.pop("Nf_phase", None)
        raw.pop("Nf_period", None)
    if nf_mode != "provided":
        raw.setdefault("Nf_global", None)
    raw.setdefault("Nf_mode", nf_mode)
    raw["Nf_birth"] = _to_float(raw.get("Nf_birth", 1.0)) or 1.0
    raw["Nf_min"] = _to_float(raw.get("Nf_min", 1e-9)) or 1e-9
    raw["Nf_max"] = _to_float(raw.get("Nf_max", None))
    raw.setdefault("mode", "stochastic")
    return SimulationConfig(**raw)
