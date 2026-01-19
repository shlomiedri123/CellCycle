"""Build script for the tau-leaping C++ extension."""

from __future__ import annotations

import pathlib

from setuptools import Extension, setup

try:
    import pybind11  # type: ignore
except ImportError as exc:
    raise SystemExit("pybind11 is required to build the tau_kernel extension") from exc


here = pathlib.Path(__file__).parent.resolve()

ext_modules = [
    Extension(
        "simulation.kernels.tau_kernel",
        sources=[str(here / "tau_kernel.cpp"), str(here / "bindings.cpp")],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3"],
    )
]


setup(
    name="simulation-kernels",
    version="0.1.0",
    description="Tau-leaping kernel for bacterial transcription simulation",
    ext_modules=ext_modules,
)
