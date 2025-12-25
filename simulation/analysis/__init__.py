"""Analysis helpers for TRIP profiles and age distributions."""

from .age_distribution import plot_age_distribution
from .m_profiles import make_nf_getter, solve_mRNA_exact, steady_profile_constant_nf
from .trip_profile import mean_adjusted_trip, mean_expression_by_bin, plot_trip_profiles_grid

__all__ = [
    "mean_adjusted_trip",
    "mean_expression_by_bin",
    "plot_trip_profiles_grid",
    "plot_age_distribution",
    "make_nf_getter",
    "steady_profile_constant_nf",
    "solve_mRNA_exact",
]
