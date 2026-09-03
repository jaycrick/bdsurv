"""bdsurv: survival-conditioned simulation for subcritical linear birth-death processes.

Two independent methods for sampling a birth-death tree conditioned on
survival to a terminal time ``T``:

- ``simulate_birth_death_process_from_contour_process`` — exact, via
  contour-process reconstruction (see ``docs/contour_process.md``).
- ``simulate_subcritical_survival_splitting`` — approximate, via fixed-factor
  (RESTART) multilevel splitting (see ``docs/splitting.md``).

Both return a ``SimulatedTree``: a plain dataclass wrapping the individual-
level records DataFrame plus the rates and terminal time used to generate it.
"""

from bdsurv.contour import simulate_birth_death_process_from_contour_process
from bdsurv.splitting import (
    ConditionedTrajectory,
    LevelDiagnostic,
    SplittingConfig,
    SplittingSimulationResult,
    select_level_times_for_band,
    simulate_subcritical_survival_splitting,
)
from bdsurv.tree import SimulatedTree

__version__ = "0.1.0"
__author__ = "Jacob Cumming"

__all__ = [
    "ConditionedTrajectory",
    "LevelDiagnostic",
    "SimulatedTree",
    "SplittingConfig",
    "SplittingSimulationResult",
    "select_level_times_for_band",
    "simulate_birth_death_process_from_contour_process",
    "simulate_subcritical_survival_splitting",
]
