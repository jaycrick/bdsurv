"""bdsurv: simulation of birth-death processes, focused on subcritical survival.

Survival-conditioned simulation of subcritical linear birth-death processes,
plus forward simulators for general (multi-epoch, time-varying-rate) processes.

Two independent methods for sampling a birth-death tree conditioned on
survival to a terminal time ``T``:

- ``simulate_birth_death_process_from_contour_process`` — exact, via
  contour-process reconstruction (see ``docs/contour_process.md``).
- ``simulate_subcritical_survival_splitting`` — approximate, via adaptive-grid
  (RESTART) multilevel splitting (see ``docs/splitting.md``).

Both return a ``SimulatedTree``: a plain dataclass wrapping the individual-
level records DataFrame plus the rates and terminal time used to generate it.

Unconditioned forward simulators return a ``SimulatedProcess``:

- ``simulate_birth_death_process`` — numpy Gillespie, piecewise-constant rates.
- ``simulate_birth_death_lifespans`` — JIT-compiled lifespan-at-birth engine.
- ``simulate_birth_death_linear`` — exact, piecewise-linear rates.
- ``sample_conditioned_survival_process`` — era-conditioned multilevel splitting.
"""

import os

# JAX raises a fatal error when jax[cuda] is installed but the GPU is
# unavailable, so default to CPU.  Set JAX_PLATFORMS before importing jax.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

# The simulators work in float64 throughout.
jax.config.update("jax_enable_x64", True)

from bdsurv.contour import (  # noqa: E402
    simulate_birth_death_process_from_contour_process,
)
from bdsurv.era_conditioning import (  # noqa: E402
    ConditionedSurvivalResult,
    EraConditioningDiagnostic,
    MatchedSurvivalSample,
    build_target_ranges,
    sample_conditioned_survival_per_posterior,
    sample_conditioned_survival_process,
)
from bdsurv.gillespie import simulate_birth_death_process  # noqa: E402
from bdsurv.lifespans import (  # noqa: E402
    LifespanSimulationArrays,
    simulate_birth_death_lifespans,
    simulate_birth_death_lifespans_arrays,
)
from bdsurv.linear_rates import (  # noqa: E402
    ConstantRateLike,
    LinearRateLike,
    PiecewiseConstant,
    PiecewiseLinear,
    simulate_birth_death_linear,
)
from bdsurv.splitting import (  # noqa: E402
    ConditionedTrajectory,
    LevelDiagnostic,
    SplittingConfig,
    SplittingSimulationResult,
    compute_level_replicates,
    default_max_restarts,
    select_level_times_for_band,
    simulate_subcritical_survival_splitting,
)
from bdsurv.tree import SimulatedProcess, SimulatedTree  # noqa: E402

__version__ = "0.3.0"
__author__ = "Jacob Cumming"

__all__ = [
    "ConditionedSurvivalResult",
    "ConditionedTrajectory",
    "ConstantRateLike",
    "EraConditioningDiagnostic",
    "LevelDiagnostic",
    "LifespanSimulationArrays",
    "LinearRateLike",
    "MatchedSurvivalSample",
    "PiecewiseConstant",
    "PiecewiseLinear",
    "SimulatedProcess",
    "SimulatedTree",
    "SplittingConfig",
    "SplittingSimulationResult",
    "build_target_ranges",
    "compute_level_replicates",
    "default_max_restarts",
    "sample_conditioned_survival_per_posterior",
    "sample_conditioned_survival_process",
    "select_level_times_for_band",
    "simulate_birth_death_lifespans",
    "simulate_birth_death_lifespans_arrays",
    "simulate_birth_death_linear",
    "simulate_birth_death_process",
    "simulate_birth_death_process_from_contour_process",
    "simulate_subcritical_survival_splitting",
]
