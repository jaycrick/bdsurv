"""bdsurv: simulation of birth-death processes, focused on survival-conditioned trees.

Every simulator returns a :class:`SimulatedTree`: the individual-level records
DataFrame plus the rates and the terminal time ``T`` used to generate it.
The rates of a tree are always rate objects; a float rate is stored as a
one-epoch :class:`PiecewiseConstant`.

Rates are given as a :data:`RateInput`: a float (constant on ``[0, T]``), or a
rate-like that is piecewise constant (:class:`PiecewiseConstant`,
:class:`ConstantRateLike`) or piecewise linear (:class:`PiecewiseLinear`,
:class:`LinearRateLike`) in time over ``[0, T]``.
Rates are finite and non-negative, and ``terminal_time`` is required when both
rates are floats.

Forward simulation, not conditioned on survival:

- ``simulate_birth_death`` - exact event simulation for any ``RateInput`` rates.
- ``simulate_birth_death_lifespans`` - JIT-compiled lifespan-at-birth engine for
  constant rates per epoch, given as arrays; ``..._arrays`` returns the raw
  per-individual arrays.

Trees conditioned on survival to a terminal time ``T``, by two independent
methods:

- ``simulate_contour_given_survival`` - exact, via contour-process
  reconstruction (see ``docs/contour_process.md``).
- ``simulate_survival_splitting`` - approximate, via adaptive-grid (RESTART)
  multilevel splitting configured by ``SplittingConfig`` (see
  ``docs/splitting.md``), with the level helpers
  ``select_level_times_for_band``, ``compute_level_replicates`` and
  ``default_max_restarts``.

Exact contour simulators conditioned on the population alive at ``T``
(see ``docs/contour_conditioned.md``):

- ``simulate_contour_given_population`` - exactly ``n`` individuals alive at ``T``.
- ``simulate_contour_given_ages`` - survivors at ``T`` with the given ages.

Era-conditioned sampling for multi-epoch processes (see
``docs/era_conditioning.md``): ``sample_conditioned_survival_process`` and
``sample_conditioned_survival_per_posterior``, with ``build_target_ranges``.

``log_survival_probability`` is the probability that a process started from one
individual is not extinct at a given time.
``log_mean_cumulative_individuals`` is the log of the expected number of
individuals ever born by a given time, the root included.
``count_draws`` counts the random draws a block of code makes.

The methods rest on published work; see ``docs/references.md``.

Side effects of ``import bdsurv``.
Importing the package sets the environment variable ``JAX_PLATFORMS`` to
``cpu`` if it is unset, and enables JAX 64-bit mode (``jax_enable_x64``) for the
whole process, not only for ``bdsurv``.
It does so because the simulators work in float64, and because a GPU build of
JAX on a machine without a GPU fails at import unless the platform is set to
CPU first.
Set ``JAX_PLATFORMS`` before importing ``bdsurv`` to use another platform.
"""

import importlib.metadata
import os

# JAX raises a fatal error when jax[cuda] is installed but the GPU is
# unavailable, so default to CPU.  Set JAX_PLATFORMS before importing jax.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

# The simulators work in float64 throughout.
jax.config.update("jax_enable_x64", True)

from bdsurv._seeding import DrawCounts, count_draws  # noqa: E402
from bdsurv.contour import simulate_contour_given_survival  # noqa: E402
from bdsurv.contour_conditioned import (  # noqa: E402
    simulate_contour_given_ages,
    simulate_contour_given_population,
)
from bdsurv.era_conditioning import (  # noqa: E402
    ConditionedSurvivalResult,
    EraConditioningDiagnostic,
    MatchedSurvivalSample,
    build_target_ranges,
    sample_conditioned_survival_per_posterior,
    sample_conditioned_survival_process,
)
from bdsurv.forward import simulate_birth_death  # noqa: E402
from bdsurv.lifespans import (  # noqa: E402
    LifespanSimulationArrays,
    simulate_birth_death_lifespans,
    simulate_birth_death_lifespans_arrays,
)
from bdsurv.rates import (  # noqa: E402
    ConstantRateLike,
    LinearRateLike,
    PiecewiseConstant,
    PiecewiseLinear,
    RateInput,
)
from bdsurv.splitting import (  # noqa: E402
    ConditionedTrajectory,
    LevelDiagnostic,
    SplittingConfig,
    SplittingSimulationResult,
    compute_level_replicates,
    default_max_restarts,
    select_level_times_for_band,
    simulate_survival_splitting,
)
from bdsurv.survival import (  # noqa: E402
    log_mean_cumulative_individuals,
    log_survival_probability,
)
from bdsurv.tree import SimulatedTree  # noqa: E402

__version__ = importlib.metadata.version("bdsurv")
__author__ = "Jacob Cumming"

__all__ = [
    "ConditionedSurvivalResult",
    "ConditionedTrajectory",
    "ConstantRateLike",
    "DrawCounts",
    "EraConditioningDiagnostic",
    "LevelDiagnostic",
    "LifespanSimulationArrays",
    "LinearRateLike",
    "MatchedSurvivalSample",
    "PiecewiseConstant",
    "PiecewiseLinear",
    "RateInput",
    "SimulatedTree",
    "SplittingConfig",
    "SplittingSimulationResult",
    "build_target_ranges",
    "compute_level_replicates",
    "count_draws",
    "default_max_restarts",
    "log_mean_cumulative_individuals",
    "log_survival_probability",
    "sample_conditioned_survival_per_posterior",
    "sample_conditioned_survival_process",
    "select_level_times_for_band",
    "simulate_birth_death",
    "simulate_birth_death_lifespans",
    "simulate_birth_death_lifespans_arrays",
    "simulate_contour_given_ages",
    "simulate_contour_given_population",
    "simulate_contour_given_survival",
    "simulate_survival_splitting",
]
