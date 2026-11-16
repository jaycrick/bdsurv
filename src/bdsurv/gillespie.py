"""Forward Gillespie simulation of birth-death processes with epoch-wise rates."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bdsurv._forward import simulate_forward
from bdsurv.rates import PiecewiseConstant, as_rate, epoch_rates
from bdsurv.tree import SimulatedProcess


def simulate_birth_death_process(
    birth_rates: float | NDArray[np.float64],
    death_rates: float | NDArray[np.float64],
    times: float | NDArray[np.float64],
    random_seed: int | None = None,
    *,
    ensure_non_extinct: bool = False,
    max_retries: int = 100,
) -> SimulatedProcess:
    """Simulate individual data from the birth-death process with time-varying rates.

    This creates a proper birth-death process simulation where records
    are born at rate λ and die at rate μ. The rates can change at specified
    time points. The simulation runs until the final time is reached and
    records both creation times and survival status.

    Parameters
    ----------
    birth_rates
        Birth rate(s) (λ) for individual creation. Either a single
        float or an array of rates corresponding to time periods.
    death_rates
        Death rate(s) (μ) for individual loss. Either a single
        float or an array of rates corresponding to time periods.
    times
        End time(s) for the simulation. Either a single float or an array
        of time points at which rates change. Must be the same length as
        birth_rates and death_rates if arrays are provided.
    random_seed
        Random seed for reproducibility
    ensure_non_extinct
        If True, redraw until at least one individual survives to the final
        time, up to ``max_retries`` redraws.
    max_retries
        Maximum number of redraws when ``ensure_non_extinct`` is True.

    Returns
    -------
    SimulatedProcess
        The simulated individual records together with the rates and epoch
        end times used to generate them.

    Raises
    ------
    ValueError
        If the three arguments differ in length, a rate is not finite and
        positive, the times are not finite, positive and strictly increasing,
        or ``max_retries`` is negative.

    """
    birth_rates_arr, death_rates_arr, times_arr = epoch_rates(
        birth_rates, death_rates, times
    )
    records, _ = simulate_forward(
        as_rate(PiecewiseConstant(birth_rates_arr, times_arr)),
        as_rate(PiecewiseConstant(death_rates_arr, times_arr)),
        random_seed,
        ensure_non_extinct=ensure_non_extinct,
        max_retries=max_retries,
        name="simulate_birth_death_process",
    )
    return SimulatedProcess(
        records=records,
        birth_rates=birth_rates_arr,
        death_rates=death_rates_arr,
        epoch_end_times=times_arr,
    )
