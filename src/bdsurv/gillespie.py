"""Forward Gillespie simulation of birth-death processes with epoch-wise rates."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bdsurv._seeding import root_rng, spawn
from bdsurv.rates import epoch_rates
from bdsurv.tree import SimulatedProcess, build_records


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

    # Root RNG from which every attempt's generator is spawned
    base_seed_rng = root_rng(random_seed, "seeding")

    if ensure_non_extinct and max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")

    attempts_allowed = max_retries if ensure_non_extinct else 0
    # We allow (attempts_allowed + 1) attempts: initial + retries
    for _attempt in range(attempts_allowed + 1):
        # Create a per-attempt RNG so retries produce different draws but are
        # reproducible when `random_seed` is provided.
        rng = spawn(base_seed_rng, "dynamics")

        # Initialize tracking variables for this attempt
        current_time = 0.0
        individuals_alive: list[tuple[float, int]] = [
            (0.0, 0),
        ]  # (creation_time, individual_id)
        # Per-individual columns, indexed by individual id
        creation: list[float] = [0.0]
        parent: list[int | None] = [None]
        death_time: list[float | None] = [None]
        new_individual_id = 1

        # Initialize rate tracking
        period_idx = 0
        birth_rate = float(birth_rates_arr[period_idx])
        death_rate = float(death_rates_arr[period_idx])
        max_time = float(times_arr[-1])

        birth_death_rate = birth_rate + death_rate
        prop_birth = birth_rate / birth_death_rate
        n_alive = len(individuals_alive)
        total_rate = n_alive * birth_death_rate

        while current_time < max_time:
            if total_rate <= 0:
                break

            # Time to next event
            dt = rng.exponential(1.0 / total_rate)
            current_time += dt

            # Check if we've crossed into a new period and update rates
            if period_idx < len(times_arr) - 1 and current_time > times_arr[period_idx]:
                # Set current time to the period boundary
                current_time = times_arr[period_idx]
                # Update to next period's parameters
                period_idx += 1
                birth_rate = float(birth_rates_arr[period_idx])
                death_rate = float(death_rates_arr[period_idx])
                birth_death_rate = birth_rate + death_rate
                prop_birth = birth_rate / birth_death_rate
                total_rate = len(individuals_alive) * birth_death_rate
                # Continue to start of loop to generate dt with new rates
                continue

            if current_time >= max_time:
                break

            # Determine if event is birth or death
            if rng.random() < prop_birth:
                # Birth event
                new_entry = (current_time, new_individual_id)
                # Insert at a random position other than the first.
                if len(individuals_alive) == 1:
                    individuals_alive.append(new_entry)
                    parent_id = individuals_alive[0][1]
                else:
                    insert_idx = int(rng.integers(1, len(individuals_alive) + 1))
                    individuals_alive.insert(insert_idx, new_entry)
                    parent_id = individuals_alive[insert_idx - 1][1]

                creation.append(current_time)
                parent.append(parent_id)
                death_time.append(None)
                new_individual_id += 1
                total_rate += birth_death_rate
            else:
                # Death event
                death_idx = rng.integers(0, len(individuals_alive))
                _, dead_id = individuals_alive.pop(death_idx)

                # Update the record
                death_time[dead_id] = current_time

                total_rate -= birth_death_rate

        # Spine = first individual alive at the end
        spine_id = individuals_alive[0][1] if individuals_alive else None
        df = build_records(creation, parent, death_time, spine_id, max_time)
        survived = df["survived"]

        # Return the first draw, or the first non-extinct one if required.
        if not ensure_non_extinct or survived.any():
            return SimulatedProcess(
                records=df,
                birth_rates=birth_rates_arr,
                death_rates=death_rates_arr,
                epoch_end_times=times_arr,
            )

        # Otherwise, loop to retry until attempts exhausted.

    # If we reach here, retries were exhausted without producing a non-extinct draw.
    msg = (
        f"simulate_birth_death_process: exhausted {attempts_allowed} retries "
        "without producing a non-extinct population"
    )
    raise RuntimeError(msg)
