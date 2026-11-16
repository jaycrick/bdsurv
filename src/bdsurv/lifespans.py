"""JIT-compiled lifespan-at-birth simulation of birth-death processes.

Each individual's entire lifespan is decided at birth from a
piecewise-exponential distribution, so the numeric core runs inside
``jax.lax.while_loop``.  Parent-ID genealogy and spine assignment are
reconstructed on the host from the stored per-individual birth/death times.

Entry points:

- :func:`simulate_birth_death_lifespans_arrays` — raw per-individual arrays.
- :func:`simulate_birth_death_lifespans` — full :class:`~bdsurv.tree.SimulatedProcess`
  with genealogy (same API as :func:`bdsurv.gillespie.simulate_birth_death_process`).
"""

# JAX's jit/while_loop wrappers are untyped, which strict mode reports as
# unknown.
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import lax
from numpy.typing import NDArray

from bdsurv._forward import retry_until_survival
from bdsurv._seeding import jax_key, root_rng
from bdsurv.rates import epoch_rates
from bdsurv.tree import SimulatedProcess, build_records

# An event within this of an epoch end counts as the epoch end.
TIE_TOLERANCE: float = 1e-12


class LifespanSimulationArrays(NamedTuple):
    """Raw per-individual arrays from the lifespan-at-birth simulator.

    Attributes
    ----------
    birth_times : jax.Array
        Float64 array of shape ``(n_born,)`` — creation time of each individual.
    death_times : jax.Array
        Float64 array of shape ``(n_born,)`` — pre-sampled death time (``inf``
        when the individual survives to the terminal time).
    birth_epochs : jax.Array
        Int32 array of shape ``(n_born,)`` — epoch index in which each
        individual was born.
    survived : jax.Array
        Bool array of shape ``(n_born,)`` — ``True`` iff the individual is
        alive at the terminal time.
    n_born : int
        Total number of individuals born (including the root).

    Notes
    -----
    Number from epoch ``e`` surviving to terminal time::

        jnp.sum((birth_epochs == e) & survived)

    This is the quantity required by the multilevel-splitting estimator.

    """

    birth_times: jax.Array
    death_times: jax.Array
    birth_epochs: jax.Array
    survived: jax.Array
    n_born: int


class LoopState(NamedTuple):
    """State carried through the event loop of :func:`advance_era`.

    Only the first ``n_born`` entries of each buffer are meaningful.
    """

    key: jax.Array  # PRNG key
    current_time: jax.Array  # float64 scalar
    n_alive: jax.Array  # int32 scalar: individuals alive at current_time
    n_born: jax.Array  # int32 scalar: total ever born (= next free buffer slot)
    overflow: jax.Array  # bool scalar: True iff max_population was exceeded
    birth_time_buf: jax.Array  # float64 [max_population]
    death_time_buf: jax.Array  # float64 [max_population] (inf = survives)
    insert_unif_buf: jax.Array  # float64 [max_population] (U[0,1) for genealogy)
    birth_epoch_buf: jax.Array  # int32 [max_population]
    n_alive_at_birth_buf: jax.Array  # int32 [max_population]: n_alive before birth


def sample_piecewise_exponential_death_time(
    birth_time: jax.Array,
    death_rates: jax.Array,
    epoch_end_times: jax.Array,
    terminal_time: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """Sample one individual's death time under piecewise-constant hazard.

    Draws ``E ~ Exp(1)`` and walks the epoch boundaries starting from
    ``birth_time``, accumulating ``rate_e * duration_e`` per epoch until the
    total hazard reaches ``E`` (inversion method).  Returns ``jnp.inf`` when
    the total hazard over ``[birth_time, terminal_time]`` is less than ``E``
    (i.e. the individual survives to the terminal time).

    JIT-safe: implemented as a ``lax.scan`` over the static epoch count.

    """
    key, subkey = jax.random.split(key)
    # Draw the target cumulative hazard (exponential quantile)
    E = jax.random.exponential(subkey, dtype=jnp.float64)

    # Epoch start times: 0 for epoch 0, epoch_end_times[k-1] for epoch k
    epoch_starts = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.float64), epoch_end_times[:-1]]
    )

    # Per-epoch scan: accumulate hazard; solve for death within the first
    # epoch that pushes accumulated hazard past E.
    def _scan_epoch(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        xs: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        accumulated, found, death_time = carry
        epoch_start, epoch_end, rate = xs[0], xs[1], xs[2]

        # Portion of this epoch after birth and before terminal time
        interval_start = jnp.maximum(birth_time, epoch_start)
        interval_end = jnp.minimum(epoch_end, terminal_time)
        duration = jnp.maximum(jnp.float64(0.0), interval_end - interval_start)

        hazard_this_epoch = rate * duration
        new_accumulated = accumulated + hazard_this_epoch

        # Solve for the death time: start of interval plus the remaining hazard / rate
        remaining = E - accumulated
        safe_rate = jnp.where(rate > 0.0, rate, jnp.float64(1.0))
        candidate_death = interval_start + remaining / safe_rate

        # Death falls in this epoch iff hazard here tips us over E (and not
        # already found in an earlier epoch, and there is a nonzero interval).
        in_epoch = (~found) & (new_accumulated >= E) & (duration > 0.0) & (rate > 0.0)
        new_found = found | in_epoch
        new_death_time = jnp.where(in_epoch, candidate_death, death_time)

        return (new_accumulated, new_found, new_death_time), None

    xs = jnp.stack([epoch_starts, epoch_end_times, death_rates], axis=1)  # [n, 3]
    (_, found, death_time), _ = lax.scan(
        _scan_epoch,
        (jnp.float64(0.0), jnp.bool_(False), jnp.float64(jnp.inf)),
        xs,
    )
    # Survives to terminal time → return inf sentinel
    return jnp.where(found, death_time, jnp.inf)


@functools.partial(jax.jit, static_argnums=(3,))
def initial_state(
    death_rates: jax.Array,
    epoch_end_times: jax.Array,
    key: jax.Array,
    max_population: int,
) -> LoopState:
    """Return the state with one root individual born at time 0.

    The root's death time is sampled over the full hazard
    (``death_rates`` on ``epoch_end_times``; the last entry is the terminal time).
    Jitted with ``max_population`` static, like :func:`advance_era`.
    """
    key, k_root_death = jax.random.split(key)
    root_death_time = sample_piecewise_exponential_death_time(
        birth_time=jnp.float64(0.0),
        death_rates=death_rates,
        epoch_end_times=epoch_end_times,
        terminal_time=epoch_end_times[-1],
        key=k_root_death,
    )
    return LoopState(
        key=key,
        current_time=jnp.float64(0.0),
        n_alive=jnp.int32(1),
        n_born=jnp.int32(1),
        overflow=jnp.bool_(False),
        birth_time_buf=jnp.full(max_population, jnp.inf, dtype=jnp.float64)
        .at[0]
        .set(jnp.float64(0.0)),
        death_time_buf=jnp.full(max_population, jnp.inf, dtype=jnp.float64)
        .at[0]
        .set(root_death_time),
        insert_unif_buf=jnp.zeros(max_population, dtype=jnp.float64),
        birth_epoch_buf=jnp.zeros(max_population, dtype=jnp.int32),
        n_alive_at_birth_buf=jnp.zeros(max_population, dtype=jnp.int32),
    )


@functools.partial(jax.jit, static_argnums=(6,))
def advance_era(  # noqa: PLR0913, PLR0917  # jitted kernel signature fixed by the plan
    birth_rate_k: jax.Array,
    death_rates: jax.Array,
    epoch_end_times: jax.Array,
    era_idx: jax.Array,
    segment_end: jax.Array,
    state: LoopState,
    max_population: int,
) -> LoopState:
    """Advance the event loop from ``state.current_time`` to ``segment_end``.

    This is the one event kernel of the package.  Births occur at rate
    ``n_alive * birth_rate_k``, which is fixed for the era.  Each new
    individual's death time is sampled at birth over the **full** hazard
    (``death_rates`` on ``epoch_end_times``), so survival to the terminal time
    ``epoch_end_times[-1]`` is known from the start.  An event within
    ``TIE_TOLERANCE`` of ``segment_end`` counts as the era boundary.

    Parameters
    ----------
    birth_rate_k
        Per-individual birth rate for this era (float64 scalar).
    death_rates
        Full per-epoch death rates, shape ``[n_epochs]``.
    epoch_end_times
        Full epoch end times, shape ``[n_epochs]``; the last is the terminal time.
    era_idx
        Era index (int32 scalar), written to ``birth_epoch_buf`` for every
        individual born in this call.
    segment_end
        End time of the era (float64 scalar).
    state
        Carry-in state, from :func:`initial_state` or a previous call.
    max_population
        **Static** buffer capacity; a new value recompiles the kernel.

    Returns
    -------
    LoopState
        State at ``current_time == segment_end``, unless ``overflow`` is set
        because more than ``max_population`` individuals would have been born.

    """
    terminal_time: jax.Array = epoch_end_times[-1]

    def _cond(s: LoopState) -> jax.Array:
        return (s.current_time < segment_end) & ~s.overflow

    def _body(s: LoopState) -> LoopState:
        key, k_birth_dt, k_death_time, k_insert = jax.random.split(s.key, 4)

        # ---- Next birth time ----
        total_birth_rate = s.n_alive.astype(jnp.float64) * birth_rate_k
        dt_birth = jax.random.exponential(k_birth_dt, dtype=jnp.float64) / jnp.where(
            total_birth_rate > 0.0, total_birth_rate, jnp.float64(1.0)
        )
        next_birth = jnp.where(
            total_birth_rate > 0.0, s.current_time + dt_birth, jnp.inf
        )

        # ---- Next scheduled death (smallest future death time in buffer) ----
        indices = jnp.arange(max_population)
        future_death_mask = (
            (indices < s.n_born)
            & jnp.isfinite(s.death_time_buf)
            & (s.death_time_buf > s.current_time)
        )
        next_death = jnp.min(jnp.where(future_death_mask, s.death_time_buf, jnp.inf))

        # ---- Choose next event (the era boundary wins ties) ----
        next_event = jnp.minimum(jnp.minimum(next_birth, next_death), segment_end)
        is_boundary = next_event >= segment_end - jnp.float64(TIE_TOLERANCE)
        is_death = (~is_boundary) & (next_death <= next_birth)
        is_birth = (~is_boundary) & (~is_death)
        new_time = jnp.where(next_event >= segment_end, segment_end, next_event)

        # ---- Birth event ----
        can_do_birth = is_birth & (s.n_born < max_population)
        new_overflow = s.overflow | (is_birth & (s.n_born >= max_population))

        new_indiv_death = sample_piecewise_exponential_death_time(
            birth_time=next_event,
            death_rates=death_rates,
            epoch_end_times=epoch_end_times,
            terminal_time=terminal_time,
            key=k_death_time,
        )
        new_insert_u = jax.random.uniform(k_insert, dtype=jnp.float64)

        # Single-index writes at the clamped slot; without a birth the slot
        # keeps its old value.
        i = jnp.minimum(s.n_born, max_population - 1)

        def write(buf: jax.Array, value: jax.Array) -> jax.Array:
            return buf.at[i].set(jnp.where(can_do_birth, value, buf[i]))

        new_n_alive = jnp.where(
            can_do_birth,
            s.n_alive + 1,
            jnp.where(is_death, jnp.maximum(s.n_alive - 1, 0), s.n_alive),
        )
        return LoopState(
            key=key,
            current_time=new_time,
            n_alive=new_n_alive,
            n_born=jnp.where(can_do_birth, s.n_born + 1, s.n_born),
            overflow=new_overflow,
            birth_time_buf=write(s.birth_time_buf, next_event),
            death_time_buf=write(s.death_time_buf, new_indiv_death),
            insert_unif_buf=write(s.insert_unif_buf, new_insert_u),
            birth_epoch_buf=write(s.birth_epoch_buf, era_idx),
            n_alive_at_birth_buf=write(s.n_alive_at_birth_buf, s.n_alive),
        )

    return lax.while_loop(_cond, _body, state)


def _reconstruct_genealogy(
    birth_times: np.ndarray,
    death_times: np.ndarray,
    insert_uniforms: np.ndarray,
    birth_epochs: np.ndarray,
    n_alive_at_birth: np.ndarray,
    terminal_time: float,
) -> pd.DataFrame:
    """Replay the event stream to assign parent IDs and spine.

    Replicates the random-insert genealogy convention of
    :func:`bdsurv.gillespie.simulate_birth_death_process`: new individuals are
    inserted at position
    ``1 + floor(u * n_alive)`` (clamped to ``n_alive``) in the ordered
    alive-list, with their parent being the individual at ``insert_pos - 1``.
    The spine is ``alive_list[0]`` at termination.

    Parameters
    ----------
    birth_times : np.ndarray, shape (n_born,)
    death_times : np.ndarray, shape (n_born,)
        ``inf`` for survivors.
    insert_uniforms : np.ndarray, shape (n_born,)
        Pre-sampled U[0,1) values for each individual's insertion position.
    birth_epochs : np.ndarray, shape (n_born,)
        Epoch index in which each individual was born.
    n_alive_at_birth : np.ndarray, shape (n_born,)
        Population size just before each individual's birth (from JIT state).
    terminal_time : float

    Returns
    -------
    pd.DataFrame
        Canonical eight-column records table (individual_id, parent_id,
        creation_time, death_time, is_spine, survived, age, lifespan) plus
        ``birth_epoch``.

    """
    n_born = len(birth_times)

    # ---- Build chronological event list ----
    # Each element: (time, kind, individual_id)
    # kind: 0 = death, 1 = birth  (deaths processed before births at same time)
    events: list[tuple[float, int, int]] = []
    for i in range(n_born):
        events.append((float(birth_times[i]), 1, i))
        if np.isfinite(death_times[i]) and death_times[i] <= terminal_time:
            events.append((float(death_times[i]), 0, i))
    events.sort(key=lambda e: (e[0], e[1]))

    # ---- Replay ----
    alive_list: list[int] = []  # ordered list of alive individual IDs
    parent_map: dict[int, int | None] = {}

    for t, kind, i in events:
        if kind == 1:  # birth
            n = int(n_alive_at_birth[i])
            if n == 0:
                # Root individual: no parent, simply append
                parent_map[i] = None
                alive_list.append(i)
            elif len(alive_list) == 1:
                parent_map[i] = alive_list[0]
                alive_list.append(i)
            else:
                # Insert at 1 + floor(u * n), clamped to n
                insert_pos = min(1 + int(float(insert_uniforms[i]) * n), n)
                parent_map[i] = alive_list[insert_pos - 1]
                alive_list.insert(insert_pos, i)
        else:  # death
            try:
                alive_list.remove(i)
            except ValueError as exc:
                msg = (
                    f"inconsistent event stream: individual {i} dies at "
                    f"t={t} but is not alive"
                )
                raise ValueError(msg) from exc

    # ---- Spine = alive_list[0] at termination ----
    spine_id = alive_list[0] if alive_list else None

    # ---- Build records DataFrame ----
    died = np.isfinite(death_times) & (death_times <= terminal_time)
    return build_records(
        birth_times,
        [parent_map.get(i) for i in range(n_born)],
        np.where(died, death_times, np.nan),
        spine_id,
        terminal_time,
        birth_epoch=np.asarray(birth_epochs, dtype=np.int64),
    )


def _validated_inputs(
    birth_rates: float | NDArray[np.float64],
    death_rates: float | NDArray[np.float64],
    times: float | NDArray[np.float64],
    max_population: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate the arguments shared by both simulators; return the rate arrays."""
    arrays = epoch_rates(birth_rates, death_rates, times)
    if max_population < 1:
        msg = "max_population must be at least 1"
        raise ValueError(msg)
    return arrays


def _run(
    birth_rates_arr: NDArray[np.float64],
    death_rates_arr: NDArray[np.float64],
    times_arr: NDArray[np.float64],
    key: jax.Array,
    max_population: int,
    *,
    caller: str,
) -> LoopState:
    """Run the event loop through every epoch and return the final state.

    Raises
    ------
    RuntimeError
        If more than ``max_population`` individuals would have been born; the
        message names ``caller``.
    """
    birth_rates_jax = jnp.array(birth_rates_arr, dtype=jnp.float64)
    death_rates_jax = jnp.array(death_rates_arr, dtype=jnp.float64)
    times_jax = jnp.array(times_arr, dtype=jnp.float64)

    state = initial_state(death_rates_jax, times_jax, key, max_population)
    for k in range(len(times_arr)):
        state = advance_era(
            birth_rates_jax[k],
            death_rates_jax,
            times_jax,
            jnp.int32(k),
            times_jax[k],
            state,
            max_population,
        )
    if bool(state.overflow):
        msg = (
            f"{caller}: population exceeded max_population={max_population}. "
            "Increase max_population or use a lower birth rate / shorter time span."
        )
        raise RuntimeError(msg)
    return state


def simulate_birth_death_lifespans_arrays(
    birth_rates: float | NDArray[np.float64],
    death_rates: float | NDArray[np.float64],
    times: float | NDArray[np.float64],
    random_seed: int | None = None,
    *,
    max_population: int = 200_000,
) -> LifespanSimulationArrays:
    """Simulate a birth-death process and return raw per-individual arrays.

    Runs the lifespan-at-birth Gillespie algorithm via a JIT-compiled JAX
    event loop.  Each individual's death time is pre-sampled at birth from a
    piecewise-exponential distribution, so the number of individuals surviving
    from each epoch to the terminal time is directly accessible from the
    returned arrays.

    This is the low-level entry point intended for downstream analysis such
    as multilevel splitting.  For a full :class:`~bdsurv.tree.SimulatedProcess` with
    genealogy use :func:`simulate_birth_death_lifespans` instead.

    Parameters
    ----------
    birth_rates
        Per-individual birth rate(s) λ.  Scalar or 1-D array
        (one value per epoch).
    death_rates
        Per-individual death rate(s) μ.  Same length as
        ``birth_rates``.
    times
        Epoch end time(s).  Scalar or 1-D array strictly increasing.
        The last entry is the terminal time T.
    random_seed
        Integer seed for JAX PRNG reproducibility, or ``None``
        for a nondeterministic draw.
    max_population
        Maximum number of individuals that can be born.
        Raises ``RuntimeError`` on overflow.  Recompiles the JIT kernel
        whenever a new value is first used.

    Returns
    -------
    LifespanSimulationArrays
        Named tuple with fields ``birth_times``, ``death_times``,
        ``birth_epochs``, ``survived``, ``n_born``.  Arrays are JAX arrays
        of length ``n_born``.

    Raises
    ------
    ValueError
        If the arguments have incompatible lengths, a rate is not finite and
        positive, the times are not finite, positive and strictly increasing,
        or ``max_population < 1``.
    RuntimeError
        If the simulated population exceeds ``max_population``.

    """
    birth_rates_arr, death_rates_arr, times_arr = _validated_inputs(
        birth_rates, death_rates, times, max_population
    )
    key = jax_key(root_rng(random_seed, "seeding"))
    final_state = _run(
        birth_rates_arr,
        death_rates_arr,
        times_arr,
        key,
        max_population,
        caller="simulate_birth_death_lifespans_arrays",
    )

    n_born = int(final_state.n_born)
    terminal_time = float(times_arr[-1])
    survived = final_state.death_time_buf[:n_born] > terminal_time

    return LifespanSimulationArrays(
        birth_times=final_state.birth_time_buf[:n_born],
        death_times=final_state.death_time_buf[:n_born],
        birth_epochs=final_state.birth_epoch_buf[:n_born],
        survived=survived,
        n_born=n_born,
    )


def simulate_birth_death_lifespans(  # noqa: PLR0913  # public signature
    birth_rates: float | NDArray[np.float64],
    death_rates: float | NDArray[np.float64],
    times: float | NDArray[np.float64],
    random_seed: int | None = None,
    *,
    ensure_non_extinct: bool = False,
    max_retries: int = 100,
    max_population: int = 200_000,
) -> SimulatedProcess:
    """Simulate a birth-death process using the lifespan-at-birth algorithm.

    A JIT-compiled alternative to
    :func:`bdsurv.gillespie.simulate_birth_death_process` with
    the same public API.  Each individual's entire lifespan is decided at
    birth from a piecewise-exponential distribution.  The numeric core runs
    inside ``jax.lax.while_loop``; parent-ID genealogy and spine assignment
    are reconstructed on the host from the stored per-individual birth/death
    times and insertion uniforms.

    The returned :class:`~bdsurv.tree.SimulatedProcess` has an
    additional ``birth_epoch`` column in its ``records`` table,
    recording the epoch in which each individual was born.

    Parameters
    ----------
    birth_rates
        Birth rate(s) (λ). Scalar or 1-D array (one per epoch).
    death_rates
        Death rate(s) (μ). Same length as ``birth_rates``.
    times
        Epoch end time(s). Scalar or strictly-increasing 1-D array.
    random_seed
        Integer seed for reproducibility, or ``None``.
    ensure_non_extinct
        If ``True``, retry until at least one individual
        survives to the terminal time.
    max_retries
        Maximum number of retry attempts when
        ``ensure_non_extinct=True``.
    max_population
        Maximum total individuals that may be born; passed
        to the JIT core as a static buffer size.

    Returns
    -------
    SimulatedProcess
        Simulated process with full genealogy and spine assignment.

    Raises
    ------
    ValueError
        On invalid inputs (lengths, rates that are not finite and positive,
        times that are not finite, positive and strictly increasing).
    RuntimeError
        If ``ensure_non_extinct=True`` and all retries produce an extinct
        population, or if ``max_population`` is exceeded.

    """
    birth_rates_arr, death_rates_arr, times_arr = _validated_inputs(
        birth_rates, death_rates, times, max_population
    )
    terminal_time = float(times_arr[-1])
    rng = root_rng(random_seed, "seeding")

    def attempt() -> SimulatedProcess | None:
        final_state = _run(
            birth_rates_arr,
            death_rates_arr,
            times_arr,
            jax_key(rng),
            max_population,
            caller="simulate_birth_death_lifespans",
        )
        n_born = int(final_state.n_born)
        death_times = np.asarray(final_state.death_time_buf[:n_born])
        if ensure_non_extinct and not (death_times > terminal_time).any():
            return None
        records = _reconstruct_genealogy(
            birth_times=np.asarray(final_state.birth_time_buf[:n_born]),
            death_times=death_times,
            insert_uniforms=np.asarray(final_state.insert_unif_buf[:n_born]),
            birth_epochs=np.asarray(final_state.birth_epoch_buf[:n_born]),
            n_alive_at_birth=np.asarray(final_state.n_alive_at_birth_buf[:n_born]),
            terminal_time=terminal_time,
        )
        return SimulatedProcess(
            records=records,
            birth_rates=birth_rates_arr,
            death_rates=death_rates_arr,
            epoch_end_times=times_arr,
        )

    return retry_until_survival(
        attempt,
        ensure_non_extinct=ensure_non_extinct,
        max_retries=max_retries,
        name="simulate_birth_death_lifespans",
    )
