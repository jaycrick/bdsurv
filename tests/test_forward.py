"""Tests for the forward event kernel in ``bdsurv._forward``."""

import math

import numpy as np
import pytest
from _helpers import assert_valid_records

from bdsurv._forward import (
    ForwardState,
    Segment,
    advance,
    retry_until_survival,
)
from bdsurv._hazard import next_event_offset

DTYPES = {
    "individual_id": "int64",
    "parent_id": "float64",
    "creation_time": "float64",
    "death_time": "float64",
    "is_spine": "bool",
    "survived": "bool",
    "age": "float64",
    "lifespan": "float64",
}


def constant_segments(birth: float, death: float, end: float) -> list[Segment]:
    """One segment on ``[0, end]`` with constant rates."""
    return [(0.0, end, birth, 0.0, death, 0.0)]


class ScriptedRng:
    """A generator that returns scripted draws, in order, for each kind."""

    def __init__(
        self,
        exponentials: list[float],
        randoms: list[float],
        integers: list[int],
    ) -> None:
        self.exponentials = list(exponentials)
        self.randoms = list(randoms)
        self.integer_draws = list(integers)
        self.integer_bounds: list[tuple[int, int]] = []

    def exponential(self) -> float:
        return self.exponentials.pop(0)

    def random(self) -> float:
        return self.randoms.pop(0)

    def integers(self, low: int, high: int) -> int:
        self.integer_bounds.append((low, high))
        return self.integer_draws.pop(0)


class TestAdvance:
    def test_two_steps_give_a_valid_state(self) -> None:
        rng = np.random.default_rng(3)
        segments = constant_segments(1.0, 0.5, 6.0)
        state = ForwardState.root()
        advance(state, segments, 2.0, rng)
        assert state.current_time == 2.0, "time after the first step"
        advance(state, segments, 6.0, rng)
        assert state.current_time == 6.0, "time after the second step"
        assert_valid_records(state.to_records(6.0), 6.0, expected_dtypes=DTYPES)

    @pytest.mark.parametrize("seed", range(4))
    def test_time_equals_target_also_when_extinct(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        state = ForwardState.root()
        advance(state, constant_segments(0.1, 5.0, 50.0), 50.0, rng)
        assert state.alive_ids == [], "the process should die out"
        assert state.current_time == 50.0, "time must be the target after extinction"

    def test_target_before_current_time_raises(self) -> None:
        state = ForwardState.root()
        advance(state, constant_segments(1.0, 1.0, 5.0), 2.0, np.random.default_rng(0))
        with pytest.raises(ValueError, match="target_time"):
            advance(
                state, constant_segments(1.0, 1.0, 5.0), 1.0, np.random.default_rng(0)
            )

    def test_copy_is_independent(self) -> None:
        rng = np.random.default_rng(1)
        segments = constant_segments(1.0, 0.2, 5.0)
        state = ForwardState.root()
        advance(state, segments, 2.0, rng)
        before = (
            state.current_time,
            state.alive_ids.copy(),
            state.creation_time.copy(),
            state.parent_id.copy(),
            state.death_time.copy(),
        )
        clone = state.copy()
        advance(clone, segments, 5.0, rng)
        after = (
            state.current_time,
            state.alive_ids,
            state.creation_time,
            state.parent_id,
            state.death_time,
        )
        assert after == before, "advancing the copy changed the original"
        assert clone.current_time == 5.0, "the copy should have advanced"

    def test_no_deaths_without_death_rate(self) -> None:
        state = ForwardState.root()
        advance(state, constant_segments(1.0, 0.0, 4.0), 4.0, np.random.default_rng(2))
        assert all(t is None for t in state.death_time), "nobody should die"
        assert len(state.alive_ids) == state.n_individuals, "all should be alive"
        assert state.n_individuals > 1, "births should happen"

    def test_no_births_without_birth_rate(self) -> None:
        state = ForwardState.root()
        for _ in range(3):
            state.alive_ids.append(len(state.creation_time))
            state.creation_time.append(0.0)
            state.parent_id.append(0)
            state.death_time.append(None)
        advance(
            state, constant_segments(0.0, 1.0, 20.0), 20.0, np.random.default_rng(2)
        )
        assert state.n_individuals == 4, "nobody should be born"
        assert state.alive_ids == [], "everybody should have died by time 20"

    def test_scripted_draws_give_the_expected_genealogy(self) -> None:
        rng = ScriptedRng(
            exponentials=[0.5, 0.4, 3.0, 1000.0],
            randoms=[0.1, 0.1, 0.9],
            integers=[1, 1],
        )
        events: list[tuple[int, float]] = []
        state = ForwardState.root()
        advance(
            state,
            constant_segments(1.0, 1.0, 10.0),
            5.0,
            rng,  # pyright: ignore[reportArgumentType]
            on_event=lambda s, t: events.append((len(s.alive_ids), t)),
        )
        # t=0.25 birth (n=1, no position draw); t=0.35 birth at position 1
        # (parent is the left neighbour, id 0); t=0.85 death of position 1.
        assert state.alive_ids == [0, 1], "alive ids"
        assert state.parent_id == [None, 0, 0], "parents"
        assert state.creation_time == pytest.approx([0.0, 0.25, 0.35], abs=1e-12)
        assert state.death_time[:2] == [None, None], "survivors have no death time"
        assert state.death_time[2] == pytest.approx(0.85, abs=1e-12), "death time"
        assert rng.integer_bounds == [(1, 3), (0, 3)], "position draw bounds"
        assert [n for n, _ in events] == [2, 3, 2], "population after each event"
        assert [t for _, t in events] == pytest.approx([0.25, 0.35, 0.85]), "times"
        assert state.current_time == 5.0, "time after the discarded last event"

    def test_event_past_a_segment_end_is_redrawn_with_the_next_rates(self) -> None:
        segments: list[Segment] = [
            (0.0, 1.0, 1.0, 0.0, 1.0, 0.0),
            (1.0, 4.0, 2.0, 0.0, 2.0, 0.0),
        ]
        # First draw: offset 3 / 2 > 1, discarded; redrawn from t=1 with a=4.
        rng = ScriptedRng([3.0, 1.0, 100.0], [0.1], [1])
        state = ForwardState.root()
        advance(state, segments, 1.5, rng)  # pyright: ignore[reportArgumentType]
        assert state.creation_time[1] == pytest.approx(1.25, abs=1e-12), (
            "the event after the boundary should use the second segment's rate"
        )


class TestFastPath:
    @pytest.mark.parametrize("a", [0.3, 1.0, 7.5])
    @pytest.mark.parametrize("e", [0.01, 0.7, 3.0])
    def test_direct_offset_equals_the_general_solution(
        self, a: float, e: float
    ) -> None:
        assert math.isclose(e / a, next_event_offset(a, 0.0, e), abs_tol=1e-12), (
            f"E/a and next_event_offset disagree for a={a}, E={e}"
        )

    def test_zero_rate_segment_has_no_event(self) -> None:
        state = ForwardState.root()
        rng = ScriptedRng([1.0], [], [])
        advance(state, [(0.0, 3.0, 0.0, 0.0, 0.0, 0.0)], 3.0, rng)  # pyright: ignore[reportArgumentType]
        assert state.n_individuals == 1, "no event is possible at rate zero"
        assert state.current_time == 3.0, "time after the segment"


class TestRetryUntilSurvival:
    def test_returns_the_first_non_none_result(self) -> None:
        results = iter([None, None, "ok", "later"])
        out = retry_until_survival(
            lambda: next(results), ensure_non_extinct=True, max_retries=5, name="sim"
        )
        assert out == "ok", "should stop at the first non-None result"

    def test_returns_an_extinct_result_when_not_ensuring(self) -> None:
        calls = []

        def attempt() -> str:
            calls.append(1)
            return "extinct"

        out = retry_until_survival(
            attempt, ensure_non_extinct=False, max_retries=-1, name="sim"
        )
        assert out == "extinct", "the first result is returned as is"
        assert len(calls) == 1, "one attempt only"

    def test_raises_when_every_attempt_fails(self) -> None:
        calls = []

        def attempt() -> None:
            calls.append(1)

        with pytest.raises(RuntimeError, match=r"sim: exhausted 3 retries without"):
            retry_until_survival(
                attempt, ensure_non_extinct=True, max_retries=3, name="sim"
            )
        assert len(calls) == 4, "max_retries + 1 attempts"

    def test_negative_max_retries_raises(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            retry_until_survival(
                lambda: "x", ensure_non_extinct=True, max_retries=-1, name="sim"
            )
