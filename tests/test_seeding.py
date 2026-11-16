"""Test the seeding module: spawned generators, JAX keys and draw counting."""

import jax
import numpy as np
import pytest

from bdsurv import (
    PiecewiseConstant,
    PiecewiseLinear,
    SplittingConfig,
    count_draws,
    simulate_contour_given_survival,
    simulate_survival_splitting,
)
from bdsurv._seeding import (
    CountingGenerator,
    DrawCounts,
    bump,
    child_seed,
    jax_key,
    root_rng,
    spawn,
)

BIRTH = PiecewiseConstant(np.array([1.0]), np.array([6.0]))
DEATH = PiecewiseLinear(np.array([0.0]), np.array([3.0]), np.array([6.0]))


class TestCountingGenerator:
    def test_counts_by_role_and_kind_and_forwards_values(self) -> None:
        counts = DrawCounts()
        counted = CountingGenerator(np.random.default_rng(5), counts, "dynamics")
        plain = np.random.default_rng(5)
        assert counted.exponential(2.0) == plain.exponential(2.0)
        assert counted.uniform() == plain.uniform()
        assert counted.random() == plain.random()
        assert counted.integers(0, 10) == plain.integers(0, 10)
        assert np.array_equal(counted.exponential(1.0, 4), plain.exponential(1.0, 4))
        assert np.array_equal(
            counted.choice(np.arange(9), size=3, replace=False),
            plain.choice(np.arange(9), size=3, replace=False),
        )
        assert counted.normal() == plain.normal()  # uncounted passthrough
        assert counts.draws[("dynamics", "exponential")] == 5
        assert counts.draws[("dynamics", "uniform")] == 2
        assert counts.draws[("dynamics", "integers")] == 1
        assert counts.draws[("dynamics", "choice")] == 3
        assert counts.total_draws == 11
        assert counts.by_role("dynamics") == 11
        assert counts.by_kind("uniform") == 2

    def test_shaped_size(self) -> None:
        counts = DrawCounts()
        counted = CountingGenerator(np.random.default_rng(0), counts, "contour")
        counted.exponential(1.0, (2, 3))
        assert counts.total_draws == 6


class TestContext:
    def test_inactive_by_default(self) -> None:
        assert not isinstance(root_rng(1, "contour"), CountingGenerator)
        bump("events")  # no-op, must not raise

    def test_nesting_and_exception_safety(self) -> None:
        with count_draws() as outer:
            root_rng(1, "contour").exponential()
            with count_draws() as inner:
                root_rng(1, "contour").exponential()
                bump("events", 3)
            assert inner.total_draws == 1
            assert inner.extras["events"] == 3
            root_rng(1, "contour").exponential()
        assert outer.total_draws == 2
        with pytest.raises(RuntimeError), count_draws():
            raise RuntimeError
        assert not isinstance(root_rng(1, "contour"), CountingGenerator)

    def test_as_row_has_every_field(self) -> None:
        row = DrawCounts().as_row()
        assert row["draws_total"] == 0
        assert {"draws_contour", "draws_exponential", "spine_resets", "events"} <= set(
            row
        )


class TestCountedRunsAreUnchanged:
    def test_constant_contour(self) -> None:
        plain = simulate_contour_given_survival(0.9, 1.0, 15.0, random_seed=3)
        with count_draws() as counts:
            counted = simulate_contour_given_survival(0.9, 1.0, 15.0, random_seed=3)
        assert plain.records.equals(counted.records)
        assert counts.by_role("seeding") == 0
        assert counts.by_role("contour") > 0
        assert counts.extras["contour_steps"] > 0

    def test_linear_contour_counts_resets(self) -> None:
        plain = simulate_contour_given_survival(BIRTH, DEATH, random_seed=4)
        with count_draws() as counts:
            counted = simulate_contour_given_survival(BIRTH, DEATH, random_seed=4)
        assert plain.records.equals(counted.records)
        # Each step draws a down-jump exponential, and most also an up-jump.
        assert counts.by_kind("exponential") >= counts.extras["contour_steps"]
        assert counts.extras["spine_resets"] >= 0

    def test_constant_splitting(self) -> None:
        config = SplittingConfig(
            birth_rate=1.0,
            death_rate=2.0,
            terminal_time=8.0,
            n_particles=100,
            random_seed=1,
            max_restarts=500,
        )
        plain = simulate_survival_splitting(config)
        with count_draws() as counts:
            counted = simulate_survival_splitting(config)
        assert plain.final_survivor_count == counted.final_survivor_count
        assert plain.n_attempts == counted.n_attempts == counts.extras["attempts"]
        assert plain.conditioned_trajectories[0].process.records.equals(
            counted.conditioned_trajectories[0].process.records
        )
        assert counts.by_role("dynamics") > 0
        assert counts.by_role("seeding") > 0
        assert counts.extras["events"] > 0
        assert counts.extras["clones"] > 0
        assert counts.extras["records_copied"] >= counts.extras["clones"]
        assert counts.extras["particle_steps"] > 0

    def test_linear_splitting(self) -> None:
        config = SplittingConfig(
            birth_rate=BIRTH,
            death_rate=DEATH,
            n_particles=100,
            random_seed=2,
            max_restarts=500,
        )
        plain = simulate_survival_splitting(config)
        with count_draws() as counts:
            counted = simulate_survival_splitting(config)
        assert plain.n_attempts == counted.n_attempts
        assert counts.total_draws > 0
        # one exponential per event hazard draw, one uniform per event type
        assert counts.draws[("dynamics", "uniform")] == counts.extras["events"]


class TestSpawn:
    def test_children_differ_from_each_other_and_the_parent(self) -> None:
        parent = root_rng(7, "seeding")
        first = spawn(parent, "dynamics")
        second = spawn(parent, "dynamics")
        draws = [g.random(4).tolist() for g in (parent, first, second)]
        assert len({tuple(d) for d in draws}) == 3

    def test_same_seed_same_children(self) -> None:
        def children() -> list[list[float]]:
            parent = root_rng(7, "seeding")
            return [spawn(parent, "dynamics").random(3).tolist() for _ in range(3)]

        assert children() == children()

    def test_spawn_does_not_advance_the_parent(self) -> None:
        spawned_from = root_rng(3, "seeding")
        spawn(spawned_from, "dynamics")
        assert spawned_from.random() == root_rng(3, "seeding").random()

    def test_unseeded_roots_differ(self) -> None:
        assert root_rng(None, "seeding").random() != root_rng(None, "seeding").random()

    def test_child_seed_is_a_nonnegative_int64(self) -> None:
        rng = root_rng(1, "seeding")
        seeds = [child_seed(rng) for _ in range(50)]
        assert len(set(seeds)) == 50
        assert all(0 <= x <= np.iinfo(np.int64).max for x in seeds)

    def test_jax_key_is_accepted_by_jax_random(self) -> None:
        key = jax_key(root_rng(5, "seeding"))
        draw = jax.random.uniform(key)
        assert 0.0 <= float(draw) < 1.0
        again = jax.random.uniform(jax_key(root_rng(5, "seeding")))
        assert float(draw) == float(again)
        other = jax.random.uniform(jax_key(root_rng(6, "seeding")))
        assert float(draw) != float(other)

    def test_counted_generators_spawn_and_add_no_draws(self) -> None:
        with count_draws() as counts:
            parent = root_rng(2, "seeding")
            assert isinstance(parent, CountingGenerator)
            child = spawn(parent, "dynamics")
            assert isinstance(child, CountingGenerator)
            jax_key(parent)
            child_seed(child)
            assert counts.total_draws == 0
            child.exponential()
        assert counts.draws[("dynamics", "exponential")] == 1
        assert counts.total_draws == 1

    def test_counted_children_match_uncounted_children(self) -> None:
        plain = spawn(root_rng(4, "seeding"), "dynamics").random(3)
        with count_draws():
            counted = spawn(root_rng(4, "seeding"), "dynamics").random(3)
        assert np.array_equal(plain, counted)
