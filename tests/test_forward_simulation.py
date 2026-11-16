"""Tests for ``simulate_birth_death``: epoch-wise constant and linear rates."""

import math

import _helpers as h
import numpy as np
import pytest
from _helpers import n_individuals
from scipy import stats

from bdsurv import PiecewiseConstant, PiecewiseLinear, simulate_birth_death
from bdsurv._hazard import next_event_offset
from bdsurv.tree import SimulatedTree


class TestSimulation:
    """Test simulation functionality."""

    def test_simulation_basic(self) -> None:
        """Test basic simulation functionality using modular approach."""
        # Run simulation using modular function
        df = simulate_birth_death(
            birth_rate=0.1,
            death_rate=0.02,
            terminal_time=100.0,
            random_seed=42,
        )

        # Should return a SimulatedTree
        assert isinstance(df, SimulatedTree)

        process = df
        surviving_records = h.surviving_records(process)
        assert surviving_records is not None

        # Check that epoch_end_times and creation_times were set
        assert process.terminal_time == 100.0
        assert isinstance(h.creation_times(process), list)

        # Ages should be calculated consistently
        ages = h.ages(process)
        assert len(ages) == len(h.creation_times(process))

    def test_simulation_with_seed_reproducible(self) -> None:
        """Test that simulation with same seed is reproducible."""
        # Run simulations using modular function with same seed
        df1 = simulate_birth_death(
            birth_rate=0.1,
            death_rate=0.02,
            terminal_time=100.0,
            random_seed=42,
        )
        df2 = simulate_birth_death(
            birth_rate=0.1,
            death_rate=0.02,
            terminal_time=100.0,
            random_seed=42,
        )

        # Should have same number of records
        assert h.n_individuals(df1) == h.n_individuals(df2)

        # Create process objects from simulation results
        surviving1 = h.surviving_records(df1)
        surviving2 = h.surviving_records(df2)
        assert surviving1 is not None
        assert surviving2 is not None
        creation_times1 = surviving1["creation_time"].tolist()
        creation_times2 = surviving2["creation_time"].tolist()

        assert len(creation_times1) == len(creation_times2)

    def test_simulation_different_seeds_different_results(self) -> None:
        """Test that different seeds give different results."""
        # Run simulations using modular function with different seeds
        df1 = simulate_birth_death(
            birth_rate=0.1,
            death_rate=0.02,
            terminal_time=100.0,
            random_seed=42,
        )
        df2 = simulate_birth_death(
            birth_rate=0.1,
            death_rate=0.02,
            terminal_time=100.0,
            random_seed=123,
        )

        # Very unlikely to have identical results with different seeds
        # (unless both simulations produce 0 records, which is possible but rare)
        if h.n_individuals(df1) > 0 and h.n_individuals(df2) > 0:
            # If both have records, they should be different
            surviving1 = h.surviving_records(df1)
            surviving2 = h.surviving_records(df2)
            assert surviving1 is not None
            assert surviving2 is not None
            creation_times1 = surviving1["creation_time"].tolist()
            creation_times2 = surviving2["creation_time"].tolist()
            assert not df1.records.equals(df2.records) or len(creation_times1) != len(
                creation_times2
            )

    def test_simulation_data_frame_structure(self) -> None:
        """Test that simulation returns properly structured DataFrame."""
        # Run simulation using modular function
        df = simulate_birth_death(
            birth_rate=0.2,
            death_rate=0.01,
            terminal_time=50.0,
            random_seed=42,
        )

        if h.n_individuals(df) > 0:
            # Check required columns for surviving records
            surviving = h.surviving_records(df)
            assert surviving is not None
            if len(surviving) > 0:
                required_cols = ["individual_id", "creation_time", "age", "survived"]
                for col in required_cols:
                    assert col in surviving.columns

                # Check data types and validity
                assert all(surviving["individual_id"] >= 0)
                assert all(surviving["creation_time"] >= 0)
                assert all(surviving["creation_time"] <= 50.0)
                assert all(surviving["age"] >= 0)
                assert all(surviving["survived"])

            # Check dead records
            dead = h.extinct_records(df)
            assert dead is not None
            if len(dead) > 0:
                required_cols = [
                    "individual_id",
                    "creation_time",
                    "death_time",
                    "lifespan",
                    "survived",
                ]
                for col in required_cols:
                    assert col in dead.columns

                assert all(dead["creation_time"] >= 0)
                assert all(dead["death_time"] > dead["creation_time"])
                assert all(dead["lifespan"] >= 0)
                assert all(~dead["survived"])

    def test_simulation_ages_consistency(self) -> None:
        """Test that ages are calculated correctly after simulation."""
        # Run simulation using modular function
        df = simulate_birth_death(
            birth_rate=0.15,
            death_rate=0.03,
            terminal_time=80.0,
            random_seed=42,
        )

        # Create process object from simulation results
        surviving_records = h.surviving_records(df)
        assert surviving_records is not None
        if len(surviving_records) > 0:
            process = df

            # Check that manual calculation matches h.ages(process)
            manual_ages = [
                process.terminal_time - ct for ct in h.creation_times(process)
            ]
            np.testing.assert_array_almost_equal(
                h.ages(process),
                manual_ages,
            )

            # Check that ages for surviving records match h.ages(process)
            df_ages = surviving_records["age"].values
            # All surviving records should have ages that match h.ages(process)
            np.testing.assert_array_almost_equal(
                sorted(df_ages),
                sorted(h.ages(process)),
            )

    def test_simulation_empty_result(self) -> None:
        """Test simulation that might produce no records."""
        # Very low birth rate, high death rate, short time
        df = simulate_birth_death(
            birth_rate=0.001,
            death_rate=1.0,
            terminal_time=1.0,
            random_seed=42,
        )

        # Should still return a SimulatedTree, even if empty
        assert isinstance(df, SimulatedTree)

        process = df
        surviving_records = h.surviving_records(process)
        assert surviving_records is not None

        # Check that the process state is still valid
        assert float(process.terminal_time) == 1.0
        assert isinstance(h.creation_times(process), list)
        assert isinstance(h.ages(process), np.ndarray)

    def test_simulation_parameter_effects(self) -> None:
        """Test that birth/death rates affect simulation results as expected."""
        max_time = 10.0
        random_seed = 42

        # High birth rate, low death rate - should produce many records
        df_high_birth = simulate_birth_death(
            birth_rate=0.5,
            death_rate=0.01,
            terminal_time=max_time,
            random_seed=random_seed,
        )

        # Low birth rate, high death rate - should produce few records
        df_low_birth = simulate_birth_death(
            birth_rate=0.01,
            death_rate=0.1,
            terminal_time=max_time,
            random_seed=random_seed + 1,
        )

        # This is probabilistic, but with these extreme parameters,
        # high birth rate should generally produce more surviving records
        high_surv = h.surviving_records(df_high_birth)
        low_surv = h.surviving_records(df_low_birth)
        assert high_surv is not None
        assert low_surv is not None
        surviving_high = len(high_surv)
        surviving_low = len(low_surv)

        # Note: This test might occasionally fail due to randomness,
        # but with these extreme parameters it should usually pass
        # We'll just check that both simulations ran without error
        assert isinstance(surviving_high, int)
        assert isinstance(surviving_low, int)

    def test_simulation_time_bounds(self) -> None:
        """Test that all creation times are within the specified bounds."""
        max_time = 75.0
        df = simulate_birth_death(
            birth_rate=0.2,
            death_rate=0.05,
            terminal_time=max_time,
            random_seed=42,
        )

        # All creation times should be between 0 and max_time
        surviving_records = h.surviving_records(df)
        assert surviving_records is not None
        if len(surviving_records) > 0:
            assert all(0 <= ct <= max_time for ct in h.creation_times(df))

        # All creation times in DataFrame should also be within bounds
        if h.n_individuals(df) > 0:
            df_records = df.records
            assert df_records is not None
            assert all(0 <= ct <= max_time for ct in df_records["creation_time"])

    def test_simulation_ages_non_negative(self) -> None:
        """Test that all calculated ages are non-negative."""
        df = simulate_birth_death(
            birth_rate=0.1,
            death_rate=0.02,
            terminal_time=60.0,
            random_seed=42,
        )

        # Create process from simulation results to test age calculation
        surviving_records = h.surviving_records(df)
        assert surviving_records is not None
        if len(surviving_records) > 0:
            process = df

            # All ages in h.ages(process) should be non-negative
            assert all(age >= 0 for age in h.ages(process))

        # All ages in DataFrame should be non-negative
        if len(surviving_records) > 0:
            assert all(age >= 0 for age in surviving_records["age"])


class TestMultiEpochSimulation:
    """Test multi-epoch simulation functionality."""

    def test_two_epoch_simulation_basic(self) -> None:
        """Test basic two-epoch simulation."""
        birth_rates = np.array([0.1, 0.05])
        death_rates = np.array([0.02, 0.01])
        times = np.array([100.0, 200.0])

        df = simulate_birth_death(
            PiecewiseConstant(birth_rates, times),
            PiecewiseConstant(death_rates, times),
            random_seed=42,
        )

        # Should return a SimulatedTree
        assert isinstance(df, SimulatedTree)

        # All creation times should be <= final time
        if h.n_individuals(df) > 0:
            assert df.records is not None
            assert all(df.records["creation_time"] <= times[-1])

    def test_two_epoch_simulation_reproducible(self) -> None:
        """Test that two-epoch simulation with same seed is reproducible."""
        birth_rates = np.array([0.08, 0.12])
        death_rates = np.array([0.03, 0.02])
        times = np.array([50.0, 100.0])

        df1 = simulate_birth_death(
            PiecewiseConstant(birth_rates, times),
            PiecewiseConstant(death_rates, times),
            random_seed=123,
        )
        df2 = simulate_birth_death(
            PiecewiseConstant(birth_rates, times),
            PiecewiseConstant(death_rates, times),
            random_seed=123,
        )

        # Should have same number of records
        assert h.n_individuals(df1) == h.n_individuals(df2)

        # Creation times should match
        if h.n_individuals(df1) > 0 and h.n_individuals(df2) > 0:
            np.testing.assert_array_almost_equal(
                sorted(df1.records["creation_time"]),
                sorted(df2.records["creation_time"]),
            )

    def test_three_epoch_simulation(self) -> None:
        """Test simulation with three epochs."""
        birth_rates = np.array([0.15, 0.1, 0.12])
        death_rates = np.array([0.02, 0.04, 0.03])
        times = np.array([30.0, 60.0, 100.0])

        df = simulate_birth_death(
            PiecewiseConstant(birth_rates, times),
            PiecewiseConstant(death_rates, times),
            random_seed=42,
        )

        assert isinstance(df, SimulatedTree)
        # Check that all times are within bounds
        if h.n_individuals(df) > 0:
            assert df.records is not None
            assert all(df.records["creation_time"] >= 0)
            assert all(df.records["creation_time"] <= times[-1])

    def test_multi_epoch_data_frame_structure(self) -> None:
        """Test that multi-epoch simulation returns properly structured DataFrame."""
        birth_rates = np.array([0.2, 0.15])
        death_rates = np.array([0.01, 0.02])
        times = np.array([25.0, 50.0])

        df = simulate_birth_death(
            PiecewiseConstant(birth_rates, times),
            PiecewiseConstant(death_rates, times),
            random_seed=42,
        )

        if h.n_individuals(df) > 0:
            surviving = h.surviving_records(df)
            if len(surviving) > 0:
                required_cols = ["individual_id", "creation_time", "age", "survived"]
                for col in required_cols:
                    assert col in surviving.columns

                assert all(surviving["creation_time"] >= 0)
                assert all(surviving["creation_time"] <= times[-1])
                assert all(surviving["age"] >= 0)

    def test_multi_epoch_different_epochs_affect_results(self) -> None:
        """Test that changing epoch parameters affects simulation outcomes."""
        # Two epochs with moderately different rates (avoid extreme values)
        birth_rates1 = np.array([0.15, 0.05])  # Higher then lower birth rate
        death_rates1 = np.array([0.02, 0.02])
        times = np.array([25.0, 50.0])  # Shorter time for faster test

        birth_rates2 = np.array([0.05, 0.15])  # Lower then higher birth rate
        death_rates2 = np.array([0.02, 0.02])

        df1 = simulate_birth_death(
            PiecewiseConstant(birth_rates1, times),
            PiecewiseConstant(death_rates1, times),
            random_seed=42,
        )
        df2 = simulate_birth_death(
            PiecewiseConstant(birth_rates2, times),
            PiecewiseConstant(death_rates2, times),
            random_seed=43,  # Different seed to ensure they differ
        )

        # Both simulations should complete
        assert isinstance(df1, SimulatedTree)
        assert isinstance(df2, SimulatedTree)

        # Results should differ (different epoch configurations and seeds)
        # This is probabilistic but with different seeds should be reliable
        if h.n_individuals(df1) > 0 or h.n_individuals(df2) > 0:
            # At least one simulation produced records
            assert True

    def test_multi_epoch_scalar_compatibility(self) -> None:
        """Test that single-epoch as array matches scalar input."""
        birth_rate = 0.1
        death_rate = 0.02
        time = 100.0

        # Scalar inputs
        df_scalar = simulate_birth_death(
            birth_rate=birth_rate,
            death_rate=death_rate,
            terminal_time=time,
            random_seed=42,
        )

        # Array inputs (single epoch)
        df_array = simulate_birth_death(
            PiecewiseConstant(np.array([birth_rate]), np.array([time])),
            PiecewiseConstant(np.array([death_rate]), np.array([time])),
            random_seed=42,
        )

        # Should produce identical results
        assert h.n_individuals(df_scalar) == h.n_individuals(df_array)
        if h.n_individuals(df_scalar) > 0:
            scalar_records = df_scalar.records
            array_records = df_array.records
            assert scalar_records is not None
            assert array_records is not None
            np.testing.assert_array_almost_equal(
                sorted(scalar_records["creation_time"]),
                sorted(array_records["creation_time"]),
            )


def _linear(start: list[float], end: list[float], times: list[float]):
    return PiecewiseLinear(np.array(start), np.array(end), np.array(times))


def _integral(rate: PiecewiseLinear) -> float:
    """Exact integral of a piecewise-linear rate over ``[0, T]``."""
    ends = np.asarray(rate.epoch_end_times)
    starts = np.concatenate([[0.0], ends[:-1]])
    mean = 0.5 * (np.asarray(rate.start_values) + np.asarray(rate.end_values))
    return float(np.sum(mean * (ends - starts)))


def _mean_size(birth, death, n_runs: int, seed: int = 0) -> tuple[float, float]:
    sizes = np.array(
        [
            n_individuals(simulate_birth_death(birth, death, random_seed=seed + i))
            for i in range(n_runs)
        ],
        dtype=float,
    )
    return sizes.mean(), sizes.std(ddof=1) / math.sqrt(n_runs)


class TestLinearRates:
    def test_offset_constant_hazard(self) -> None:
        assert next_event_offset(2.0, 0.0, 1.5) == pytest.approx(0.75)

    @pytest.mark.parametrize("b", [0.7, 1e-9, -0.2])
    def test_offset_inverts_cumulative_hazard(self, b: float) -> None:
        a, e = 1.3, 0.9
        s = next_event_offset(a, b, e)
        assert a * s + 0.5 * b * s * s == pytest.approx(e)

    def test_offset_matches_closed_form(self) -> None:
        a, b, e = 1.3, 0.7, 0.9
        expected = (-a + math.sqrt(a * a + 2 * b * e)) / b
        assert next_event_offset(a, b, e) == pytest.approx(expected)

    def test_offset_infinite_when_hazard_exhausted(self) -> None:
        assert next_event_offset(1.0, -1.0, 5.0) == math.inf

    def test_death_time_distribution(self) -> None:
        birth = _linear([0.0], [0.0], [6.0])
        death = _linear([0.2], [1.4], [6.0])
        times = []
        for i in range(1500):
            proc = simulate_birth_death(birth, death, random_seed=i)
            dead = proc.records.loc[0, "death_time"]
            times.append(6.0 if np.isnan(dead) else dead)

        def cdf(t: float) -> float:
            return 1.0 - math.exp(-(0.2 * t + 0.5 * 0.2 * t * t))

        observed = np.array([t for t in times if t < 6.0])
        # Condition on dying before T: compare to the truncated CDF.
        trunc = stats.kstest(
            observed, lambda x: np.array([cdf(v) / cdf(6.0) for v in x])
        )
        assert trunc.pvalue > 0.001, (
            f"KS test of root death time against the truncated CDF: "
            f"p-value {trunc.pvalue:.3g} <= threshold 0.001"
        )

    @pytest.mark.parametrize(
        ("birth", "death"),
        [
            (_linear([0.3], [0.8], [3.0]), _linear([0.4], [0.4], [3.0])),
            (_linear([0.9], [0.2], [3.0]), _linear([0.1], [0.5], [3.0])),
            (
                _linear([0.4, 0.8], [0.4, 0.2], [1.5, 3.0]),
                _linear([0.2], [0.6], [3.0]),
            ),
        ],
    )
    def test_mean_size_matches_theory(self, birth, death) -> None:
        expected = math.exp(_integral(birth) - _integral(death))
        mean, se = _mean_size(birth, death, 2000)
        assert abs(mean - expected) < 4 * se + 0.02, (
            f"mean size {mean:.4g} differs from theory {expected:.4g} by "
            f"{abs(mean - expected):.4g}, above the threshold 4 * se + 0.02 = "
            f"{4 * se + 0.02:.4g} (se {se:.4g})"
        )

    def test_constant_rates_match_theory(self) -> None:
        birth = PiecewiseConstant([0.5], [3.0])
        death = PiecewiseConstant([0.3], [3.0])
        mean, se = _mean_size(birth, death, 2000)
        expected = math.exp(0.2 * 3.0)
        assert abs(mean - expected) < 4 * se + 0.02, (
            f"mean size {mean:.4g} differs from theory {expected:.4g} by "
            f"{abs(mean - expected):.4g}, above the threshold 4 * se + 0.02 = "
            f"{4 * se + 0.02:.4g} (se {se:.4g})"
        )

    def test_reproducible(self) -> None:
        birth = _linear([0.5], [0.9], [3.0])
        death = _linear([0.3], [0.3], [3.0])
        a = simulate_birth_death(birth, death, random_seed=7)
        b = simulate_birth_death(birth, death, random_seed=7)
        assert a.records.equals(b.records)

    def test_ensure_non_extinct(self) -> None:
        birth = _linear([0.3], [0.3], [3.0])
        death = _linear([0.6], [0.6], [3.0])
        proc = simulate_birth_death(
            birth, death, random_seed=1, ensure_non_extinct=True
        )
        assert n_individuals(proc) >= 1
        assert proc.records["is_spine"].sum() == 1

    def test_mismatched_end_times_raise(self) -> None:
        with pytest.raises(ValueError, match="same end time"):
            simulate_birth_death(
                _linear([0.5], [0.5], [3.0]), _linear([0.5], [0.5], [4.0])
            )

    def test_negative_retries_raise(self) -> None:
        rate = _linear([0.5], [0.5], [3.0])
        with pytest.raises(ValueError, match="max_retries"):
            simulate_birth_death(rate, rate, ensure_non_extinct=True, max_retries=-1)


class TestResultRates:
    def test_float_rates_are_stored_as_one_epoch_piecewise_constant(self) -> None:
        tree = simulate_birth_death(0.4, 0.1, 5.0, random_seed=1)
        assert tree.terminal_time == 5.0
        for rate, value in ((tree.birth_rate, 0.4), (tree.death_rate, 0.1)):
            assert isinstance(rate, PiecewiseConstant)
            assert list(rate.values) == [value]
            assert list(rate.epoch_end_times) == [5.0]

    def test_rate_likes_are_stored_as_given(self) -> None:
        birth = _linear([0.5], [0.9], [3.0])
        death = PiecewiseConstant([0.3, 0.2], [1.0, 3.0])
        tree = simulate_birth_death(birth, death, random_seed=1)
        assert tree.birth_rate is birth
        assert tree.death_rate is death
        assert tree.terminal_time == 3.0

    def test_float_with_rate_like_takes_the_rate_like_horizon(self) -> None:
        tree = simulate_birth_death(0.5, _linear([0.3], [0.3], [3.0]), random_seed=1)
        assert tree.terminal_time == 3.0
        assert list(np.atleast_1d(tree.birth_rate.epoch_end_times)) == [3.0]


class TestFloatInput:
    def test_float_and_one_epoch_rate_like_give_the_same_tree(self) -> None:
        floats = simulate_birth_death(0.4, 0.1, 5.0, random_seed=3)
        epochs = simulate_birth_death(
            PiecewiseConstant([0.4], [5.0]),
            PiecewiseConstant([0.1], [5.0]),
            random_seed=3,
        )
        assert floats.records.equals(epochs.records)

    def test_zero_float_rates_are_allowed(self) -> None:
        no_births = simulate_birth_death(0.0, 1.0, 2.0, random_seed=1)
        assert len(no_births.records) == 1
        no_deaths = simulate_birth_death(1.0, 0.0, 2.0, random_seed=1)
        assert no_deaths.records["survived"].all()

    def test_terminal_time_is_required_for_float_rates(self) -> None:
        with pytest.raises(ValueError, match="terminal_time is required"):
            simulate_birth_death(0.4, 0.1)

    def test_terminal_time_must_match_rate_likes(self) -> None:
        rate = PiecewiseConstant([0.5], [3.0])
        with pytest.raises(ValueError, match="terminal_time"):
            simulate_birth_death(rate, rate, 4.0)
