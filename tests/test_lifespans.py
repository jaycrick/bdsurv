"""Tests for the lifespan-at-birth birth-death simulator."""

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _helpers import (
    ages,
    creation_times,
    extinct_records,
    n_individuals,
    surviving_records,
)
from scipy.stats import ks_2samp

from bdsurv import (
    LifespanSimulationArrays,
    simulate_birth_death_lifespans,
    simulate_birth_death_lifespans_arrays,
    simulate_birth_death_process,
)
from bdsurv._extinction import pr_extinction
from bdsurv.lifespans import (
    _sample_piecewise_exponential_death_time,  # pyright: ignore[reportPrivateUsage]
)
from bdsurv.tree import SimulatedProcess

_SEED = 42


class TestLifespanSimulation:
    """Test single-epoch simulate_birth_death_lifespans."""

    def test_returns_birth_death_process(self) -> None:
        result = simulate_birth_death_lifespans(0.1, 0.02, 100.0, random_seed=_SEED)
        assert isinstance(result, SimulatedProcess)

    def test_epoch_end_time_correct(self) -> None:
        result = simulate_birth_death_lifespans(0.1, 0.02, 100.0, random_seed=_SEED)
        assert float(result.epoch_end_times[-1]) == 100.0

    def test_creation_times_in_bounds(self) -> None:
        T = 100.0
        result = simulate_birth_death_lifespans(0.1, 0.02, T, random_seed=_SEED)
        cts = np.array(creation_times(result))
        assert np.all(cts >= 0.0)
        assert np.all(cts <= T)

    def test_ages_non_negative(self) -> None:
        result = simulate_birth_death_lifespans(0.1, 0.02, 100.0, random_seed=_SEED)
        assert np.all(ages(result) >= 0.0)

    def test_ages_consistency(self) -> None:
        """Age == terminal_time - creation_time for each survivor."""
        T = 100.0
        result = simulate_birth_death_lifespans(0.1, 0.02, T, random_seed=_SEED)
        surviving = surviving_records(result)
        assert surviving is not None
        if surviving.empty:
            pytest.skip("no survivors for this seed")
        manual_ages = T - surviving["creation_time"].to_numpy(dtype=np.float64)
        np.testing.assert_array_almost_equal(
            sorted(surviving["age"].to_numpy(dtype=np.float64)),
            sorted(manual_ages),
        )

    def test_surviving_records_structure(self) -> None:
        result = simulate_birth_death_lifespans(0.2, 0.01, 50.0, random_seed=_SEED)
        surviving = surviving_records(result)
        assert surviving is not None
        if surviving.empty:
            pytest.skip("no survivors for this seed")
        for col in ["individual_id", "creation_time", "age", "survived"]:
            assert col in surviving.columns
        assert all(surviving["survived"])
        assert all(surviving["age"] >= 0.0)
        assert all(surviving["creation_time"] >= 0.0)

    def test_extinct_records_structure(self) -> None:
        result = simulate_birth_death_lifespans(0.2, 0.01, 50.0, random_seed=_SEED)
        dead = extinct_records(result)
        assert dead is not None
        if dead.empty:
            pytest.skip("no extinctions for this seed")
        for col in [
            "individual_id",
            "creation_time",
            "death_time",
            "lifespan",
            "survived",
        ]:
            assert col in dead.columns
        assert all(~dead["survived"])
        assert all(dead["death_time"] > dead["creation_time"])
        assert all(dead["lifespan"] >= 0.0)

    def test_reproducible_with_same_seed(self) -> None:
        p1 = simulate_birth_death_lifespans(0.1, 0.02, 100.0, random_seed=_SEED)
        p2 = simulate_birth_death_lifespans(0.1, 0.02, 100.0, random_seed=_SEED)
        assert n_individuals(p1) == n_individuals(p2)
        assert creation_times(p1) == creation_times(p2)

    def test_different_seeds_give_different_results(self) -> None:
        p1 = simulate_birth_death_lifespans(0.1, 0.02, 100.0, random_seed=_SEED)
        p2 = simulate_birth_death_lifespans(0.1, 0.02, 100.0, random_seed=_SEED + 1)
        if n_individuals(p1) > 0 and n_individuals(p2) > 0:
            assert creation_times(p1) != creation_times(p2)

    def test_empty_result_is_valid(self) -> None:
        """Very high death rate should often produce extinction; result must be valid."""
        result = simulate_birth_death_lifespans(0.001, 2.0, 1.0, random_seed=_SEED)
        assert isinstance(result, SimulatedProcess)
        assert float(result.epoch_end_times[-1]) == 1.0
        assert isinstance(creation_times(result), list)

    def test_exactly_one_spine_when_survivors_present(self) -> None:
        result = simulate_birth_death_lifespans(
            0.1, 0.02, 100.0, random_seed=_SEED, ensure_non_extinct=True
        )
        surviving = surviving_records(result)
        assert surviving is not None
        assert not surviving.empty
        assert surviving["is_spine"].sum() == 1

    def test_ensure_non_extinct_has_survivors(self) -> None:
        result = simulate_birth_death_lifespans(
            0.001, 0.02, 100.0, random_seed=_SEED, ensure_non_extinct=True
        )
        surviving = surviving_records(result)
        assert surviving is not None
        assert not surviving.empty

    def test_birth_epoch_column_present(self) -> None:
        result = simulate_birth_death_lifespans(0.1, 0.02, 100.0, random_seed=_SEED)
        assert result.records is not None
        assert "birth_epoch" in result.records.columns

    def test_overflow_raises(self) -> None:
        with pytest.raises(RuntimeError, match="max_population"):
            simulate_birth_death_lifespans(
                0.5, 0.01, 100.0, random_seed=1, max_population=5
            )

    def test_invalid_lengths_raises(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"birth_rates, death_rates, and times must have the same lengt",
        ):
            simulate_birth_death_lifespans([0.1, 0.2], [0.05], [100.0, 200.0])

    def test_non_positive_birth_rate_raises(self) -> None:
        with pytest.raises(
            ValueError, match=r"Birth rates and death rates must be positive"
        ):
            simulate_birth_death_lifespans(0.0, 0.05, 100.0)

    def test_non_positive_times_raises(self) -> None:
        with pytest.raises(ValueError, match=r"Times must be positive"):
            simulate_birth_death_lifespans(0.1, 0.02, -1.0)


class TestLifespanMultiEpoch:
    """Test multi-epoch simulate_birth_death_lifespans.

    Use short time spans to keep population well within max_population:
    rates ~0.1/0.02 over 20+20 years → E[N] ~ 55 at terminal time.
    """

    # Short durations keep expected population manageable (E[N] < 100)
    _T2: ClassVar[list[float]] = [20.0, 40.0]
    _B2: ClassVar[list[float]] = [0.1, 0.15]
    _D2: ClassVar[list[float]] = [0.02, 0.03]

    def test_two_epoch_returns_process(self) -> None:
        result = simulate_birth_death_lifespans(
            self._B2, self._D2, self._T2, random_seed=_SEED
        )
        assert isinstance(result, SimulatedProcess)
        assert float(result.epoch_end_times[-1]) == 40.0

    def test_three_epoch_structure(self) -> None:
        result = simulate_birth_death_lifespans(
            [0.1, 0.15, 0.12],
            [0.02, 0.03, 0.025],
            [15.0, 30.0, 45.0],
            random_seed=_SEED,
        )
        assert isinstance(result, SimulatedProcess)
        assert len(result.epoch_end_times) == 3

    def test_scalar_equals_single_element_array(self) -> None:
        p_scalar = simulate_birth_death_lifespans(0.1, 0.02, 100.0, random_seed=_SEED)
        p_array = simulate_birth_death_lifespans(
            np.array([0.1]), np.array([0.02]), np.array([100.0]), random_seed=_SEED
        )
        assert creation_times(p_scalar) == creation_times(p_array)

    def test_two_epoch_creation_times_in_bounds(self) -> None:
        T = 40.0
        result = simulate_birth_death_lifespans(
            self._B2, self._D2, [20.0, T], random_seed=_SEED
        )
        cts = np.array(creation_times(result))
        assert np.all(cts >= 0.0)
        assert np.all(cts <= T)

    def test_epoch_rates_affect_results(self) -> None:
        """Different epoch rates should (with high probability) give different results."""
        p1 = simulate_birth_death_lifespans(
            [0.1, 0.1], [0.02, 0.02], [20.0, 40.0], random_seed=_SEED
        )
        p2 = simulate_birth_death_lifespans(
            [0.1, 0.3], [0.02, 0.25], [20.0, 40.0], random_seed=_SEED
        )
        # Different second-epoch rates should produce different results
        assert n_individuals(p1) != n_individuals(p2) or creation_times(
            p1
        ) != creation_times(p2)

    def test_two_epoch_birth_epochs_in_range(self) -> None:
        result = simulate_birth_death_lifespans(
            self._B2, self._D2, self._T2, random_seed=_SEED
        )
        records = result.records
        assert records is not None
        assert records["birth_epoch"].between(0, 1).all()

    def test_two_epoch_reproducible(self) -> None:
        p1 = simulate_birth_death_lifespans(
            self._B2, self._D2, self._T2, random_seed=_SEED
        )
        p2 = simulate_birth_death_lifespans(
            self._B2, self._D2, self._T2, random_seed=_SEED
        )
        assert creation_times(p1) == creation_times(p2)

    def test_non_increasing_times_raises(self) -> None:
        with pytest.raises(ValueError, match=r"Times must be strictly increasing"):
            simulate_birth_death_lifespans([0.1, 0.15], [0.02, 0.03], [40.0, 20.0])


class TestLifespanDeathTimeSampling:
    """Unit tests for the piecewise-exponential death-time helper."""

    def _make_key(self, seed: int) -> jax.Array:
        return jax.random.key(seed)

    def test_single_epoch_survives_when_hazard_tiny(self) -> None:
        """Near-zero death rate → individual nearly always survives (death time = inf)."""
        dr = jnp.array([1e-9], dtype=jnp.float64)
        et = jnp.array([1_000.0], dtype=jnp.float64)
        terminal = jnp.float64(1_000.0)
        n_inf = sum(
            1
            for seed in range(50)
            if not jnp.isfinite(
                _sample_piecewise_exponential_death_time(
                    jnp.float64(0.0), dr, et, terminal, self._make_key(seed)
                )
            )
        )
        # With rate 1e-9 and T=1000, expected hazard = 1e-6 → almost all survive
        assert n_inf >= 45

    def test_single_epoch_high_rate_mostly_dies(self) -> None:
        """High death rate → nearly all individuals die before terminal time."""
        dr = jnp.array([5.0], dtype=jnp.float64)
        et = jnp.array([100.0], dtype=jnp.float64)
        terminal = jnp.float64(100.0)
        n_dead = sum(
            1
            for seed in range(50)
            if jnp.isfinite(
                _sample_piecewise_exponential_death_time(
                    jnp.float64(0.0), dr, et, terminal, self._make_key(seed)
                )
            )
        )
        assert n_dead >= 45

    def test_single_epoch_death_time_in_range(self) -> None:
        """Finite death times must lie in (birth_time, terminal_time]."""
        dr = jnp.array([0.1], dtype=jnp.float64)
        et = jnp.array([50.0], dtype=jnp.float64)
        terminal = jnp.float64(50.0)
        birth = jnp.float64(10.0)
        for seed in range(30):
            dt = _sample_piecewise_exponential_death_time(
                birth, dr, et, terminal, self._make_key(seed)
            )
            if jnp.isfinite(dt):
                assert float(dt) > float(birth)
                assert float(dt) <= float(terminal) + 1e-9

    def test_two_epoch_death_time_in_range(self) -> None:
        """Piecewise (2-epoch) finite death times must lie after birth_time."""
        dr = jnp.array([0.05, 0.15], dtype=jnp.float64)
        et = jnp.array([50.0, 100.0], dtype=jnp.float64)
        terminal = jnp.float64(100.0)
        birth = jnp.float64(0.0)
        for seed in range(30):
            dt = _sample_piecewise_exponential_death_time(
                birth, dr, et, terminal, self._make_key(seed)
            )
            if jnp.isfinite(dt):
                assert float(dt) > 0.0
                assert float(dt) <= 100.0 + 1e-9

    def test_statistical_mean_matches_theory(self) -> None:
        """Single-epoch sampled death time mean ≈ theoretical mean of a truncated Exp."""
        rate = 0.02
        T = 50.0
        dr = jnp.array([rate], dtype=jnp.float64)
        et = jnp.array([T], dtype=jnp.float64)
        terminal = jnp.float64(T)
        birth = jnp.float64(0.0)

        n = 2_000
        keys = jax.random.split(self._make_key(0), n)
        samples = np.asarray(
            jax.vmap(
                lambda key: _sample_piecewise_exponential_death_time(
                    birth, dr, et, terminal, key
                )
            )(keys)
        )
        # Only finite (dead) samples contribute to the conditional mean
        dead = [s for s in samples if np.isfinite(s)]
        if len(dead) < 50:
            pytest.skip("too few dead samples to compute mean")

        # E[death | death < T] = (1/rate) - T * exp(-rate*T) / (1 - exp(-rate*T))
        p_die = 1.0 - np.exp(-rate * T)
        theoretical_mean = (1.0 / rate) - T * np.exp(-rate * T) / p_die
        empirical_mean = float(np.mean(dead))
        # Loose tolerance: 15% relative
        assert abs(empirical_mean - theoretical_mean) / theoretical_mean < 0.15


class TestLifespanArraysEntryPoint:
    """Tests for simulate_birth_death_lifespans_arrays."""

    def test_returns_namedtuple(self) -> None:
        result = simulate_birth_death_lifespans_arrays(
            0.1, 0.02, 100.0, random_seed=_SEED
        )
        assert isinstance(result, LifespanSimulationArrays)

    def test_field_shapes_consistent(self) -> None:
        result = simulate_birth_death_lifespans_arrays(
            0.1, 0.02, 100.0, random_seed=_SEED
        )
        n = result.n_born
        assert result.birth_times.shape == (n,)
        assert result.death_times.shape == (n,)
        assert result.birth_epochs.shape == (n,)
        assert result.survived.shape == (n,)

    def test_survived_matches_death_times(self) -> None:
        T = 100.0
        result = simulate_birth_death_lifespans_arrays(0.1, 0.02, T, random_seed=_SEED)
        expected_survived = result.death_times > T
        np.testing.assert_array_equal(
            np.asarray(result.survived), np.asarray(expected_survived)
        )

    def test_epoch_count_per_epoch(self) -> None:
        """birth_epochs for 2-epoch run must be 0 or 1 only."""
        result = simulate_birth_death_lifespans_arrays(
            [0.1, 0.15], [0.02, 0.03], [20.0, 40.0], random_seed=_SEED
        )
        epochs = np.asarray(result.birth_epochs)
        assert np.all((epochs == 0) | (epochs == 1))

    def test_overflow_raises(self) -> None:
        with pytest.raises(RuntimeError, match="max_population"):
            simulate_birth_death_lifespans_arrays(
                0.5, 0.01, 100.0, random_seed=1, max_population=5
            )

    def test_reproducible(self) -> None:
        r1 = simulate_birth_death_lifespans_arrays(0.1, 0.02, 100.0, random_seed=_SEED)
        r2 = simulate_birth_death_lifespans_arrays(0.1, 0.02, 100.0, random_seed=_SEED)
        assert r1.n_born == r2.n_born
        np.testing.assert_array_equal(
            np.asarray(r1.birth_times), np.asarray(r2.birth_times)
        )


@pytest.mark.slow
class TestLifespanEquivalence:
    """Statistical equivalence: lifespan sim ≈ Gillespie sim ≈ theory (KS test)."""

    _N_SIMS = 300
    _BIRTH_RATE = 0.01
    _DEATH_RATE = 0.003
    _T = 600.0

    def _run_many(
        self,
        use_lifespan: bool,
        n: int,
        seed_offset: int = 0,
    ) -> list[np.ndarray]:
        """Return list of age arrays (one per non-extinct simulation)."""
        fn = (
            simulate_birth_death_lifespans
            if use_lifespan
            else simulate_birth_death_process
        )
        age_arrays: list[np.ndarray] = []
        for s in range(n):
            proc = fn(
                self._BIRTH_RATE,
                self._DEATH_RATE,
                self._T,
                random_seed=s + seed_offset,
            )
            surv = surviving_records(proc)
            if surv is not None and not surv.empty:
                age_arrays.append(surv["age"].to_numpy(dtype=np.float64))
        return age_arrays

    def test_mean_population_close_to_gillespie(self) -> None:
        """Mean survivor count should agree between the two simulators."""
        lifespan_pops = [
            len(a) for a in self._run_many(use_lifespan=True, n=self._N_SIMS)
        ]
        gillespie_pops = [
            len(a)
            for a in self._run_many(
                use_lifespan=False, n=self._N_SIMS, seed_offset=10_000
            )
        ]
        if not lifespan_pops or not gillespie_pops:
            pytest.skip("all simulations went extinct")

        mean_l = float(np.mean(lifespan_pops))
        mean_g = float(np.mean(gillespie_pops))
        # Mean population should agree within 30% (simulation noise is high)
        rel_diff = abs(mean_l - mean_g) / max(mean_g, 1.0)
        assert rel_diff < 0.30, (
            f"Lifespan mean pop {mean_l:.1f} vs Gillespie {mean_g:.1f} "
            f"(rel diff {rel_diff:.2%})"
        )

    def test_age_distribution_not_rejected_by_ks(self) -> None:
        """Pooled survivor ages: KS test should not reject at p < 0.01."""
        lifespan_ages = np.concatenate(
            self._run_many(use_lifespan=True, n=self._N_SIMS)
        )
        gillespie_ages = np.concatenate(
            self._run_many(use_lifespan=False, n=self._N_SIMS, seed_offset=10_000)
        )
        if lifespan_ages.size < 30 or gillespie_ages.size < 30:
            pytest.skip("too few survivors for KS test")

        stat, p_value = ks_2samp(lifespan_ages, gillespie_ages)
        assert p_value > 0.01, (
            f"KS test rejected age distributions: stat={stat:.4f}, p={p_value:.4f}. "
            "This may be a fluke; re-run before concluding a bug."
        )

    def test_extinction_probability_close_to_theory(self) -> None:
        """Fraction of extinct simulations should be near the theoretical value."""
        n = self._N_SIMS
        n_extinct = 0
        for s in range(n):
            proc = simulate_birth_death_lifespans(
                self._BIRTH_RATE, self._DEATH_RATE, self._T, random_seed=s
            )
            surv = surviving_records(proc)
            if surv is None or surv.empty:
                n_extinct += 1

        empirical_extinct = n_extinct / n
        theoretical_extinct = pr_extinction(self._BIRTH_RATE, self._DEATH_RATE, self._T)
        # Binomial std for proportion: sqrt(p(1-p)/n)
        binom_std = float(np.sqrt(theoretical_extinct * (1 - theoretical_extinct) / n))
        assert abs(empirical_extinct - theoretical_extinct) < 4 * binom_std, (
            f"Empirical extinction {empirical_extinct:.3f} vs "
            f"theoretical {theoretical_extinct:.3f} "
            f"(4σ bound = {4 * binom_std:.3f})"
        )
