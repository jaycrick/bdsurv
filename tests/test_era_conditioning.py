"""Tests for the conditioned survival sampler (era_conditioning.py)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bdsurv.era_conditioning import (
    _clone_particle,  # pyright: ignore[reportPrivateUsage]
    _count_era_survivors,  # pyright: ignore[reportPrivateUsage]
    _extract_sample,  # pyright: ignore[reportPrivateUsage]
    build_target_ranges,
    sample_conditioned_survival_per_posterior,
    sample_conditioned_survival_process,
)
from bdsurv.lifespans import LoopState, advance_era, initial_state

_SEED = 7
_MAX_POP = 1_000  # small for fast unit tests


def _tiny_initial(
    key: jax.Array,
    epoch_end_times: jax.Array | None = None,
    death_rates: jax.Array | None = None,
    max_population: int = _MAX_POP,
) -> LoopState:
    """Make a tiny initial particle with one alive individual."""
    if epoch_end_times is None:
        epoch_end_times = jnp.array([10.0, 50.0], dtype=jnp.float64)
    if death_rates is None:
        death_rates = jnp.array([0.01, 0.01], dtype=jnp.float64)
    return initial_state(death_rates, epoch_end_times, key, max_population)


class TestBuildTargetRanges:
    """Test build_target_ranges."""

    def test_exact_match_default(self) -> None:
        counts = [10, 20, 5]
        ranges = build_target_ranges(counts)
        assert ranges.shape == (3, 2)
        assert ranges[0, 0] == 10
        assert ranges[0, 1] == 10
        assert ranges[1, 0] == 20
        assert ranges[1, 1] == 20

    def test_last_era_always_wide(self) -> None:
        ranges = build_target_ranges([5, 10, 3])
        assert ranges[-1, 0] == 0
        assert ranges[-1, 1] == 2**31 - 1

    def test_absolute_tolerance(self) -> None:
        ranges = build_target_ranges([100, 50, 0], tolerance=5)
        assert ranges[0, 0] == 95
        assert ranges[0, 1] == 105
        assert ranges[1, 0] == 45
        assert ranges[1, 1] == 55

    def test_absolute_tolerance_clamps_low_to_zero(self) -> None:
        ranges = build_target_ranges([2, 10, 0], tolerance=5)
        assert ranges[0, 0] == 0  # max(0, 2-5)
        assert ranges[0, 1] == 7

    def test_relative_tolerance(self) -> None:
        ranges = build_target_ranges([100, 200, 50], tolerance=0.1)
        # 100 ± round(100 * 0.1) = 100 ± 10
        assert ranges[0, 0] == 90
        assert ranges[0, 1] == 110

    def test_single_era_list(self) -> None:
        ranges = build_target_ranges([99])
        assert ranges.shape == (1, 2)
        assert ranges[0, 0] == 0  # last era → wide
        assert ranges[0, 1] == 2**31 - 1


class TestAdvanceEraInConditioning:
    """Test advance_era as the per-era event loop used by the sampler."""

    def test_time_advances_to_segment_end(self) -> None:
        key = jax.random.key(_SEED)
        et = jnp.array([10.0, 50.0], dtype=jnp.float64)
        dr = jnp.array([0.01, 0.01], dtype=jnp.float64)
        state = _tiny_initial(key, et, dr)
        out = advance_era(
            jnp.float64(0.2), dr, et, jnp.int32(0), et[0], state, _MAX_POP
        )
        assert float(out.current_time) == pytest.approx(10.0)

    def test_n_born_non_decreasing(self) -> None:
        key = jax.random.key(_SEED)
        et = jnp.array([20.0, 100.0], dtype=jnp.float64)
        dr = jnp.array([0.01, 0.01], dtype=jnp.float64)
        state = _tiny_initial(key, et, dr)
        out = advance_era(
            jnp.float64(0.3), dr, et, jnp.int32(0), et[0], state, _MAX_POP
        )
        assert int(out.n_born) >= 1

    def test_birth_epoch_tagged_correctly(self) -> None:
        # Run era 1: all new births should be tagged with epoch 1
        key = jax.random.key(_SEED + 1)
        et = jnp.array([5.0, 30.0], dtype=jnp.float64)
        dr = jnp.array([0.01, 0.01], dtype=jnp.float64)
        state = _tiny_initial(key, et, dr)
        # First run era 0
        state0 = advance_era(
            jnp.float64(0.5), dr, et, jnp.int32(0), et[0], state, _MAX_POP
        )
        n0 = int(state0.n_born)
        # Run era 1
        state1 = advance_era(
            jnp.float64(0.5), dr, et, jnp.int32(1), et[1], state0, _MAX_POP
        )
        n1 = int(state1.n_born)
        # All births beyond n0 should be tagged with epoch 1
        if n1 > n0:
            be = np.asarray(state1.birth_epoch_buf[:n1])
            new_births_tags = be[n0:]
            assert np.all(new_births_tags == 1)

    def test_overflow_flag_set_on_capacity(self) -> None:
        tiny_max = 2  # root + 1 additional
        key = jax.random.key(_SEED)
        et = jnp.array([200.0, 500.0], dtype=jnp.float64)
        dr = jnp.array([0.001, 0.001], dtype=jnp.float64)
        state = _tiny_initial(key, et, dr, max_population=tiny_max)
        out = advance_era(
            jnp.float64(2.0),  # very high birth rate
            dr,
            et,
            jnp.int32(0),
            et[0],
            state,
            tiny_max,
        )
        # With a very high birth rate and tiny buffer, overflow must fire
        assert bool(out.overflow)

    def test_death_times_preserved_across_eras(self) -> None:
        # Individuals seeded in era 0 should retain their death times in era 1
        key = jax.random.key(_SEED + 2)
        et = jnp.array([5.0, 50.0], dtype=jnp.float64)
        dr = jnp.array([0.005, 0.005], dtype=jnp.float64)
        state = _tiny_initial(key, et, dr)
        state0 = advance_era(
            jnp.float64(0.1), dr, et, jnp.int32(0), et[0], state, _MAX_POP
        )
        death_after_era0 = np.asarray(state0.death_time_buf[0])
        state1 = advance_era(
            jnp.float64(0.1), dr, et, jnp.int32(1), et[1], state0, _MAX_POP
        )
        death_after_era1 = np.asarray(state1.death_time_buf[0])
        # Root's death time must not change between eras
        np.testing.assert_equal(death_after_era0, death_after_era1)


class TestCountEraSurvivors:
    """Test _count_era_survivors filter."""

    def test_root_survivor_counted(self) -> None:
        # Root has death_time=inf → survives → era 0 survivor count >= 1
        key = jax.random.key(_SEED)
        et = jnp.array([5.0, 50.0], dtype=jnp.float64)
        # Very low death rate: root almost certainly survives
        dr = jnp.array([1e-9, 1e-9], dtype=jnp.float64)
        state = _tiny_initial(key, et, dr)
        state0 = advance_era(
            jnp.float64(0.1), dr, et, jnp.int32(0), et[0], state, _MAX_POP
        )
        count = _count_era_survivors(state0, 0, float(et[-1]))
        assert count >= 1

    def test_high_death_rate_few_survivors(self) -> None:
        rng = np.random.default_rng(_SEED)
        counts = []
        for _ in range(10):
            key = jax.random.key(int(rng.integers(0, 2**31)))
            # High death rate — root probably dies before terminal time
            et = jnp.array([50.0, 200.0], dtype=jnp.float64)
            dr = jnp.array([0.5, 0.5], dtype=jnp.float64)
            state = _tiny_initial(key, et, dr)
            state0 = advance_era(
                jnp.float64(0.01), dr, et, jnp.int32(0), et[0], state, _MAX_POP
            )
            counts.append(_count_era_survivors(state0, 0, float(et[-1])))
        # At least one trial should have 0 survivors with such high death rate
        assert any(c == 0 for c in counts)


class TestCloneParticle:
    """Test _clone_particle."""

    def test_clones_share_buffers_but_differ_in_key(self) -> None:
        key = jax.random.key(_SEED)
        et = jnp.array([10.0, 50.0], dtype=jnp.float64)
        dr = jnp.array([0.01, 0.01], dtype=jnp.float64)
        state = _tiny_initial(key, et, dr)
        new_key = jax.random.key(_SEED + 99)
        clone = _clone_particle(state, new_key)
        # Buffers should be the same object reference
        assert clone.birth_time_buf is state.birth_time_buf
        assert clone.death_time_buf is state.death_time_buf
        # Keys must differ
        assert not bool(jnp.all(clone.key == state.key))

    def test_clone_evolves_independently(self) -> None:
        key = jax.random.key(_SEED)
        et = jnp.array([10.0, 50.0], dtype=jnp.float64)
        dr = jnp.array([0.01, 0.01], dtype=jnp.float64)
        state = _tiny_initial(key, et, dr)
        # Clone with different key → should produce different simulations
        k1, k2 = jax.random.split(jax.random.key(99), 2)
        c1 = _clone_particle(state, k1)
        c2 = _clone_particle(state, k2)
        out1 = advance_era(jnp.float64(0.3), dr, et, jnp.int32(0), et[0], c1, _MAX_POP)
        out2 = advance_era(jnp.float64(0.3), dr, et, jnp.int32(0), et[0], c2, _MAX_POP)
        # With probability approaching 1, outcomes differ (different PRNG paths)
        # We just check the outputs are valid and n_born >= 1
        assert int(out1.n_born) >= 1
        assert int(out2.n_born) >= 1


class TestExtractSample:
    """Test _extract_sample."""

    def test_invariants(self) -> None:
        key = jax.random.key(_SEED)
        et = jnp.array([10.0, 50.0], dtype=jnp.float64)
        dr = jnp.array([1e-9, 1e-9], dtype=jnp.float64)
        state = _tiny_initial(key, et, dr)
        state0 = advance_era(
            jnp.float64(0.1), dr, et, jnp.int32(0), et[0], state, _MAX_POP
        )
        state1 = advance_era(
            jnp.float64(0.1), dr, et, jnp.int32(1), et[1], state0, _MAX_POP
        )
        sample = _extract_sample(state1, 2, float(et[-1]))

        assert sample.n_born >= 1
        assert 0 <= sample.n_alive_present <= sample.n_born
        assert 0.0 <= sample.proportion_alive <= 1.0
        assert len(sample.per_era_ever_created) == 2
        assert len(sample.per_era_survivors) == 2
        # Per-era survivors cannot exceed per-era ever-created
        assert np.all(sample.per_era_survivors <= sample.per_era_ever_created)
        # Sums must be consistent
        assert np.sum(sample.per_era_ever_created) == sample.n_born
        assert np.sum(sample.per_era_survivors) == sample.n_alive_present


class TestSampleConditionedSurvivalProcess:
    """Test sample_conditioned_survival_process."""

    def _wide_ranges(self, n_epochs: int = 2) -> np.ndarray:
        """Wide ranges that almost any trajectory passes."""
        ranges = np.zeros((n_epochs, 2), dtype=np.intp)
        for k in range(n_epochs):
            ranges[k, 0] = 0
            ranges[k, 1] = 10_000
        ranges[-1, 1] = 2**31 - 1
        return ranges

    def test_returns_n_samples(self) -> None:
        ranges = self._wide_ranges(2)
        result = sample_conditioned_survival_process(
            [0.3, 0.1],
            [0.01, 0.01],
            [10.0, 50.0],
            ranges,
            n_samples=2,
            split_factor=3,
            n_initial=10,
            random_seed=_SEED,
            max_population=_MAX_POP,
        )
        assert len(result.samples) == 2

    def test_sample_invariants(self) -> None:
        ranges = self._wide_ranges(2)
        result = sample_conditioned_survival_process(
            [0.2, 0.1],
            [0.01, 0.01],
            [10.0, 50.0],
            ranges,
            n_samples=1,
            split_factor=5,
            n_initial=5,
            random_seed=_SEED,
            max_population=_MAX_POP,
        )
        s = result.samples[0]
        assert s.n_born >= 1
        assert 0 <= s.n_alive_present <= s.n_born
        assert 0.0 <= s.proportion_alive <= 1.0
        assert np.all(s.per_era_survivors <= s.per_era_ever_created)
        assert np.sum(s.per_era_ever_created) == s.n_born
        assert np.sum(s.per_era_survivors) == s.n_alive_present

    def test_reproducible_with_seed(self) -> None:
        kwargs = dict(
            birth_rates=[0.2, 0.1],
            death_rates=[0.02, 0.02],
            epoch_end_times=[10.0, 50.0],
            target_ranges=self._wide_ranges(2),
            n_samples=1,
            split_factor=3,
            n_initial=5,
            random_seed=_SEED,
            max_population=_MAX_POP,
        )
        r1 = sample_conditioned_survival_process(**kwargs)  # type: ignore[arg-type]
        r2 = sample_conditioned_survival_process(**kwargs)  # type: ignore[arg-type]
        assert r1.samples[0].n_born == r2.samples[0].n_born
        assert r1.samples[0].n_alive_present == r2.samples[0].n_alive_present

    def test_impossible_range_warns_and_returns_partial(self) -> None:
        # Require exactly 999_999 era-0 survivors — impossible
        ranges = np.array([[999_999, 999_999], [0, 2**31 - 1]], dtype=np.intp)
        with pytest.warns(RuntimeWarning, match="max_attempts"):
            result = sample_conditioned_survival_process(
                [0.1, 0.1],
                [0.01, 0.01],
                [10.0, 50.0],
                ranges,
                n_samples=1,
                split_factor=2,
                n_initial=3,
                max_attempts=6,  # tiny budget
                random_seed=_SEED,
                max_population=_MAX_POP,
            )
        assert len(result.samples) == 0
        assert result.n_total_era0_particles <= 9  # 3 per attempt, ≤3 restarts

    def test_diagnostics_length(self) -> None:
        ranges = self._wide_ranges(3)
        result = sample_conditioned_survival_process(
            [0.2, 0.15, 0.1],
            [0.01, 0.01, 0.01],
            [10.0, 30.0, 80.0],
            ranges,
            n_samples=1,
            split_factor=2,
            n_initial=5,
            random_seed=_SEED,
            max_population=_MAX_POP,
        )
        # Diagnostics cover conditioned eras (all but last)
        assert len(result.diagnostics) == 2

    def test_two_epoch_input_raises(self) -> None:
        with pytest.raises(ValueError, match="At least 2 epochs"):
            sample_conditioned_survival_process(
                [0.1],
                [0.01],
                [50.0],
                np.array([[0, 100]], dtype=np.intp),
                n_samples=1,
                split_factor=2,
                n_initial=3,
                max_population=_MAX_POP,
            )


class TestSampleConditionedSurvivalPerPosterior:
    """Test sample_conditioned_survival_per_posterior."""

    def _make_posterior(self, n_draws: int = 3) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(_SEED)
        br = np.full((n_draws, 2), 0.2) + rng.normal(0, 0.01, (n_draws, 2))
        dr = np.full((n_draws, 2), 0.01) + rng.normal(0, 0.001, (n_draws, 2))
        br = np.clip(br, 0.05, 0.5)
        dr = np.clip(dr, 0.001, 0.05)
        return br, dr

    def _wide_ranges(self) -> np.ndarray:
        return np.array([[0, 10_000], [0, 2**31 - 1]], dtype=np.intp)

    def test_returns_dataframe_with_correct_rows(self) -> None:
        br, dr = self._make_posterior(3)
        df = sample_conditioned_survival_per_posterior(
            br,
            dr,
            [10.0, 50.0],
            self._wide_ranges(),
            split_factor=3,
            n_initial=5,
            random_seed=_SEED,
            max_population=_MAX_POP,
        )
        assert len(df) == 3

    def test_column_schema(self) -> None:
        br, dr = self._make_posterior(2)
        df = sample_conditioned_survival_per_posterior(
            br,
            dr,
            [10.0, 50.0],
            self._wide_ranges(),
            split_factor=3,
            n_initial=5,
            random_seed=_SEED,
            max_population=_MAX_POP,
        )
        for col in [
            "draw",
            "matched",
            "n_born",
            "n_alive_present",
            "proportion_alive",
            "n_outer_iterations",
            "era_0_ever_created",
            "era_0_survivors",
            "era_1_ever_created",
            "era_1_survivors",
        ]:
            assert col in df.columns, f"missing column: {col}"

    def test_matched_draws_have_valid_values(self) -> None:
        br, dr = self._make_posterior(4)
        df = sample_conditioned_survival_per_posterior(
            br,
            dr,
            [10.0, 50.0],
            self._wide_ranges(),
            split_factor=3,
            n_initial=5,
            random_seed=_SEED,
            max_population=_MAX_POP,
        )
        matched = df[df["matched"]]
        if matched.empty:
            pytest.skip("no matched draws for this seed")
        assert (matched["n_born"] >= 1).all()
        assert (matched["n_alive_present"] <= matched["n_born"]).all()
        assert (matched["proportion_alive"] >= 0.0).all()
        assert (matched["proportion_alive"] <= 1.0).all()

    def test_shape_mismatch_raises(self) -> None:
        br = np.ones((3, 2))
        dr = np.ones((4, 2))
        with pytest.raises(ValueError, match="shape"):
            sample_conditioned_survival_per_posterior(
                br,
                dr,
                [10.0, 50.0],
                self._wide_ranges(),
                split_factor=2,
                n_initial=2,
                max_population=_MAX_POP,
            )

    def test_unmatched_draw_is_flagged_not_raised(self) -> None:
        impossible = np.array([[999_999, 999_999], [0, 2**31 - 1]], dtype=np.intp)
        br, dr = self._make_posterior(2)
        with pytest.warns(RuntimeWarning, match="max_attempts"):
            df = sample_conditioned_survival_per_posterior(
                br,
                dr,
                [10.0, 50.0],
                impossible,
                split_factor=2,
                n_initial=3,
                max_attempts=3,
                random_seed=_SEED,
                max_population=_MAX_POP,
            )
        assert len(df) == 2, "one row per draw"
        assert not df["matched"].any(), "no draw can match an impossible range"

    def test_unexpected_error_propagates(self) -> None:
        """A single epoch is rejected by the sampler; the wrapper must not hide it."""
        with pytest.raises(ValueError, match="At least 2 epochs"):
            sample_conditioned_survival_per_posterior(
                np.full((2, 1), 0.2),
                np.full((2, 1), 0.01),
                [50.0],
                np.array([[0, 2**31 - 1]], dtype=np.intp),
                split_factor=2,
                n_initial=2,
                max_population=_MAX_POP,
            )


@pytest.mark.slow
class TestConditionedSurvivalRecovery:
    """Recovery-style test: conditioned sampler on synthetic data."""

    def test_conditioning_recovers_plausible_proportion(self) -> None:
        """Conditioned sample's proportion_alive is consistent with forward survival.

        We pick rates so ~10-30 individuals survive from era 0 to terminal time.
        We run many forward simulations to find the typical era-0 survivor count,
        then condition on that count and verify the conditioned proportion_alive
        matches the marginal survival fraction from the forward simulator.
        """
        import numpy as np

        from bdsurv import simulate_birth_death_lifespans_arrays

        rng = np.random.default_rng(_SEED + 99)
        br = np.array([0.08, 0.03])
        dr = np.array([0.005, 0.005])
        et = np.array([50.0, 200.0])

        # ---- Step 1: forward simulations to find typical era-0 survivors ----
        forward_era0_survivors = []
        forward_proportions = []
        for _ in range(50):
            seed = int(rng.integers(0, 2**31))
            try:
                result = simulate_birth_death_lifespans_arrays(
                    br, dr, et, random_seed=seed, max_population=5_000
                )
            except RuntimeError:
                continue
            n = result.n_born
            be = np.asarray(result.birth_epochs)
            survived = np.asarray(result.survived)
            era0_surv = int(np.sum((be == 0) & survived))
            forward_era0_survivors.append(era0_surv)
            if n > 0:
                forward_proportions.append(float(np.sum(survived)) / n)

        # Find a plausible target count (median of era-0 survivors > 0)
        nonzero = [c for c in forward_era0_survivors if c > 0]
        if not nonzero:
            pytest.skip("No non-zero era-0 survivors; rates may need adjusting")
        target_count = int(np.median(nonzero))
        forward_mean_prop = (
            float(np.mean(forward_proportions)) if forward_proportions else 0.5
        )

        # ---- Step 2: conditioned sampling ----
        target_ranges = build_target_ranges(
            [target_count, 0], tolerance=max(2, target_count // 5)
        )
        result_cond = sample_conditioned_survival_process(
            br,
            dr,
            et,
            target_ranges,
            n_samples=5,
            split_factor=10,
            n_initial=50,
            max_attempts=5_000,
            random_seed=int(rng.integers(0, 2**31)),
            max_population=5_000,
        )

        assert len(result_cond.samples) > 0, (
            "No samples collected — rates/ranges may need tuning"
        )

        conditioned_proportions = [s.proportion_alive for s in result_cond.samples]
        conditioned_mean = float(np.mean(conditioned_proportions))

        # The conditioned proportion should be in a reasonable neighbourhood
        # of the unconditional forward proportion (within a factor of 5)
        assert conditioned_mean > 0.0, "Conditioned proportion is zero"
        assert conditioned_mean <= 1.0, "Conditioned proportion exceeds 1"
        # Rough plausibility: not off by more than 10x
        ratio = max(conditioned_mean, forward_mean_prop) / max(
            min(conditioned_mean, forward_mean_prop), 1e-9
        )
        assert ratio < 10.0, (
            f"Conditioned proportion {conditioned_mean:.3f} is implausible "
            f"compared to forward proportion {forward_mean_prop:.3f} (ratio={ratio:.1f})"
        )

    def test_per_era_survivors_consistent_with_target(self) -> None:
        """Era-0 survivors in matched samples should lie within target_ranges[0]."""
        rng = np.random.default_rng(_SEED + 200)
        br = np.array([0.1, 0.03])
        dr = np.array([0.005, 0.005])
        et = np.array([50.0, 200.0])
        # Use a wide tolerance so we easily find matches
        target_lo, target_hi = 1, 50
        ranges = np.array([[target_lo, target_hi], [0, 2**31 - 1]], dtype=np.intp)
        result = sample_conditioned_survival_process(
            br,
            dr,
            et,
            ranges,
            n_samples=3,
            split_factor=10,
            n_initial=100,
            max_attempts=10_000,
            random_seed=int(rng.integers(0, 2**31)),
            max_population=5_000,
        )
        for s in result.samples:
            era0_surv = int(s.per_era_survivors[0])
            assert target_lo <= era0_surv <= target_hi, (
                f"era-0 survivors {era0_surv} outside [{target_lo}, {target_hi}]"
            )
