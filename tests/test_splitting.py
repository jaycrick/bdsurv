"""Tests for splitting-based conditioned survival simulation."""

import itertools

import numpy as np
import pytest

from bdsurv import PiecewiseConstant, PiecewiseLinear
from bdsurv._hazard import log_survival_probability_constant
from bdsurv.splitting import (
    _BAND_LOWER,
    _BAND_UPPER,
    SplittingConfig,
    _initial_level_count,
    compute_level_replicates,
    default_max_restarts,
    select_level_times_for_band,
    simulate_survival_splitting,
)
from bdsurv.tree import SimulatedTree


class TestSplittingSimulation:
    """Validate rare-event splitting simulations."""

    def test_splitting_returns_expected_shapes(self) -> None:
        config = SplittingConfig(
            birth_rate=0.9,
            death_rate=1.0,
            terminal_time=40.0,
            n_particles=200,
            n_levels=8,
            random_seed=17,
        )

        result = simulate_survival_splitting(config)

        # n_levels is ignored when level_times is None; levels are auto-selected
        n_levels_used = len(result.level_times) - 1
        assert result.level_times.shape[0] == n_levels_used + 1
        assert len(result.level_diagnostics) == n_levels_used
        assert result.level_mean_populations.shape[0] == n_levels_used + 1

    def test_splitting_reproducible_with_fixed_seed(self) -> None:
        config = SplittingConfig(
            birth_rate=0.9,
            death_rate=1.0,
            terminal_time=25.0,
            n_particles=120,
            n_levels=5,
            random_seed=2026,
        )

        first = simulate_survival_splitting(config)
        second = simulate_survival_splitting(config)

        assert first.final_survivor_count == second.final_survivor_count
        assert np.allclose(first.level_mean_populations, second.level_mean_populations)
        assert [t.particle_id for t in first.conditioned_trajectories] == [
            t.particle_id for t in second.conditioned_trajectories
        ]

    def test_conditioned_trajectories_survive_to_terminal_time(self) -> None:
        config = SplittingConfig(
            birth_rate=0.9,
            death_rate=1.0,
            terminal_time=40.0,
            n_particles=250,
            n_levels=8,
            random_seed=11,
        )

        result = simulate_survival_splitting(config)

        assert len(result.conditioned_trajectories) == result.final_survivor_count

        for trajectory in result.conditioned_trajectories:
            assert isinstance(trajectory.process, SimulatedTree)
            assert np.isclose(trajectory.times[-1], config.terminal_time)
            assert trajectory.populations[-1] > 0

    def test_splitting_level_times_invariant_on_early_extinction(self) -> None:
        # Regression test: when all particles go extinct before the final level
        # the loop breaks early and level_times was previously not truncated,
        # leaving len(level_times) >> len(level_diagnostics) + 1.
        # Use n_particles=2 so that after the first level only 0 or 1 particle
        # can survive; with a near-zero birth rate, all-extinct is guaranteed.
        # n_levels is ignored because level_times=None triggers auto-selection.
        config = SplittingConfig(
            birth_rate=0.05,
            death_rate=1.0,
            terminal_time=50.0,
            n_particles=2,
            n_levels=15,
            random_seed=0,
        )

        with pytest.warns(RuntimeWarning):
            result = simulate_survival_splitting(config)

        # Core invariant: always one more boundary than level diagnostics.
        assert len(result.level_times) == len(result.level_diagnostics) + 1
        assert len(result.level_times) == len(result.level_mean_populations)
        # Confirm the early-exit path was taken: all particles went extinct.
        assert result.conditioned_trajectories == []
        assert result.final_survivor_count == 0

    def test_splitting_spine_is_alive_list_head(self) -> None:
        # Verify that each conditioned trajectory has exactly one is_spine
        # individual that is also a survivor — consistent with the exact
        # Gillespie index-0 alive-list convention.
        config = SplittingConfig(
            birth_rate=0.9,
            death_rate=1.0,
            terminal_time=10.0,
            n_particles=500,
            random_seed=77,
        )

        result = simulate_survival_splitting(config)

        assert len(result.conditioned_trajectories) > 0, "expected non-zero survivors"
        for traj in result.conditioned_trajectories:
            records = traj.process.records
            spine_rows = records[records["is_spine"]]
            assert len(spine_rows) == 1, "expected exactly one spine per tree"
            assert bool(spine_rows["survived"].iloc[0]), "spine must be a survivor"


def _ratios(birth_rate: float, death_rate: float, levels: np.ndarray) -> np.ndarray:
    log_prs = [
        log_survival_probability_constant(
            birth_rate=birth_rate, death_rate=death_rate, time=t
        )
        for t in levels
    ]
    return np.exp(np.diff(log_prs))


def _reference_grid(
    birth_rate: float, death_rate: float, terminal_time: float, n_levels: int
) -> np.ndarray:
    """Plain restatement of the refine-then-coarsen rule, with no protected times."""

    def log_pr(t: float) -> float:
        return log_survival_probability_constant(
            birth_rate=birth_rate, death_rate=death_rate, time=t
        )

    times = list(np.linspace(0.0, terminal_time, n_levels + 1))
    # Refine: add midpoints until every ratio is at least e^-2.
    while True:
        out = [times[0]]
        added = False
        for left, right in itertools.pairwise(times):
            if log_pr(right) - log_pr(left) < -2.0:
                out.append(0.5 * (left + right))
                added = True
            out.append(right)
        times = out
        if not added:
            break
    # Coarsen: try dropping t_i when its ratio exceeds 1/2, sequentially, until stable.
    changed = True
    while changed:
        changed = False
        i = 1
        while i < len(times) - 1:
            too_large = log_pr(times[i]) - log_pr(times[i - 1]) > np.log(0.5)
            merged_ok = log_pr(times[i + 1]) - log_pr(times[i - 1]) >= -2.0
            if too_large and merged_ok:
                del times[i]
                changed = True
            else:
                i += 1
    return np.asarray(times)


class TestLevelGrid:
    """Adaptive level-grid selection."""

    def test_log_survival_matches_analytic_survival(self) -> None:
        # Pr(Z_t > 0) = (lam - mu) / (lam - mu * exp(-(lam - mu) * t)).
        lam, mu = 0.9, 1.0
        for t in (0.1, 1.0, 20.0):
            expected = (lam - mu) / (lam - mu * np.exp(-(lam - mu) * t))
            got = np.exp(
                log_survival_probability_constant(
                    birth_rate=0.9, death_rate=1.0, time=t
                )
            )
            assert np.isclose(got, expected, rtol=1e-9)

    def test_log_survival_stable_for_huge_horizon(self) -> None:
        value = log_survival_probability_constant(
            birth_rate=1.0, death_rate=2.0, time=1e6
        )
        assert np.isfinite(value)

    @pytest.mark.parametrize(
        ("birth_rate", "death_rate", "terminal_time", "n_levels"),
        [
            (0.9, 1.0, 50.0, 10),
            (1.0, 2.0, 20.0, 1),
            (1.0, 10.0, 20.0, 400),
            (1.0, 1.1, 5.0, 3),
        ],
    )
    def test_grid_is_valid_and_in_band(
        self,
        birth_rate: float,
        death_rate: float,
        terminal_time: float,
        n_levels: int,
    ) -> None:
        levels = select_level_times_for_band(
            birth_rate=birth_rate,
            death_rate=death_rate,
            terminal_time=terminal_time,
            n_levels=n_levels,
        )
        assert float(levels[0]) == 0.0
        assert np.isclose(float(levels[-1]), terminal_time)
        assert np.all(np.diff(levels) > 0)

        ratios = _ratios(birth_rate, death_rate, levels)
        # Nothing may fall below the band unless the whole horizon is that short.
        assert np.all(ratios >= _BAND_LOWER - 1e-12) or len(levels) == 2
        # Coarsening is stable: removing an interior point whose ratio is above
        # 1/2 would push the merged ratio below the band.
        for j in range(len(ratios) - 1):
            if ratios[j] > _BAND_UPPER + 1e-12:
                assert ratios[j] * ratios[j + 1] < _BAND_LOWER + 1e-12

    def test_matches_reference_implementation(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(60):
            birth_rate = float(rng.uniform(0.3, 3.0))
            death_rate = float(rng.uniform(0.3, 6.0))
            terminal_time = float(rng.uniform(0.5, 60.0))
            n_levels = int(rng.integers(1, 40))
            got = select_level_times_for_band(
                birth_rate=birth_rate,
                death_rate=death_rate,
                terminal_time=terminal_time,
                n_levels=n_levels,
            )
            expected = _reference_grid(birth_rate, death_rate, terminal_time, n_levels)
            np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    def test_coarse_start_is_refined(self) -> None:
        levels = select_level_times_for_band(
            birth_rate=1.0, death_rate=2.0, terminal_time=20.0, n_levels=1
        )
        assert len(levels) > 2
        assert np.all(_ratios(1.0, 2.0, levels) >= _BAND_LOWER - 1e-12)

    def test_fine_start_is_coarsened(self) -> None:
        fine = select_level_times_for_band(
            birth_rate=1.0, death_rate=2.0, terminal_time=20.0, n_levels=1000
        )
        assert len(fine) < 1001
        assert np.all(_ratios(1.0, 2.0, fine) >= _BAND_LOWER - 1e-12)

    def test_initial_level_times_are_used(self) -> None:
        levels = select_level_times_for_band(
            birth_rate=0.9,
            death_rate=1.0,
            terminal_time=10.0,
            initial_level_times=[0.0, 5.0, 10.0],
        )
        assert float(levels[0]) == 0.0
        assert float(levels[-1]) == 10.0


class TestReplicates:
    """Per-level replicate counts."""

    def test_half_ratio_gives_two(self) -> None:
        # Find t_1 with Pr(t_1) = 1/2 for (lambda, mu) = (1, 2): 1/(1+2*kernel)
        # with kernel = 1 - exp(-t) ... solve numerically.
        lo, hi = 1e-6, 50.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            pr = np.exp(
                log_survival_probability_constant(
                    birth_rate=1.0, death_rate=2.0, time=mid
                )
            )
            if pr > 0.5:
                lo = mid
            else:
                hi = mid
        replicates = compute_level_replicates(
            birth_rate=1.0, death_rate=2.0, level_times=[0.0, 0.5 * (lo + hi)]
        )
        assert replicates.tolist() == [2]

    def test_expected_survivors_at_least_one_and_minimal(self) -> None:
        birth_rate, death_rate = 0.9, 1.0
        levels = select_level_times_for_band(
            birth_rate=birth_rate, death_rate=death_rate, terminal_time=40.0
        )
        replicates = compute_level_replicates(
            birth_rate=birth_rate, death_rate=death_rate, level_times=levels
        )
        assert np.all(replicates >= 2)
        pr = np.exp(
            [
                log_survival_probability_constant(
                    birth_rate=birth_rate, death_rate=death_rate, time=t
                )
                for t in levels
            ]
        )
        cumulative = np.cumprod(replicates.astype(float))
        assert np.all(cumulative * pr[1:] >= 1.0 - 1e-9)
        for i, r in enumerate(replicates):
            if r > 2:
                smaller = cumulative[i] / r * (r - 1)
                assert smaller * pr[i + 1] < 1.0


class TestRestarts:
    """Restart cap and behaviour."""

    def test_default_max_restarts_meets_target(self) -> None:
        levels = np.array([0.0, 1.0, 2.0, 3.0])
        replicates = compute_level_replicates(
            birth_rate=0.9, death_rate=1.0, level_times=levels
        )
        n = default_max_restarts(
            birth_rate=0.9,
            death_rate=1.0,
            level_times=levels,
            replicates=replicates,
        )
        log_prs = [
            log_survival_probability_constant(birth_rate=0.9, death_rate=1.0, time=t)
            for t in levels
        ]
        failure = 0.0
        for i in range(len(replicates), 0, -1):
            r = np.exp(log_prs[i] - log_prs[i - 1])
            failure = (1.0 - r + r * failure) ** int(replicates[i - 1])
        assert 1.0 - failure**n >= 0.999
        assert n == 1 or 1.0 - failure ** (n - 1) < 0.999

    def test_default_max_restarts_is_one_when_certain(self) -> None:
        # Huge replicate count makes failure negligible.
        n = default_max_restarts(
            birth_rate=0.9,
            death_rate=1.0,
            level_times=[0.0, 0.1],
            replicates=[1000],
        )
        assert n == 1

    def test_cap_drops_are_recorded(self) -> None:
        # A tiny cap must bind; a huge one must not.
        def run(cap: int):
            return simulate_survival_splitting(
                SplittingConfig(
                    birth_rate=0.9,
                    death_rate=1.0,
                    terminal_time=40.0,
                    n_particles=cap,
                    random_seed=17,
                ),
            )

        capped, free = run(5), run(1_000_000)
        assert free.n_dropped_by_cap == 0
        assert free.n_capped_levels == 0
        assert capped.n_dropped_by_cap > 0
        assert capped.n_capped_levels > 0
        last_attempt = sum(d.n_dropped_by_cap for d in capped.level_diagnostics)
        assert 0 < last_attempt <= capped.n_dropped_by_cap
        assert capped.level_diagnostics[-1].n_dropped_by_cap == 0

    def test_max_restarts_caps_attempts(self) -> None:
        config = SplittingConfig(
            birth_rate=0.05,
            death_rate=1.0,
            terminal_time=50.0,
            n_particles=50,
            random_seed=0,
            max_restarts=3,
        )
        with pytest.warns(RuntimeWarning):
            result = simulate_survival_splitting(config)
        assert result.n_attempts <= 3
        assert result.max_restarts == 3

    def test_invalid_max_restarts_raises(self) -> None:
        config = SplittingConfig(
            birth_rate=0.9,
            death_rate=1.0,
            terminal_time=5.0,
            n_particles=10,
            max_restarts=0,
        )
        with pytest.raises(ValueError, match="max_restarts"):
            simulate_survival_splitting(config)

    def test_restarts_happen_and_succeed(self) -> None:
        # With R_1 small attempts often fail; some seed needs a restart.
        attempts = []
        for seed in range(8):
            config = SplittingConfig(
                birth_rate=0.9,
                death_rate=1.0,
                terminal_time=30.0,
                n_particles=200,
                random_seed=seed,
            )
            result = simulate_survival_splitting(config)
            attempts.append(result.n_attempts)
            assert result.n_attempts <= result.max_restarts
            if result.n_attempts < result.max_restarts or result.final_survivor_count:
                assert result.final_survivor_count > 0
        assert max(attempts) > 1


class TestOverrides:
    """Explicit level_times and replicates."""

    def test_level_times_and_replicates_used_as_given(self) -> None:
        config = SplittingConfig(
            birth_rate=1.0,
            death_rate=1.1,
            terminal_time=20.0,
            n_particles=200,
            level_times=[0.0, 20.0],
            replicates=[2],
            random_seed=15,
            max_restarts=100,
        )
        with pytest.warns(RuntimeWarning, match="No particle survived"):
            result = simulate_survival_splitting(config)
        assert result.replicates.tolist() == [2]
        assert result.level_times.tolist() == [0.0, 20.0]
        assert len(result.conditioned_trajectories) == result.final_survivor_count

    def test_replicates_length_mismatch_raises(self) -> None:
        config = SplittingConfig(
            birth_rate=0.9,
            death_rate=1.0,
            terminal_time=10.0,
            n_particles=50,
            level_times=[0.0, 5.0, 10.0],
            replicates=[2],
        )
        with pytest.raises(ValueError, match="one entry per level"):
            simulate_survival_splitting(config)

    def test_replicates_without_level_times_raises(self) -> None:
        config = SplittingConfig(
            birth_rate=0.9,
            death_rate=1.0,
            terminal_time=10.0,
            n_particles=50,
            replicates=[2],
        )
        with pytest.raises(ValueError, match="requires level_times"):
            simulate_survival_splitting(config)


class TestInitialLevelCount:
    """Starting number of levels from the terminal survival probability."""

    @pytest.mark.parametrize(
        ("pr_terminal", "expected"),
        [(1e-9, 19), (0.2, 1), (0.5, 1), (1.0, 1), (3.0**-7, 7)],
    )
    def test_count_matches_one_third_rule(
        self, pr_terminal: float, expected: int
    ) -> None:
        assert _initial_level_count(float(np.log(pr_terminal))) == expected

    def test_default_equals_explicit_rule_value(self) -> None:
        k = _initial_level_count(
            log_survival_probability_constant(birth_rate=1.0, death_rate=2.0, time=20.0)
        )
        default = select_level_times_for_band(
            birth_rate=1.0, death_rate=2.0, terminal_time=20.0
        )
        explicit = select_level_times_for_band(
            birth_rate=1.0, death_rate=2.0, terminal_time=20.0, n_levels=k
        )
        np.testing.assert_array_equal(default, explicit)


class TestAnyRates:
    """Supercritical, critical and easy (direct-simulation) cases."""

    def test_supercritical_splits_with_few_replicates(self) -> None:
        config = SplittingConfig(
            birth_rate=1.0,
            death_rate=0.8,
            terminal_time=10.0,
            n_particles=200,
            random_seed=3,
        )
        result = simulate_survival_splitting(config)
        assert result.replicates[0] >= 2
        assert result.max_restarts is not None
        assert len(result.conditioned_trajectories) > 0
        for trajectory in result.conditioned_trajectories:
            assert trajectory.populations[-1] > 0

    def test_critical_runs(self) -> None:
        config = SplittingConfig(
            birth_rate=1.0,
            death_rate=1.0,
            terminal_time=10.0,
            n_particles=200,
            random_seed=4,
        )
        result = simulate_survival_splitting(config)
        assert len(result.conditioned_trajectories) > 0

    def test_easy_case_is_simulated_directly(self) -> None:
        # death / birth < 0.5 so Pr(T) > 0.5 for every T.
        config = SplittingConfig(
            birth_rate=1.0,
            death_rate=0.4,
            terminal_time=15.0,
            n_particles=200,
            random_seed=5,
        )
        result = simulate_survival_splitting(config)
        assert result.replicates.tolist() == [1]
        assert result.level_times.tolist() == [0.0, 15.0]
        assert result.initial_particle_count == 1
        assert result.max_restarts is None
        assert len(result.conditioned_trajectories) == 1
        assert result.conditioned_trajectories[0].populations[-1] > 0

    def test_direct_case_respects_explicit_max_restarts(self) -> None:
        config = SplittingConfig(
            birth_rate=1.0,
            death_rate=0.4,
            terminal_time=15.0,
            n_particles=50,
            random_seed=6,
            max_restarts=1000,
        )
        result = simulate_survival_splitting(config)
        assert result.max_restarts == 1000
        assert result.n_attempts <= 1000

    def test_negative_rates_rejected(self) -> None:
        config = SplittingConfig(
            birth_rate=-0.1, death_rate=1.0, terminal_time=5.0, n_particles=10
        )
        with pytest.raises(ValueError, match="non-negative"):
            simulate_survival_splitting(config)

    def test_zero_float_birth_rate_is_accepted(self) -> None:
        config = SplittingConfig(
            birth_rate=0.0,
            death_rate=0.5,
            terminal_time=3.0,
            n_particles=100,
            random_seed=1,
            max_restarts=200,
        )
        result = simulate_survival_splitting(config)
        assert result.final_survivor_count > 0
        for trajectory in result.conditioned_trajectories:
            assert len(trajectory.process.records) == 1


def _rate_like_pair(horizon: float) -> tuple[PiecewiseConstant, PiecewiseLinear]:
    return (
        PiecewiseConstant([1.0], [horizon]),
        PiecewiseLinear([0.0], [1.6], [horizon]),
    )


class TestRateInputs:
    """Floats and rate-likes through the one set of entry points."""

    def test_terminal_time_is_required_for_float_rates(self) -> None:
        config = SplittingConfig(birth_rate=1.0, death_rate=2.0, n_particles=10)
        with pytest.raises(ValueError, match="terminal_time is required"):
            simulate_survival_splitting(config)
        with pytest.raises(ValueError, match="terminal_time is required"):
            select_level_times_for_band(birth_rate=1.0, death_rate=2.0)

    def test_terminal_time_must_match_rate_likes(self) -> None:
        birth, death = _rate_like_pair(4.0)
        config = SplittingConfig(
            birth_rate=birth, death_rate=death, terminal_time=5.0, n_particles=10
        )
        with pytest.raises(ValueError, match="terminal_time must equal"):
            simulate_survival_splitting(config)

    def test_rate_likes_without_terminal_time(self) -> None:
        birth, death = _rate_like_pair(4.0)
        levels = select_level_times_for_band(birth_rate=birth, death_rate=death)
        assert levels[0] == 0.0
        assert levels[-1] == pytest.approx(4.0)

    def test_trees_store_rate_objects(self) -> None:
        config = SplittingConfig(
            birth_rate=1.0,
            death_rate=2.0,
            terminal_time=4.0,
            n_particles=100,
            random_seed=1,
            max_restarts=200,
        )
        tree = simulate_survival_splitting(config).conditioned_trajectories[0].process
        assert tree.birth_rate == PiecewiseConstant([1.0], [4.0])
        assert tree.death_rate == PiecewiseConstant([2.0], [4.0])
        assert tree.terminal_time == 4.0

    def test_constant_rate_like_matches_float_helpers(self) -> None:
        horizon = 6.0
        floats = {"birth_rate": 1.0, "death_rate": 2.0}
        likes = {
            "birth_rate": PiecewiseConstant([1.0], [horizon]),
            "death_rate": PiecewiseConstant([2.0], [horizon]),
        }
        np.testing.assert_allclose(
            select_level_times_for_band(**floats, terminal_time=horizon),
            select_level_times_for_band(**likes),
            rtol=1e-9,
        )
        levels = [0.0, 2.0, 4.0, horizon]
        np.testing.assert_array_equal(
            compute_level_replicates(**floats, level_times=levels),
            compute_level_replicates(**likes, level_times=levels),
        )
        replicates = [3, 3, 3]
        assert default_max_restarts(
            **floats, level_times=levels, replicates=replicates
        ) == default_max_restarts(**likes, level_times=levels, replicates=replicates)

    def test_float_horizon_defaults_to_the_last_level_time(self) -> None:
        levels = [0.0, 2.0, 5.0]
        np.testing.assert_array_equal(
            compute_level_replicates(
                birth_rate=1.0, death_rate=2.0, level_times=levels
            ),
            compute_level_replicates(
                birth_rate=1.0, death_rate=2.0, level_times=levels, terminal_time=5.0
            ),
        )

    def test_linear_rate_objects_in_the_helpers(self) -> None:
        birth, death = _rate_like_pair(6.0)
        levels = select_level_times_for_band(birth_rate=birth, death_rate=death)
        replicates = compute_level_replicates(
            birth_rate=birth, death_rate=death, level_times=levels
        )
        assert replicates.shape == (len(levels) - 1,)
        assert np.all(replicates >= 2)
        assert (
            default_max_restarts(
                birth_rate=birth,
                death_rate=death,
                level_times=levels,
                replicates=replicates,
            )
            >= 1
        )

    def test_mixed_float_and_rate_like(self) -> None:
        death = PiecewiseLinear([0.0], [1.6], [6.0])
        levels = select_level_times_for_band(birth_rate=1.0, death_rate=death)
        assert levels[-1] == pytest.approx(6.0)


class TestPartialGrids:
    """The helpers accept a grid that stops short of T, not one beyond it."""

    @pytest.mark.parametrize("last", [4.0, 3.0])
    def test_grid_ending_at_or_before_t_is_accepted(self, last: float) -> None:
        birth, death = PiecewiseConstant([1.0], [4.0]), PiecewiseConstant([1.5], [4.0])
        levels = [0.0, 2.0, last]
        replicates = compute_level_replicates(
            birth_rate=birth, death_rate=death, level_times=levels
        )
        assert replicates.shape == (2,)
        restarts = default_max_restarts(
            birth_rate=birth,
            death_rate=death,
            level_times=levels,
            replicates=replicates,
        )
        assert restarts >= 1

    def test_grid_beyond_t_is_rejected(self) -> None:
        birth, death = PiecewiseConstant([1.0], [5.0]), PiecewiseConstant([1.5], [5.0])
        with pytest.raises(ValueError, match="level_times"):
            compute_level_replicates(
                birth_rate=birth, death_rate=death, level_times=[0, 2.5, 9.0]
            )


class TestKeywordOnlyConfig:
    def test_positional_construction_raises(self) -> None:
        with pytest.raises(TypeError):
            SplittingConfig(1.0, 2.0, 8.0, 200)  # type: ignore[misc]
        with pytest.raises(TypeError):
            SplittingConfig(1.0, 2.0, 200)  # type: ignore[misc]
