"""Tests for splitting-based conditioned survival simulation."""

import numpy as np
import pytest

from bdsurv._extinction import pr_extinction
from bdsurv.splitting import (
    _BAND_LOWER,
    SplittingConfig,
    _log_survival_probability,
    compute_level_replicates,
    default_max_restarts,
    select_level_times_for_band,
    simulate_subcritical_survival_splitting,
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

        result = simulate_subcritical_survival_splitting(config)

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

        first = simulate_subcritical_survival_splitting(config)
        second = simulate_subcritical_survival_splitting(config)

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

        result = simulate_subcritical_survival_splitting(config)

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
            result = simulate_subcritical_survival_splitting(config)

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

        result = simulate_subcritical_survival_splitting(config)

        assert len(result.conditioned_trajectories) > 0, "expected non-zero survivors"
        for traj in result.conditioned_trajectories:
            records = traj.process.records
            spine_rows = records[records["is_spine"]]
            assert len(spine_rows) == 1, "expected exactly one spine per tree"
            assert bool(spine_rows["survived"].iloc[0]), "spine must be a survivor"


def _ratios(birth_rate: float, death_rate: float, levels: np.ndarray) -> np.ndarray:
    log_prs = [
        _log_survival_probability(birth_rate=birth_rate, death_rate=death_rate, time=t)
        for t in levels
    ]
    return np.exp(np.diff(log_prs))


class TestLevelGrid:
    """Adaptive level-grid selection."""

    def test_log_survival_matches_pr_extinction(self) -> None:
        for t in (0.1, 1.0, 20.0):
            expected = 1.0 - pr_extinction(0.9, 1.0, t)
            got = np.exp(
                _log_survival_probability(birth_rate=0.9, death_rate=1.0, time=t)
            )
            assert np.isclose(got, expected, rtol=1e-9)

    def test_log_survival_stable_for_huge_horizon(self) -> None:
        value = _log_survival_probability(birth_rate=1.0, death_rate=2.0, time=1e6)
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
                _log_survival_probability(birth_rate=1.0, death_rate=2.0, time=mid)
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
                _log_survival_probability(
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
            _log_survival_probability(birth_rate=0.9, death_rate=1.0, time=t)
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
            result = simulate_subcritical_survival_splitting(config)
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
            simulate_subcritical_survival_splitting(config)

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
            result = simulate_subcritical_survival_splitting(config)
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
            random_seed=5,
            max_restarts=100,
        )
        result = simulate_subcritical_survival_splitting(config)
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
            simulate_subcritical_survival_splitting(config)

    def test_replicates_without_level_times_raises(self) -> None:
        config = SplittingConfig(
            birth_rate=0.9,
            death_rate=1.0,
            terminal_time=10.0,
            n_particles=50,
            replicates=[2],
        )
        with pytest.raises(ValueError, match="requires level_times"):
            simulate_subcritical_survival_splitting(config)
