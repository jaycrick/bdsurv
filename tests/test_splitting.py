"""Tests for splitting-based conditioned survival simulation."""

import numpy as np

from bdsurv.splitting import (
    _BAND_LOWER,
    _BAND_UPPER,
    _MAX_BAND_LEVELS,
    SplittingConfig,
    _adaptive_particle_count,
    _calculate_extinction_probability,
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
        assert result.level_survival_probabilities.shape[0] == n_levels_used
        assert result.level_mean_populations.shape[0] == n_levels_used + 1
        assert 0.0 <= result.survival_probability_estimate <= 1.0

    def test_splitting_product_matches_stored_estimate(self) -> None:
        config = SplittingConfig(
            birth_rate=0.9,
            death_rate=1.0,
            terminal_time=30.0,
            n_particles=150,
            n_levels=6,
            random_seed=91,
        )

        result = simulate_subcritical_survival_splitting(config)
        reconstructed = float(np.prod(result.level_survival_probabilities))
        assert np.isclose(
            reconstructed,
            result.survival_probability_estimate,
            atol=1e-12,
            rtol=1e-12,
        )

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

        assert np.isclose(
            first.survival_probability_estimate,
            second.survival_probability_estimate,
        )
        assert np.allclose(
            first.level_survival_probabilities,
            second.level_survival_probabilities,
        )

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

    def test_extinction_helper_uses_requested_band(self) -> None:
        probability = _calculate_extinction_probability(
            birth_rate=0.9,
            death_rate=1.0,
            interval_time=0.5,
        )

        assert np.e ** (-2.0) < probability < 0.5
        assert _adaptive_particle_count(
            birth_rate=0.9,
            death_rate=1.0,
            interval_time=0.5,
        ) == int(np.ceil(1.0 / probability))

    def test_splitting_level_times_invariant_on_early_extinction(self) -> None:
        # Regression test: when all particles go extinct before the final level
        # the loop breaks early and level_times was previously not truncated,
        # leaving len(level_times) >> len(level_survival_probabilities) + 1.
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

        result = simulate_subcritical_survival_splitting(config)

        # Core invariant: always one more boundary than probability estimates.
        assert len(result.level_times) == len(result.level_survival_probabilities) + 1
        assert len(result.level_times) == len(result.level_mean_populations)
        # Confirm the early-exit path was taken: all particles went extinct.
        assert result.survival_probability_estimate == 0.0
        assert result.final_survivor_count == 0

    def test_splitting_band_and_initial_count(self) -> None:
        # Verify the enrichment scheme: first-increment pr_ext in band, initial
        # ensemble == min(R, cap), and every level's entering_count <= cap.
        config = SplittingConfig(
            birth_rate=0.9,
            death_rate=1.0,
            terminal_time=50.0,
            n_particles=500,
            random_seed=42,
        )

        result = simulate_subcritical_survival_splitting(config)

        # Derive what the first increment should be from the auto-selected grid
        first_increment = float(result.level_times[1]) - float(result.level_times[0])
        pr_ext = _calculate_extinction_probability(
            birth_rate=config.birth_rate,
            death_rate=config.death_rate,
            interval_time=first_increment,
        )

        # pr_ext must be in band [e^{-2}, 0.5]
        assert np.e**-2.0 <= pr_ext <= 0.5, (
            f"First-increment pr_ext {pr_ext:.4f} is outside band"
        )

        # initial_particle_count == min(ceil(1/pr_ext), n_particles)
        expected_r = int(np.ceil(1.0 / pr_ext))
        expected_initial = min(expected_r, config.n_particles)
        assert result.splitting_factor == expected_r
        assert result.initial_particle_count == expected_initial

        # Every LevelDiagnostic.entering_count must be <= n_particles (cap respected)
        for diag in result.level_diagnostics:
            assert diag.entering_count is not None
            assert diag.entering_count <= config.n_particles, (
                f"Level {diag.level_index}: entering_count {diag.entering_count} "
                f"exceeds cap {config.n_particles}"
            )

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

    def test_splitting_selects_balanced_split(self) -> None:
        # Verify that auto-selection chooses the split count whose R*pr_ext is
        # closest to 1, not merely the coarsest in-band grid.
        birth_rate = 0.9
        death_rate = 1.0
        terminal_time = 50.0

        config = SplittingConfig(
            birth_rate=birth_rate,
            death_rate=death_rate,
            terminal_time=terminal_time,
            n_particles=500,
            random_seed=42,
        )
        result = simulate_subcritical_survival_splitting(config)
        n_selected = len(result.level_times) - 1

        # Independently compute the best n by brute force
        best_score = np.inf
        best_n = 0
        n = 1
        while n <= _MAX_BAND_LEVELS:
            pr = _calculate_extinction_probability(
                birth_rate=birth_rate,
                death_rate=death_rate,
                interval_time=terminal_time / n,
            )
            if pr < _BAND_LOWER:
                break
            if pr <= _BAND_UPPER:
                r = int(np.ceil(1.0 / pr))
                score = abs((r * pr) ** n - 1.0)
                if score < best_score:
                    best_score = score
                    best_n = n
            n += 1

        assert best_n > 0, "reference search found no valid split (test setup error)"
        assert n_selected == best_n, (
            f"simulator chose n={n_selected} but brute-force optimum is n={best_n}"
        )

    def test_select_level_times_for_band_is_valid_grid(self) -> None:
        # The public selector must return a strictly increasing grid starting at
        # 0 and ending at terminal_time, with all increments in the band.
        birth_rate = 1.0
        death_rate = 2.0
        terminal_time = 20.0

        levels = select_level_times_for_band(
            birth_rate=birth_rate,
            death_rate=death_rate,
            terminal_time=terminal_time,
        )

        assert float(levels[0]) == 0.0
        assert np.isclose(float(levels[-1]), terminal_time)
        assert len(levels) >= 2
        # strictly increasing
        assert np.all(np.diff(levels) > 0)
        # first increment pr_ext in band
        first_increment = float(levels[1]) - float(levels[0])
        pr = _calculate_extinction_probability(
            birth_rate=birth_rate,
            death_rate=death_rate,
            interval_time=first_increment,
        )
        assert _BAND_LOWER <= pr <= _BAND_UPPER

    def test_splitting_factor_override_bypasses_band_check(self) -> None:
        # When both level_times and splitting_factor are supplied, the band
        # validation is skipped even if the first increment is out of band.
        # Use a very coarse grid (one level) whose pr_ext is far above _BAND_UPPER,
        # combined with a hand-picked R=2 — this would normally raise ValueError.
        birth_rate = 1.0
        death_rate = 1.1
        terminal_time = 20.0

        # Single level — pr_ext for the whole horizon is above 0.5 for mild subcritical
        config = SplittingConfig(
            birth_rate=birth_rate,
            death_rate=death_rate,
            terminal_time=terminal_time,
            n_particles=200,
            level_times=[0.0, terminal_time],
            splitting_factor=2,
            random_seed=5,
        )

        result = simulate_subcritical_survival_splitting(config)

        # Override was respected
        assert result.splitting_factor == 2
        assert 0.0 <= result.survival_probability_estimate <= 1.0
