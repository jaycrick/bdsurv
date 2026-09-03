# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Initial extraction from `bdpinf`: `simulate_birth_death_process_from_contour_process`
  (exact contour-process reconstruction) and `simulate_subcritical_survival_splitting`
  (fixed-factor RESTART multilevel splitting), both returning the new pure
  `numpy`/`pandas` `SimulatedTree` result type.
- Closed-form single-epoch extinction probability (`bdsurv._extinction.pr_extinction`),
  replacing the JAX-based multi-epoch kernel used internally by `bdpinf`.
