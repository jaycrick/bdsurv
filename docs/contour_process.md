# Contour process simulation

## Overview

This document describes the contour-process based simulator in
`src/bdsurv/contour.py`.

## Concepts

- Spine component: the distinguished lineage.
  Built as an excursion started
  at the terminal time `T` that alternates between downward (death) and
  upward (birth) moves.
  If an upward jump would hit the terminal time `T`,
  the spine generation restarts from `T` (reset rule).

- Non-spine component: a single excursion started at `T` that alternates
  death/birth moves and terminates when reaching 0.

- Contour process list: the concatenation

  reverse(spine_component) + non_spine_component[1:]

  reversed spine ensures the root (0) appears first, then up-and-down
  alternations consistent with the stack-based reconstruction.

## Construction algorithm

1. Produce a spine component using `_generate_spine_component`.
2. Produce a non-spine component using `_generate_non_spine_component`.
3. Build the contour list with `_build_contour_process_list`.
4. Validate with `_validate_contour_process_list` (odd length, start/end 0,
   strict alternation of peaks/valleys).
5. Reconstruct a canonical `SimulatedTree` with
   `_contour_process_list_to_tree`.

## Reconstruction (stack algorithm)

- The contour is interpreted as alternating death and birth times,
  starting and ending at 0.
- Use two stacks: one for birth times and one for individual numbers.
- Individual 0 (root) is created with creation time 0.
- On reading a birth time, pop the birth-time stack until the top birth time
  is strictly less than the new birth time; the individual number on the
  top of the individual stack becomes the parent of the new individual.
- This reproduces the usual genealogical reconstruction from a contour.

## Assumptions and invariants

- Rates (`birth_rate`, `death_rate`) are positive.
- Terminal time `T` is positive.
- Generated contour lists must be strictly alternating and start/end with 0.
- The public API returns a `SimulatedTree` (a plain `records` DataFrame plus
  the rates and terminal time used to generate it).

## Spine / survival semantics

- Records whose recorded death time equals the terminal time `T` are
  treated as still alive at observation and their `death_time` is stored as
  `NaN` in the canonical `records` table.
  This ensures the
  `survived` boolean and `age` fields reflect survival to `T`.
- Exactly one individual is marked `is_spine = True`: the individual at the
  top of the genealogy stack the first time the reconstruction reaches `T`
  — equivalently, the **last-born** individual alive at `T`.

  > **Note (as of this extraction):** an earlier draft of this document, and
  > the comparison notebook in `bdpinf`, both described the spine as
  > the *earliest-created* surviving individual instead. That was never
  > true of the implementation — see `_contour_process_list_to_tree` in
  > `src/bdsurv/contour.py`, which has always marked the last-born
  > individual. This note exists so the discrepancy is visible before it is
  > written into the paper; whichever convention is intended, the code,
  > this document, and the comparison notebook's caveat text should agree.

## Linear (time-varying) rates

`simulate_linear_rate_contour_process` accepts piecewise-constant or
piecewise-linear rates $\lambda(t)$, $\mu(t)$ on $[0, T]$.
The construction and reconstruction are unchanged; only the jumps differ, and
they now depend on the absolute time:

- a downward jump from level $x$ goes to the $s < x$ with
  $\int_s^x \lambda = E$ (the previous birth time), floored at 0;
- an upward jump from $y$ goes to the $s > y$ with $\int_y^s \mu = E$ (a death
  time), capped at $T$.

Here $E \sim \mathrm{Exp}(1)$.
Within a linear segment $\int_t^{t+s}(a + b u)\,du = a s + b s^2/2 = E$ has the
stable root $s = 2E / (a + \sqrt{a^2 + 2 b E})$; when the segment's hazard is
exhausted before $E$ is reached, the remainder carries into the neighbouring
segment (`forward_hazard_time` / `backward_hazard_time`).
The spine's reset-on-$T$ rule is the same.
Constant rates recover the exponential jumps above.
The death rate must not vanish identically, or the spine would reset forever.

The result is a `SimulatedTree` whose scalar rates are NaN and whose
`birth_rate_function` / `death_rate_function` hold the rates.
`tests/test_linear_conditioned.py` checks the population at $T$ against
rejection-sampled forward simulation.

## Reproducibility and RNG

- All draws use `numpy.random.default_rng` style `Generator` objects.
- The public entrypoint accepts a `random_seed` and derives child
  `Generator`s in the local helper functions to make deterministic,
  isolated draws.

## Usage

Call `simulate_birth_death_process_from_contour_process(...)` from the
`bdsurv` package.
See the function docstring and
`src/bdsurv/contour.py` for parameter descriptions and examples.
