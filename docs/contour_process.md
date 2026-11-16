# Contour process simulation

## Overview

This document describes the contour-process based simulator in
`src/bdsurv/contour.py`.
Lambert (2010) is the standard reference for contour processes of splitting
trees (see [References](#references)).

## Concepts

- Excursion: a run started at the terminal time `T` that alternates between
  downward jumps and upward jumps.
  A downward jump is driven by the birth rate and lands on a birth time (a valley);
  an upward jump is driven by the death rate and lands on a death time (a peak).
  The run ends when a downward jump reaches 0 (`T0`) or an upward jump reaches `T` (`TT`).

- Spine component: the distinguished lineage.
  It is the first excursion from `T` that ends at 0.
  An excursion that returns to `T` first is rejected and a new one starts afresh
  from `T` (the spine restart, counted as `spine_resets`).

- Further excursions from `T`: each one ending at `T` is another individual alive at `T`
  and is kept; the first one ending at 0 is the last piece.

- Contour process list: the concatenation

  reverse(spine_component) + the `TT` excursions without their first point
  + last piece without its first point

  (`join_pieces`).
  The reversed spine ensures the root (0) appears first, then up-and-down
  alternations consistent with the stack-based reconstruction.

## Construction algorithm

1. Draw the spine component and the further excursions with `_survival_contour`,
   which steps `Walker.top_excursion` (the one stepper, in `contour.py`).
2. Join the pieces into the contour list with `join_pieces`.
3. Validate with `validate_contour_process_list` (odd length, start/end 0,
   strict alternation of peaks/valleys); the decoder calls it.
4. Reconstruct a canonical `SimulatedTree` with `contour_process_list_to_tree`.

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

- Rates (`birth_rate`, `death_rate`) are finite and non-negative (floats or rate
  objects), and the death rate is not identically zero.
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
  > true of the implementation — see `contour_process_list_to_tree` in
  > `src/bdsurv/contour.py`, which has always marked the last-born
  > individual. This note exists so the discrepancy is visible before it is
  > written into the paper; whichever convention is intended, the code,
  > this document, and the comparison notebook's caveat text should agree.

## Linear (time-varying) rates

`simulate_contour_given_survival` accepts floats or piecewise-constant or
piecewise-linear rates $\lambda(t)$, $\mu(t)$ on $[0, T]$ (a float is constant
on $[0, T]$).
The construction and reconstruction are unchanged; only the jumps differ, and
they now depend on the absolute time:

- a downward jump from level $x$ goes to the $s < x$ with
  $\int_s^x \lambda = E$ (the previous birth time), floored at 0;
- an upward jump from $y$ goes to the $s > y$ with $\int_y^s \mu = E$ (a death
  time), capped at $T$.

Here $E \sim \mathrm{Exp}(1)$.
Within a linear segment, with $a$ the rate at $t$ and $b$ its slope,
$\int_0^{s}(a + b u)\,du = a s + b s^2/2 = E$ has the
stable root $s = 2E / (a + \sqrt{a^2 + 2 b E})$ (here $s$ is the offset from $t$); when the segment's hazard is
exhausted before $E$ is reached, the remainder carries into the neighbouring
segment (`forward_hazard_time` / `backward_hazard_time`).
The spine restart on reaching $T$ is the same.
A constant rate gives an exponential jump, $E / \text{rate}$.
The death rate must not vanish identically, or the spine would restart forever.

The result is a `SimulatedTree` whose `birth_rate` and `death_rate` hold the
rate objects (a float rate as a one-epoch `PiecewiseConstant`).
`tests/test_linear_conditioned.py` checks the population at $T$ against
rejection-sampled forward simulation.

## Reproducibility and RNG

- All draws use `numpy.random.Generator` objects.
- The public entrypoint accepts a `random_seed` and builds its one generator
  with `bdsurv._seeding.root_rng`, so a seed reproduces the tree.

## Usage

Call `simulate_contour_given_survival(birth_rate, death_rate, terminal_time,
random_seed)` from the `bdsurv` package.
See the function docstring and
`src/bdsurv/contour.py` for parameter descriptions and examples.

## References

- Lambert (2010): the standard reference for contour processes of splitting trees.
  TODO(author): locator

The full entries are in [`docs/references.md`](references.md).
