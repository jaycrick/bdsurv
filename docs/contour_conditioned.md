# Contour processes conditioned on the population at $T$

## Overview

`src/bdsurv/contour_conditioned.py` has two exact simulators:

- `simulate_contour_given_population` draws a birth-death tree conditioned on
  exactly $n$ individuals alive at $T$;
- `simulate_contour_given_ages` draws a tree conditioned on the survivors at $T$
  having the given ages.

Rates are floats (then `terminal_time` is required) or piecewise constant /
linear rate-likes on $[0, T]$.
Write $\Lambda(t) = \int_0^t \lambda$, $M(t) = \int_0^t \mu$,
$\Gamma = \Lambda - M$ and $I(t) = \int_0^t \mu(s) e^{-\Gamma(s)}\,ds$.

## Pieces

The contour starts and ends at 0.
Peaks (odd positions) are death times, valleys (even positions) are birth times,
and a peak at $T$ is an individual alive at $T$.
With $n$ individuals alive the contour visits $T$ $n$ times, so it splits into
independent pieces:

- one first-passage piece $0 \to T$;
- $n - 1$ excursions $T \to T$ that do not touch 0 or $T$ in between;
- one piece $T \to 0$.

## Steps

- A step down from $x$ uses the **birth** rate:
  it goes to $s < x$ with $\int_s^x \lambda = E$, $E \sim \mathrm{Exp}(1)$,
  and is snapped to exactly 0 when close.
- A step up from $y$ uses the **death** rate:
  it goes to $s > y$ with $\int_y^s \mu = E$, snapped to exactly $T$ when close.
- A step never returns its start level (zero-length steps are redrawn).
- A run alternates steps until a downward step reaches 0 or an upward step reaches $T$.

A top excursion starts at $T$ with a step down and ends at $T$ (`TT`) or at 0 (`T0`).
A bottom excursion starts at 0 with a step up and ends at $T$ (`0T`) or at 0 (rejected).

## Direction rule

The end pieces can be generated from the $T$ end or from the 0 end,
whichever is cheaper to accept.
A top excursion ends at 0 with probability $S_\mathrm{rev}$,
the survival probability of the process with rates $(\mu(T - t), \lambda(T - t))$;
a bottom excursion ends at $T$ with probability $S_\mathrm{fwd}$,
the survival probability of the process itself.
They satisfy
$$
S_\mathrm{rev} = S_\mathrm{fwd}\, e^{-\Gamma(T)},
$$
so `direction="auto"` generates from the top when $\Gamma(T) < 0$ and from the bottom otherwise.
`"top"` and `"bottom"` force the choice; all three give the same distribution.

## ALG1: $n$ alive at $T$

1. Draw top excursions until $n - 1$ are `TT`.
   Keep the first two `T0` seen on the way.
2. If fewer than two `T0` are kept, complete the two end pieces:
   from the top, by drawing top excursions and keeping `T0` (`TT` discarded);
   from the bottom, by drawing bottom excursions and keeping `0T`.
3. Orient one end piece $0 \to T$ and the other $T \to 0$, and join
   $\text{first} + \text{TT excursions} + \text{last}$.

The first $k$ excursions of a given type in an i.i.d. sequence are i.i.d.
from that type's law, so step 1 is exact.

## ALG2: given ages

The decoder gives a survivor the age $T$ minus the valley just before its peak at $T$.
So each non-spine survivor of age $a$ contributes an excursion
ending $(\dots, b, T)$ with $b = T - a$, obtained as a run upward from $b$
conditioned to reach $T$ before 0 and then reversed.
The spine survivor ends the $0 \to T$ piece.

- **Spine choice.**
  Unless `spine_index` is given, the spine is chosen with probability proportional to
  $$
  w_i = \frac{I(T) - I(b_i)}{1 + I(b_i)}, \qquad b_i = T - a_i,
  $$
  the ratio of the spine and non-spine age densities.
  An age equal to $T$ ($b = 0$) forces the spine, and the piece is $[0, T]$.
- **Spine piece**, valley $b > 0$:
  - from the top (direct start): run upward from $(T, b)$ until it reaches 0, and reverse;
  - from the bottom (cut): draw $0 \to T$ pieces until one has a down-crossing of level $b$,
    cut at the **last** down-crossing and end the piece with $(\dots, b, T)$.
- **Last piece**: an unconditioned $T \to 0$ piece, from the top or the bottom as above.
- The non-spine excursions are placed in a uniformly random order between the two.

## Why it is exact

- A piece has density $\prod \mu(\text{peaks}) \prod \lambda(\text{valleys})\,
  \exp(-\int_\text{down} \lambda - \int_\text{up} \mu)$.
  Generating a piece from the other end swaps the roles of the legs and
  changes the density by $e^{\pm\Gamma}$ over the net span,
  a constant for fixed endpoints, so reversed pieces have the right conditional law.
- At a down-crossing of $b$ the path regenerates, so
  $P(\text{prefix}, \text{this crossing is the last}, \text{path ends at } T)
  = P(\text{prefix})\,P(\text{leg crosses } b)\cdot\text{const}$,
  which makes the cut at the last crossing exact.
- With $u(b) = (1 + I(b)) / (1 + I(T))$, the probability that a run from valley $b$ reaches $T$ before 0:
  the non-spine age density is $\propto \lambda(b)\, e^{-(\Lambda(T) - \Lambda(b))}\, u(b)$,
  and the spine's is $\propto \lambda(b)\, e^{-(\Lambda(T) - \Lambda(b))}\,(1 - u(b))$
  with an atom at $a = T$ of unnormalised mass $e^{-\Lambda(T)}$.

## Usage

```python
from bdsurv import (
    PiecewiseConstant,
    PiecewiseLinear,
    simulate_contour_given_ages,
    simulate_contour_given_population,
)

# Exactly three individuals alive at T = 4, constant rates.
tree = simulate_contour_given_population(
    1.2, 0.7, n_alive=3, terminal_time=4.0, random_seed=0
)

# Survivors of ages 1.0 and 2.5 at T, with linear rates (Arley process).
birth = PiecewiseConstant([1.0], [4.0])
death = PiecewiseLinear([0.0], [3.2], [4.0])
tree = simulate_contour_given_ages(birth, death, ages=[1.0, 2.5], random_seed=0)
```

Both return a `SimulatedTree`, with NaN scalar rates and `*_rate_function` set for rate-like inputs.

## Tests

`tests/test_contour_conditioned.py`:

- structure: pieces joined as in the worked example, contour validity, tree validity,
  seeding, steps never returning their start, input errors;
- ages: decoded survivor ages equal the input, the spine row has the chosen age,
  the last-down-crossing cut, spine weights against quadrature;
- statistical (`-m slow`, fixed seeds, level $10^{-3}$):
  - the $S_\mathrm{rev}$ identity against explicitly reversed rates;
  - ALG1 against forward simulation with rejection on $Z_T = n$
    (KS on total individuals, pooled survivor ages, oldest age);
  - `"top"` against `"bottom"` for both algorithms;
  - composition: ages of an ALG1 tree fed to ALG2 reproduce ALG1's total size;
  - spine-choice frequency against the normalised weights;
  - the ALG1 spine age against the spine law above.
