# Hazard families

## Overview

A `Hazard` is a non-negative function $h(t)$ of absolute time $t \ge 0$ with cumulative hazard $H(t) = \int_0^t h$.
Every simulator that accepts a rate (`RateInput`) accepts a `Hazard` as well, and `sample_residual_lifetimes` samples from one directly.

The basic operation is exact left-truncated sampling.
An individual that is at risk from time $t_0$ has residual time $R$ to its next event with

$$
H(t_0 + R) - H(t_0) = E, \qquad E \sim \mathrm{Exp}(1),
$$

so $R$ has hazard $h(t_0 + \cdot)$.
This is the forward inversion, `forward_offsets(t0, e)`.
The backward inversion, `backward_offsets(x, e)`, finds the offset $D \in [0, x]$ with $H(x) - H(x - D) = E$, the time back from $x$ to the previous event.
If $H(x) \le E$ the hazard over $[0, x]$ is exhausted and $D = x$, so the event time is 0.

Which inversion each simulator uses:

- The contour process (`docs/contour_process.md`) inverts the birth rate backward in time for the down-steps and the death rate forward in time for the up-steps.
  Nothing else changes with the family.
- `simulate_birth_death` and `simulate_survival_splitting` use the forward Gillespie scheme.
  For rates that are all piecewise constant or linear it is the original merged-segment kernel.
  If either rate is a `Hazard` it uses competing risks: each of the $n$ living individuals shares one birth hazard and one death hazard, so the time to the next birth is the forward inversion of the birth hazard with target $E_b / n$, the time to the next death likewise with $E_d / n$, and the earlier one is the event.
  If both times are at or after the stopping time the run ends there.
- `log_survival_probability`, `log_mean_cumulative_individuals` and the conditioned contour simulators need integrals of the rates, which are done by quadrature (see "Quadrature").

## The `Hazard` interface

Times are absolute floats at least 0.
The array methods accept array-likes and return `float64` arrays; the scalar methods take and return floats.

| Member | Meaning |
| --- | --- |
| `hazard(t)` | $h(t)$; may be `inf` at a singular point |
| `cumulative_hazard(t)` | $H(t)$, with $H(0) = 0$ |
| `cumulative_between(s, e)` | $H(e) - H(s)$; the default is the difference, and families override it with a form that avoids cancellation |
| `cumulative_limit` | $H(\infty)$; `inf` by default, finite means the hazard is defective |
| `breakpoints` | times where $h$ is not smooth; default `()` |
| `singular_points` | times where $h = \infty$; default `()` |
| `end_time` | the end of the domain; `None` for parametric families, the last epoch end for `PiecewiseHazard` |
| `has_direct_sampler` | `True` if the single-`E` offsets are closed forms (a class variable; a property for `PiecewiseHazard`) |
| `n_draws` | number of $\mathrm{Exp}(1)$ draws per event in the split samplers; 1 except for `SumHazard` |
| `forward_time_split(start, es, cap)`, `backward_time_split(start, es)`, `forward_offsets_split(t0, E)` | the same inversions taking `n_draws` exponentials (`E` has shape `(n_draws, *t0.shape)`); the default uses `es[0]` only; see "Superposition" |
| `check_nonnegative(end_time)` | raises `ValueError` if $h < 0$ somewhere on $[0, \text{end\_time}]$; the default does nothing |
| `forward_offsets(t0, e)` | $R$ with $H(t_0 + R) - H(t_0) = e$; `inf` if defective |
| `backward_offsets(x, e)` | $D \in [0, x]$ with $H(x) - H(x - D) = e$; $x$ if $H(x) \le e$ |
| `forward_time(start, e, cap)` | scalar `min(start + R, cap)`; `cap` also when no event |
| `backward_time(start, e)` | scalar `max(start - D, 0)` |
| `forward_offsets_numerical`, `backward_offsets_numerical` | always Brent, whatever the subclass; the defaults of the offsets and the reference in tests |

## Families

| Family | $h(t)$ | $H(t)$ | Parameters | Sampler | Singular points | Defective |
| --- | --- | --- | --- | --- | --- | --- |
| `PowerHazard(c, p)` | $c\,t^p$ | $(t/\eta)^k$ | $c > 0$, $p > -1$; $k = p + 1$, $\eta = (k/c)^{1/k}$ | direct | $0$ if $p < 0$ | no |
| `PowerHazard.weibull(shape, scale)` | $(k/\eta)(t/\eta)^{k-1}$ | $(t/\eta)^k$ | `shape`, `scale` positive | direct | $0$ if $k < 1$ | no |
| `PowerHazard.rayleigh(c)` | $c\,t$ | $c\,t^2/2$ | $c > 0$ | direct | none | no |
| `GompertzHazard(a, b)` | $a\,e^{bt}$ | $a\,t\,\varphi(bt)$, $\varphi(x) = \mathrm{expm1}(x)/x$ | $a > 0$, $b$ real | direct | none | yes if $b < 0$, $H(\infty) = -a/b$ |
| `LogLogisticHazard(shape, scale)` | $\dfrac{(k/\eta)(t/\eta)^{k-1}}{1 + (t/\eta)^k}$ | $\mathrm{log1p}((t/\eta)^k)$ | `shape` $k$, `scale` $\eta$ positive | direct | $0$ if $k < 1$ | no |
| `LomaxHazard(shape, scale)` | $k/(\eta + t)$ | $k\,\mathrm{log1p}(t/\eta)$ | `shape` $k$, `scale` $\eta$ positive | direct | none | no |
| `QuadraticHazard(a, b, c)` | $a + b t + c t^2$ | $a t + b t^2/2 + c t^3/3$ | $a \ge 0$, $b$, $c$ finite | closed form + Newton (Brent fallback) | none | no |
| `CubicHazard(a, b, c, d)` | $a + b t + c t^2 + d t^3$ | $a t + b t^2/2 + c t^3/3 + d t^4/4$ | $a \ge 0$, $b$, $c$, $d$ finite | closed form + Newton (Brent fallback) | none | no |
| `LogisticHazard(L, k, c)` | $L\,\sigma(k(t - c))$ | $(L/k)\,[\mathrm{softplus}(k(t - c)) - \mathrm{softplus}(-kc)]$ | $L > 0$, $k \ne 0$, $c$ finite | direct | none | yes if $k < 0$, $H(\infty) = (L/\lvert k\rvert)\,\mathrm{softplus}(\lvert k\rvert c)$ |
| `PiecewiseHazard(pieces, epoch_end_times)` | piece $k$ in local time | sum over epochs | one `Hazard` per epoch; ends positive, strictly increasing | direct iff every piece is | the pieces', shifted | no |

Parameters are validated at construction (`ValueError`, messages of the form "`name` must be positive and finite").

## Stable formulas of the direct samplers

Here $t_0$ is the start of a forward draw, $x$ the start of a backward draw, and $e$ the Exp(1) draw.
The forms use `log1p` and `expm1` so that a large $t_0$ (where $H(t_0) + e$ would lose $e$) or a small $e$ stays accurate.
The backward offsets $D$ are computed directly, not as $x - s$, so there is no cancellation when $D \ll x$.
The offset is clipped to $[0, x]$.

`PowerHazard`, with $H_0 = (t_0/\eta)^k$ and $H_x = (x/\eta)^k$:

- forward: $R = t_0\,\mathrm{expm1}(\mathrm{log1p}(e/H_0)/k)$; if $t_0 = 0$ (so $H_0 = 0$), $R = \eta\,e^{1/k}$ written as `eta * e ** (1/k)`;
- backward: $D = -x\,\mathrm{expm1}(\mathrm{log1p}(-e/H_x)/k)$, and $D = x$ if $e \ge H_x$;
- between: $H(s)\,\mathrm{expm1}(k\,\mathrm{log1p}((t - s)/s))$, and $H(t)$ if $s = 0$.

`GompertzHazard`:

- forward: with $z = b\,e\,e^{-b t_0}/a$, $R = \mathrm{log1p}(z)/b$, and $R = \infty$ if $z \le -1$ (the hazard is defective; this also catches an overflow to $-\infty$);
- backward: with $w = -b\,e\,e^{-b x}/a$, $D = -\mathrm{log1p}(w)/b$, and $D = x$ if $w \le -1$;
- $b = 0$ exactly is the constant hazard: $R = e/a$, and $D = \min(e/a, x)$;
- between: $(a/b)\,e^{bs}\,\mathrm{expm1}(b(t - s))$.

`LogLogisticHazard`, with $x_0 = (t_0/\eta)^k$ and $x_x = (x/\eta)^k$:

- forward: $R = t_0\,\mathrm{expm1}(\mathrm{log1p}((1 + x_0)\,\mathrm{expm1}(e)/x_0)/k)$, and $R = \eta\,\mathrm{expm1}(e)^{1/k}$ if $t_0 = 0$;
- backward: with $d = (1 + x_x)(-\mathrm{expm1}(-e))$, $D = -x\,\mathrm{expm1}(\mathrm{log1p}(-d/x_x)/k)$, and $D = x$ if $d \ge x_x$;
- between: $\mathrm{log1p}((x_1 - x_0)/(1 + x_0))$.

`LomaxHazard`:

- forward: $R = (\eta + t_0)\,\mathrm{expm1}(e/k)$;
- backward: $D = \min\big((\eta + x)(-\mathrm{expm1}(-e/k)),\, x\big)$;
- between: $k\,\mathrm{log1p}((t - s)/(\eta + s))$.

`LogisticHazard`, with $\mathrm{softplus}(x) = \log(1 + e^x)$, $\mathrm{sd}(x_0, d) = \mathrm{softplus}(x_0 + d) - \mathrm{softplus}(x_0)$ and $u = kE/L$:

- $\mathrm{sd}(x_0, d) = \mathrm{log1p}(\sigma(x_0)\,\mathrm{expm1}(d))$ with $\sigma(x_0) = e^{-\mathrm{softplus}(-x_0)}$, and for $d > 30$ the log-space form $\mathrm{softplus}(\log\mathrm{expm1}(d) - \mathrm{softplus}(-x_0))$;
- between: $(L/k)\,\mathrm{sd}(k(s - c), k(t - s))$, and $H(t) = (L/k)\,\mathrm{sd}(-kc, kt)$;
- forward, with $x = k(t_0 - c)$: $R = \mathrm{softplus}(\log\mathrm{expm1}(u) + \mathrm{softplus}(-x))/k$, and $R = 0$ if $e = 0$;
- backward, with $z = k(x_t - c)$ and $w = \log\mathrm{expm1}(u) - z$: $D = \min((u - \log(-\mathrm{expm1}(w)))/k,\, x_t)$, and $D = x_t$ if $w \ge 0$ (also $D = 0$ if $e = 0$ or $x_t = 0$).

For $k < 0$ the same `between` and $H$ formulas hold unchanged, $u = kE/L < 0$, and $H(\infty) = -(L/k)\,\mathrm{softplus}(-kc)$ (checked against quadrature).
With $\mathrm{l1m}(v) = \log(-\mathrm{expm1}(v))$ for $v < 0$:

- forward, with $x = k(t_0 - c)$ and $w = -x + \log\mathrm{expm1}(-u)$: $R = (u + \mathrm{l1m}(w))/k$, and $R = \infty$ if $w \ge 0$ (the draw exceeds the defect, $-u \ge \mathrm{softplus}(x)$).
  The shorter form $\log\mathrm{1p}((1 + e^{-x})\mathrm{expm1}(u))/k$ rounds its argument to $-1$ when $\lvert u\rvert$ is large, so $w$ is built from $e^u(1 - e^{-x}\mathrm{expm1}(-u))$ instead;
- backward, with $z = k(x_t - c)$: $D = \min(\mathrm{softplus}(\log\mathrm{expm1}(-u) + \mathrm{softplus}(-z))/\lvert k\rvert,\, x_t)$. It is never exhausted except by reaching 0.

## Brent fallback

`QuadraticHazard` and `CubicHazard` invert by a gated closed form (`bdsurv._polyroots`): Newton from the linear-hazard offset where the nonlinear part is small, the algebraic cubic or quartic root polished by Newton otherwise, and Brent only for elements that fail a residual or conditioning check. They still report `has_direct_sampler = False`, because an uncapped call can raise for a hazard valid only on $[0, T]$; with a finite `end_time` the simulators use the vectorised capped closed form.

A hazard without a direct sampler (`SumHazard` (single `E`), or a user subclass) inverts with `scipy.optimize.brentq` (`bdsurv._invert`).
The solved function is $g(r) = H(t_0 + r) - H(t_0) - e$, evaluated through `cumulative_between` so that the stable form is used.

Forward:

1. $e = 0$ gives 0.
2. If $e \ge H(\infty) - H(t_0)$ the result is `inf`.
3. If a finite `cap` is given and $g(\mathrm{cap} - t_0) < 0$ there is no event before the cap and the result is the cap offset.
   Otherwise the bracket is $[0, \mathrm{cap} - t_0]$.
   The hazard is never evaluated beyond `cap`, so a `QuadraticHazard` with $c < 0$ that is non-negative only on $[0, T]$ is safe.
4. With no cap the upper end starts at $e / h(t_0)$ if $h(t_0)$ is finite and positive, and at $\max(t_0, 1)$ otherwise.
   It is doubled up to `MAX_BRACKET_EXPANSIONS = 200` times until $g \ge 0$.

Backward: the bracket is $[0, x]$, and the result is $x$ if $H(x) \le e$.

Tolerances are `BRENT_XTOL = 1e-14` and `BRENT_RTOL = 4 eps`, keyword arguments of `forward_offset` and `backward_offset` in `bdsurv._invert`.
The result is clipped to be at least 0.

Errors:

- `HazardInversionError` (a `RuntimeError`) if $g$ is not finite, no bracket is found after the expansion limit, or `brentq` does not converge.
  It is distinct from the `ValueError` for invalid parameters, a negative or non-finite $e$, or a hazard that is negative where it is used.
- `QuadraticHazard`/`CubicHazard` `forward_offsets` without a cap needs $h \ge 0$ on $[0, \infty)$ and raises `ValueError` otherwise; `backward_offsets` checks $[0, \max x]$.

## No-event conventions

When the hazard is defective or the window ends, there is no event:

- In the simulators (`forward_time`, `backward_time`) the result is the `cap` going forward, the stopping time $T$, and 0 going backward.
- In `sample_residual_lifetimes` it is `inf`: for a defective draw, for $t_0 + R >$ `end_time`, and for a residual that runs past the `end_time` of a `PiecewiseHazard`.

## Piecewise hazards

Piece $k$ applies on $(\tau_{k-1}, \tau_k]$, with $\tau_0 = 0$, and sees the local time $u = t - \tau_{k-1}$.
So the piece's own clock restarts at each epoch, as for `PiecewiseLinear`.
The hazard values may jump at an epoch boundary; continuity is the caller's responsibility.
The constructor checks the epoch ends and calls `check_nonnegative(length)` on each piece for its epoch length.

The inversions walk the epochs, as `forward_hazard_time` and `backward_hazard_time` do for a piecewise-linear rate.
Going forward in epoch $k$ from local time $u_0$, if `piece.cumulative_between(u0, length)` is below the remaining target it is subtracted and the walk moves to the next epoch at $u = 0$; otherwise the piece inverts within the epoch.
The backward walk is the mirror image.
Past the last epoch the forward walk finds no event.
`hazard` and `cumulative_hazard` clip $t$ to $[0, \tau_{\mathrm{last}}]$.

`PiecewiseHazard.from_polynomials(coefficients, epoch_end_times)` builds piecewise-polynomial hazards.
Row $k$ gives $(a, b, c, d)$ in increasing power order, in the local time $u = t - \tau_{k-1}$.
A row may have one to four entries.
A row with $d = 0$ becomes a `QuadraticHazard`; otherwise a `CubicHazard`.
This is how a spline rate is passed to the simulators.

## Non-negativity

`check_nonnegative(end_time)` is called by `resolve_rates` (with $T$) and by `sample_residual_lifetimes` (when `end_time` is finite).
`QuadraticHazard` checks $h(0)$, $h(T)$ and the vertex $-b/(2c)$ if it lies in $(0, T)$, with a small relative tolerance.
`CubicHazard` checks $h(0)$, $h(T)$ and the real roots of $h'$ that lie in $(0, T)$, with the same tolerance.
Standalone and uncapped it needs $h \ge 0$ on $[0, \infty)$.
The other families are non-negative by their parameter constraints.

## Quadrature

The survival probability and the mean-born count need $\int \lambda$, $\int \mu$ and their combination in log space.
When all rates are constant or linear the existing quadrature is used unchanged.
If any rate is a `Hazard`, the generic quadrature is used:

- The edges are $0$, $t$ and the breakpoints of all three rates below $t$.
- Each interval is cut into $\lceil (\Delta\Lambda + \Delta M) / 4 \rceil$ panels, with $\Delta\Lambda$ and $\Delta M$ the cumulative hazards of birth and death over the interval.
  Their sum bounds the change of the exponent $|\Delta\Gamma|$ even if $\lambda - \mu$ changes sign.
- Each singular point $p$ gets geometric grading on its right, with extra edges at $p + w 2^{-j}$ for $j = 1, \dots, 40$ ($w$ the interval width).
- Each panel uses Gauss-Legendre nodes, with the panel-edge offset `PANEL_EDGE_OFFSET` as in the existing code.

The accuracy is high for smooth integrands.
It is limited to about $10^{-8}$ relative when the integrand itself is singular, for example the mean number born with a Weibull birth rate of shape $k < 1$ (the weight is the birth rate, which is infinite at 0).

## Draw mix

The generic forward path draws two exponentials per event, one each for birth and death.
The original merged-segment path draws one exponential and one uniform.
The totals from `count_draws` are therefore not comparable between a run with a `Hazard` and a run with constant or linear rates.
Seeded output of the original path is unchanged.

## Superposition (floors and sums)

If $h = h_1 + \dots + h_m$ and part $i$ has its own independent $E_i \sim \mathrm{Exp}(1)$, then $R = \min_i R_i(E_i)$ (forward) and $x - \max_i (x - D_i)$ (backward, floored at 0, an exhausted part giving 0) are exact in distribution for $h$: this is the competing-risks argument that `simulate_birth_death` already uses for birth against death.
Each part keeps its closed form, so a floor plus a closed-form family is closed form, at the cost of $m$ draws per event instead of 1.
The law is exact but the coupling to a single $E$ differs, so `forward_offsets(t0, e)` of a `SumHazard` (which must satisfy $H(t_0 + R) - H(t_0) = e$ exactly) still uses Brent.
Only samplers that draw their own exponentials use the split methods: `sample_residual_lifetimes`, the generic forward path and the contour up/down steps.

`SumHazard(parts)` takes at least two hazards (nested sums are flattened; no `PiecewiseHazard`, no part with an `end_time`).
`SumHazard.with_floor(floor, hazard)` adds a constant (`GompertzHazard(floor, 0.0)`) and returns `hazard` unchanged if `floor == 0`.
`LogisticHazard.with_floor(floor, L, k, c)` is the common case: a rising step ($k > 0$) or a falling step ($k < 0$, from $\text{floor} + L\sigma(-kc)$ down to `floor`).
A tanh step is a floor plus a logistic, through $\tanh z = 2\sigma(2z) - 1$.
For $b \ne 0$, $a \ge \lvert b\rvert$, $k > 0$, the hazard $a + b\tanh(k(t - c))$ equals `LogisticHazard.with_floor(a - abs(b), 2 * abs(b), 2 * sign(b) * k, c)`, and `LogisticHazard(2 * b, 2 * k, c)` when $a = b > 0$.
(For $b < 0$ the logistic is falling, $k_{\text{logistic}} = -2k$.)

Draw counts: a hazard with `n_draws > 1` draws that many exponentials per event as one vector (`size=m`), before any following draw; hazards with `n_draws == 1` draw exactly as before, so seeded output of existing hazards and rates is unchanged.
The totals from `count_draws` therefore grow with the number of parts.

## Usage

```python
import numpy as np

from bdsurv import (
    GompertzHazard,
    PiecewiseHazard,
    PowerHazard,
    sample_residual_lifetimes,
    simulate_birth_death,
    simulate_contour_given_survival,
)

# Weibull birth with shape 1.5, Gompertz death: terminal_time is required.
birth = PowerHazard.weibull(shape=1.5, scale=10.0)
death = GompertzHazard(a=0.05, b=0.1)
tree = simulate_contour_given_survival(birth, death, terminal_time=20.0, random_seed=0)

# A piecewise hazard fixes the end time, so terminal_time can be omitted.
piecewise = PiecewiseHazard(
    [PowerHazard.rayleigh(0.02), GompertzHazard(0.1, -0.05)], [5.0, 10.0]
)
process = simulate_birth_death(0.5, piecewise, random_seed=0)

# Residual lifetimes from several start times, left-truncated.
r = sample_residual_lifetimes(birth, np.array([0.0, 1.0, 5.0]), random_seed=1)
```

## Adding a family

1. Subclass `Hazard` and implement `hazard` and `cumulative_hazard`; they must be vectorised in numpy.
2. Override `cumulative_between` with a stable form if $H(e) - H(s)$ loses accuracy, and set `cumulative_limit`, `breakpoints` and `singular_points` where they apply.
   Override `check_nonnegative` if $h$ can be negative.
3. Optionally override `forward_offsets` and `backward_offsets` with closed forms and set `has_direct_sampler = True`.
   Without them the Brent defaults are used.
4. Add an instance to the shared parametrisation in `tests/test_hazard_families.py`, which checks known values, $H$ against $\int h$, sampling identities and the direct sampler against Brent.
