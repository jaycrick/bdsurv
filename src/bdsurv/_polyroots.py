r"""Closed-form and Newton inversion of polynomial cumulative hazards.

The hazard is $h(t) = a + b t + c t^2 + d t^3$ (``d = 0`` for the quadratic).
Everything is solved in *local* time $r \ge 0$ from the start $t_0$, which avoids
the cancellation of large $t_0$:

$$P(r) = \alpha r + \beta r^2/2 + \gamma r^3/3 + \delta r^4/4 = e,$$

with $\alpha = h(t_0)$, $\beta = h'(t_0)$, $\gamma = c + 3 d t_0$ and
$\delta = d$. Backward from $x$ the coefficients are $(\alpha, -\beta, \gamma,
-\delta)$ evaluated at $x$, and $r$ runs over $[0, x]$.

Regimes (see :data:`ETA_NEWTON`)
--------------------------------
The solvers count which regime resolved each element in a
:class:`RegimeCounts` that the caller owns and passes in (``counts=``); there
are no module-level counters.

1. ``newton``: Newton from the quadratic truncation
   $r_q = $ :func:`~bdsurv._hazard.next_event_offset`, used when the
   nonlinearity $\eta = (|\gamma| r_q^3/3 + |\delta| r_q^4/4) / e$ is below
   :data:`ETA_NEWTON`. The linear hazard ($c = d = 0$) is this regime with
   $\eta = 0$ and no Newton step.
2. ``algebraic``: Cardano ($d = 0$) or Ferrari ($d \ne 0$) for the start, then
   two bracketed Newton steps.
3. ``brent``: the element falls back to :mod:`bdsurv._invert` when Newton does
   not converge, the hazard is not positive at an iterate, no admissible root
   exists, the residual check fails, or the problem is ill-conditioned (a
   near-double zero of the local hazard inside the window).

``method`` forces a regime: ``"gated"`` (default), ``"closed"`` (always regime
2, apart from the linear hazard) or ``"newton"`` (always regime 1). The Brent
fallback and the ill-conditioning check always apply.

Scalars use :mod:`math` only; arrays use numpy, with Brent only for the
elements that were flagged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bdsurv._hazard import next_event_offset
from bdsurv._invert import backward_offset, forward_offset

_Array = NDArray[np.float64]
_LocalArray = tuple[_Array, _Array, _Array, _Array]

type Method = Literal["gated", "closed", "newton"]

EPS: float = float(np.finfo(float).eps)
ETA_NEWTON: float = 1e-2
MAX_NEWTON: int = 6
POLISH_STEPS: int = 2
ILL_CONDITIONED: float = 1e-8
RESIDUAL_RTOL: float = 1e-10
RESIDUAL_EPS: float = 32.0
REAL_ROOT_TOL: float = 1e-8


class _Hazard(Protocol):
    @property
    def cumulative_limit(self) -> float: ...
    def hazard(self, t: float) -> object: ...
    def cumulative_hazard(self, t: float) -> object: ...
    def cumulative_between(self, s: float, e: float) -> object: ...


@dataclass
class RegimeCounts:
    """Number of elements resolved in each regime; pass one in and read it after.

    Elements with ``e == 0`` or no event in the window need no root and are not
    counted.
    """

    newton: int = 0
    algebraic: int = 0
    brent: int = 0

    @property
    def total(self) -> int:
        """The number of counted elements."""
        return self.newton + self.algebraic + self.brent

    @property
    def fallback_rate(self) -> float:
        """The fraction of counted elements that went to Brent."""
        return self.brent / self.total if self.total else 0.0

    def reset(self) -> None:
        """Set every count to 0."""
        self.newton = self.algebraic = self.brent = 0


Coefficients = tuple[float, float, float, float]
_Local = tuple[float, float, float, float]


# --------------------------------------------------------------------------
# Scalar path (math only)
# --------------------------------------------------------------------------


def _local_scalar(
    coef: Coefficients, t: float, sign: float
) -> tuple[float, float, float, float]:
    a, b, c, d = coef
    alpha = a + t * (b + t * (c + t * d))
    beta = b + t * (2.0 * c + 3.0 * d * t)
    gamma = c + 3.0 * d * t
    return alpha, sign * beta, gamma, sign * d


def _poly_scalar(al: float, be: float, ga: float, de: float, r: float) -> float:
    return r * (al + r * (be / 2.0 + r * (ga / 3.0 + r * de / 4.0)))


def _quad_real_scalar(b: float, c: float) -> list[float]:
    """Real roots of ``y^2 + b y + c`` (a near-double complex pair counts as real)."""
    disc = b * b - 4.0 * c
    if disc >= 0.0:
        sq = math.sqrt(disc)
        q = -0.5 * (b + math.copysign(sq, b))
        roots = [q]
        if q != 0.0:
            roots.append(c / q)
        else:
            roots.append(q)
        return roots
    re = -b / 2.0
    if math.sqrt(-disc) / 2.0 <= REAL_ROOT_TOL * (1.0 + abs(re)):
        return [re]
    return []


def cubic_roots_scalar(a3: float, a2: float, a1: float, a0: float) -> list[float]:
    """Real roots of ``a3 x^3 + a2 x^2 + a1 x + a0`` (``a3 != 0``).

    Cardano with the larger-magnitude cube root, so there is no cancellation
    in the one-real-root case; the trigonometric form when there are three
    real roots.
    """
    p2, p1, p0 = a2 / a3, a1 / a3, a0 / a3
    s = p2 / 3.0
    p = p1 - 3.0 * s * s
    q = 2.0 * s * s * s - s * p1 + p0
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    if disc > 0.0:
        w = -q / 2.0 - math.copysign(math.sqrt(disc), q)
        t = math.copysign(abs(w) ** (1.0 / 3.0), w)
        y = t - p / (3.0 * t) if t != 0.0 else 0.0
        return [y - s]
    if p == 0.0:
        return [-s]
    m = 2.0 * math.sqrt(-p / 3.0)
    cos3 = max(-1.0, min(1.0, -4.0 * q / (m * m * m)))
    theta = math.acos(cos3)
    return [m * math.cos((theta - 2.0 * math.pi * k) / 3.0) - s for k in range(3)]


def quartic_roots_scalar(
    c4: float, c3: float, c2: float, c1: float, c0: float
) -> list[float]:
    """Real roots of a quartic (``c4 != 0``) by Ferrari via the resolvent cubic."""
    b3, b2, b1, b0 = c3 / c4, c2 / c4, c1 / c4, c0 / c4
    s = b3 / 4.0
    p = b2 - 6.0 * s * s
    q = b1 - 2.0 * b2 * s + 8.0 * s**3
    r0 = b0 - b1 * s + b2 * s * s - 3.0 * s**4
    m = 0.0
    if q != 0.0:
        m = max(cubic_roots_scalar(8.0, 8.0 * p, 2.0 * p * p - 8.0 * r0, -q * q))
    if m <= 1e-14 * (abs(p) + math.sqrt(abs(r0))) or q == 0.0:
        out: list[float] = []
        for z in _quad_real_scalar(p, r0):
            if z > 0.0:
                root = math.sqrt(z)
                out.extend((root, -root))
            elif z == 0.0:
                out.append(0.0)
        return [y - s for y in out]
    s2 = math.sqrt(2.0 * m)
    half = p / 2.0 + m
    shift = q / (2.0 * s2)
    roots = _quad_real_scalar(s2, half - shift) + _quad_real_scalar(-s2, half + shift)
    return [y - s for y in roots]


def _crit_scalar(be: float, ga: float, de: float, *, quartic: bool) -> list[float]:
    """Positive zeros of ``h'(r) = be + 2 ga r + 3 de r^2``."""
    if not quartic:
        return [-be / (2.0 * ga)] if ga != 0.0 else []
    disc = ga * ga - 3.0 * de * be
    if disc < 0.0:
        return []
    q = -(ga + math.copysign(math.sqrt(disc), ga))
    out = [q / (3.0 * de)]
    if q != 0.0:
        out.append(be / q)
    return [r for r in out if r > 0.0]


def _ill_scalar(
    al: float, be: float, ga: float, de: float, hi: float, *, quartic: bool
) -> bool:
    for rc in _crit_scalar(be, ga, de, quartic=quartic):
        if 0.0 < rc < hi:
            hv = al + rc * (be + rc * (ga + rc * de))
            scale = abs(al) + abs(be) * rc + abs(ga) * rc**2 + abs(de) * rc**3
            if hv < ILL_CONDITIONED * scale:
                return True
    return False


def _newton_scalar(
    loc: _Local,
    e: float,
    hi: float,
    r: float,
    *,
    steps: int,
    converge: bool,
) -> float | None:
    """Bracketed Newton from ``r`` on ``[0, hi]``; ``None`` on failure."""
    al, be, ga, de = loc
    lo = 0.0
    violations = 0
    for _ in range(steps):
        f = _poly_scalar(al, be, ga, de, r) - e
        if f == 0.0:
            return r
        if f < 0.0:
            lo = r
        else:
            hi = min(hi, r)
        h = al + r * (be + r * (ga + r * de))
        if not h > 0.0:
            return None
        rn = r - f / h
        noise = (
            abs(al) * r + abs(be) * r * r / 2 + abs(ga) * r**3 / 3 + abs(de) * r**4 / 4
        )
        if abs(rn - r) <= 4.0 * EPS * rn or abs(f) <= 4.0 * EPS * (noise + e):
            return rn if rn > 0.0 else r
        if not lo < rn < hi:
            violations += 1
            if violations >= 2 or not math.isfinite(hi):
                return None
            rn = 0.5 * (lo + hi)
        r = rn
    return None if converge else r


def _residual_ok_scalar(
    al: float, be: float, ga: float, de: float, e: float, r: float
) -> bool:
    if not math.isfinite(r) or r < 0.0:
        return False
    noise = abs(al) * r + abs(be) * r * r / 2 + abs(ga) * r**3 / 3 + abs(de) * r**4 / 4
    f = abs(_poly_scalar(al, be, ga, de, r) - e)
    return f <= RESIDUAL_RTOL * e + RESIDUAL_EPS * EPS * noise


def _algebraic_scalar(
    loc: _Local, e: float, hi: float, *, quartic: bool
) -> float | None:
    al, be, ga, de = loc
    if quartic:
        roots = quartic_roots_scalar(de / 4.0, ga / 3.0, be / 2.0, al, -e)
    else:
        roots = cubic_roots_scalar(ga / 3.0, be / 2.0, al, -e)
    limit = hi * (1.0 + 1e-9) if math.isfinite(hi) else math.inf
    good = [r for r in roots if -1e-12 <= r <= limit]
    if not good:
        return None
    return min(max(min(good), 0.0), hi)


def solve_scalar(  # noqa: C901, PLR0911
    coef: Coefficients,
    t: float,
    e: float,
    hi: float,
    sign: float,
    method: Method = "gated",
) -> tuple[float, int]:
    """Smallest ``r`` in ``[0, hi]`` with ``P(r) = e``, and the regime used.

    Returns ``(hi, 0)`` if there is no event in the window, ``(0.0, 0)`` for
    ``e == 0``, and ``(nan, 3)`` when the caller must use Brent.
    """
    if e == 0.0:
        return 0.0, 0
    al, be, ga, de = _local_scalar(coef, t, sign)
    if hi == 0.0:
        return 0.0, 0
    quartic = coef[3] != 0.0
    if coef[2] == 0.0 and not quartic:
        r = next_event_offset(al, be, e)
        return (hi, 0) if r > hi else (r, 1)
    if math.isfinite(hi) and _poly_scalar(al, be, ga, de, hi) < e:
        return hi, 0
    if _ill_scalar(al, be, ga, de, hi, quartic=quartic):
        return math.nan, 3
    rq = next_event_offset(al, be, e)
    if math.isfinite(rq):
        eta = (abs(ga) * rq**3 / 3.0 + abs(de) * rq**4 / 4.0) / e
    else:
        eta = math.inf
    regime = (
        1 if (method == "newton" or (method == "gated" and eta < ETA_NEWTON)) else 2
    )
    r: float | None = None
    if regime == 1:
        if math.isfinite(rq):
            r = _newton_scalar(
                (al, be, ga, de), e, hi, min(rq, hi), steps=MAX_NEWTON, converge=True
            )
    else:
        start = _algebraic_scalar((al, be, ga, de), e, hi, quartic=quartic)
        if start is not None:
            r = _newton_scalar(
                (al, be, ga, de), e, hi, start, steps=POLISH_STEPS, converge=False
            )
    if r is None or not _residual_ok_scalar(al, be, ga, de, e, r):
        return math.nan, 3
    return min(r, hi), regime


# --------------------------------------------------------------------------
# Vectorised path (numpy)
# --------------------------------------------------------------------------


def _local_array(
    coef: Coefficients, t: _Array, sign: float
) -> tuple[_Array, _Array, _Array, _Array]:
    a, b, c, d = coef
    alpha = a + t * (b + t * (c + t * d))
    beta = b + t * (2.0 * c + 3.0 * d * t)
    gamma = c + 3.0 * d * t
    return alpha, sign * beta, gamma, np.full_like(t, sign * d)


def _poly_array(al: _Array, be: _Array, ga: _Array, de: _Array, r: _Array) -> _Array:
    return r * (al + r * (be / 2.0 + r * (ga / 3.0 + r * de / 4.0)))


def _noise_array(al: _Array, be: _Array, ga: _Array, de: _Array, r: _Array) -> _Array:
    return (
        np.abs(al) * r
        + np.abs(be) * r * r / 2.0
        + np.abs(ga) * r**3 / 3.0
        + np.abs(de) * r**4 / 4.0
    )


def _next_event_array(a: _Array, b: _Array, e: _Array) -> _Array:
    disc = a * a + 2.0 * b * e
    denom = a + np.sqrt(np.where(disc < 0.0, np.nan, disc))
    return np.where(denom > 0.0, 2.0 * e / denom, np.inf)


def _quad_real_array(b: _Array, c: _Array) -> _Array:
    """Real roots of ``y^2 + b y + c`` as an ``(n, 2)`` array (nan if none)."""
    disc = b * b - 4.0 * c
    sq = np.sqrt(np.maximum(disc, 0.0))
    q = -0.5 * (b + np.copysign(sq, b))
    safe_q = np.where(q != 0.0, q, 1.0)
    second = np.where(q != 0.0, c / safe_q, q)
    first = q
    near = (disc < 0.0) & (
        np.sqrt(-np.minimum(disc, 0.0)) / 2.0 <= REAL_ROOT_TOL * (1.0 + np.abs(b / 2.0))
    )
    first = np.where(near, -b / 2.0, first)
    second = np.where(near, np.nan, second)
    bad = (disc < 0.0) & ~near
    first = np.where(bad, np.nan, first)
    second = np.where(bad, np.nan, second)
    return np.stack([first, second], axis=1)


def cubic_roots_array(a3: _Array, a2: _Array, a1: _Array, a0: _Array) -> _Array:
    """Real roots of cubics as an ``(n, 3)`` array, ``nan`` where not real."""
    p2, p1, p0 = a2 / a3, a1 / a3, a0 / a3
    s = p2 / 3.0
    p = p1 - 3.0 * s * s
    q = 2.0 * s * s * s - s * p1 + p0
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    one = disc > 0.0
    out = np.full((*a3.shape, 3), np.nan)
    w = -q / 2.0 - np.copysign(np.sqrt(np.maximum(disc, 0.0)), q)
    t = np.cbrt(w)
    safe_t = np.where(t != 0.0, t, 1.0)
    y1 = np.where(t != 0.0, t - p / (3.0 * safe_t), 0.0) - s
    out[:, 0] = np.where(one, y1, np.nan)
    m = 2.0 * np.sqrt(np.maximum(-p / 3.0, 0.0))
    safe_m = np.where(m > 0.0, m, 1.0)
    cos3 = np.clip(-4.0 * q / (safe_m**3), -1.0, 1.0)
    theta = np.arccos(cos3)
    for k in range(3):
        yk = np.where(m > 0.0, m * np.cos((theta - 2.0 * np.pi * k) / 3.0), 0.0) - s
        out[:, k] = np.where(one, out[:, k], yk)
    return out


def quartic_roots_array(
    c4: _Array, c3: _Array, c2: _Array, c1: _Array, c0: _Array
) -> _Array:
    """Real roots of quartics as an ``(n, 4)`` array, ``nan`` where not real."""
    b3, b2, b1, b0 = c3 / c4, c2 / c4, c1 / c4, c0 / c4
    s = b3 / 4.0
    p = b2 - 6.0 * s * s
    q = b1 - 2.0 * b2 * s + 8.0 * s**3
    r0 = b0 - b1 * s + b2 * s * s - 3.0 * s**4
    res = cubic_roots_array(
        np.full_like(p, 8.0), 8.0 * p, 2.0 * p * p - 8.0 * r0, -q * q
    )
    m = np.nanmax(np.where(np.isnan(res), -np.inf, res), axis=1)
    m = np.where(np.isfinite(m), m, 0.0)
    biquad = (q == 0.0) | (m <= 1e-14 * (np.abs(p) + np.sqrt(np.abs(r0))))
    # Ferrari factorisation
    s2 = np.sqrt(2.0 * np.where(biquad, 1.0, m))
    half = p / 2.0 + m
    shift = q / (2.0 * s2)
    qa = _quad_real_array(s2, half - shift)
    qb = _quad_real_array(-s2, half + shift)
    ferrari = np.concatenate([qa, qb], axis=1)
    # Biquadratic case: solve for z = y^2.
    zs = _quad_real_array(p, r0)
    bq = np.full((*p.shape, 4), np.nan)
    for k in range(2):
        z = zs[:, k]
        root = np.sqrt(np.where(z >= 0.0, z, np.nan))
        bq[:, 2 * k] = root
        bq[:, 2 * k + 1] = -root
    out = np.where(biquad[:, None], bq, ferrari)
    return out - s[:, None]


def _crit_array(be: _Array, ga: _Array, de: _Array, *, quartic: bool) -> list[_Array]:
    nan = np.full_like(be, np.nan)
    if not quartic:
        safe = np.where(ga != 0.0, ga, 1.0)
        return [np.where(ga != 0.0, -be / (2.0 * safe), nan)]
    disc = ga * ga - 3.0 * de * be
    ok = disc >= 0.0
    q = -(ga + np.copysign(np.sqrt(np.where(ok, disc, 0.0)), ga))
    r1 = np.where(ok, q / (3.0 * de), np.nan)
    safe_q = np.where(q != 0.0, q, 1.0)
    r2 = np.where(ok & (q != 0.0), be / safe_q, np.nan)
    return [r1, r2]


def _ill_array(
    al: _Array, be: _Array, ga: _Array, de: _Array, hi: _Array, *, quartic: bool
) -> NDArray[np.bool_]:
    ill = np.zeros(al.shape, dtype=bool)
    for rc in _crit_array(be, ga, de, quartic=quartic):
        inside = (rc > 0.0) & (rc < hi)
        rr = np.where(inside, rc, 0.0)
        hv = al + rr * (be + rr * (ga + rr * de))
        scale = np.abs(al) + np.abs(be) * rr + np.abs(ga) * rr**2 + np.abs(de) * rr**3
        ill |= inside & (hv < ILL_CONDITIONED * scale)
    return ill


def _newton_array(
    loc: _LocalArray,
    e: _Array,
    hi0: _Array,
    r0: _Array,
    *,
    steps: int,
    converge: bool,
) -> tuple[_Array, NDArray[np.bool_]]:
    """Bracketed Newton on ``[0, hi0]``; returns the iterate and a success mask."""
    al, be, ga, de = loc
    r = r0.copy()
    lo = np.zeros_like(r)
    hi = hi0.copy()
    violations = np.zeros(r.shape, dtype=np.int64)
    active = np.ones(r.shape, dtype=bool)
    failed = np.zeros(r.shape, dtype=bool)
    for _ in range(steps):
        if not active.any():
            break
        f = _poly_array(al, be, ga, de, r) - e
        lo = np.where(active & (f < 0.0), r, lo)
        hi = np.where(active & (f > 0.0), np.minimum(hi, r), hi)
        h = al + r * (be + r * (ga + r * de))
        failed |= active & ~(h > 0.0)
        safe_h = np.where(h > 0.0, h, 1.0)
        rn = r - f / safe_h
        noise = _noise_array(al, be, ga, de, r)
        done = (np.abs(rn - r) <= 4.0 * EPS * rn) | (
            np.abs(f) <= 4.0 * EPS * (noise + e)
        )
        done |= f == 0.0
        outside = active & ~failed & ~done & ~((rn > lo) & (rn < hi))
        violations += outside
        failed |= outside & (~np.isfinite(hi) | (violations >= 2))
        rn = np.where(outside, 0.5 * (lo + np.where(np.isfinite(hi), hi, lo)), rn)
        rn = np.where((f == 0.0) | ~(rn > 0.0), r, rn)
        r = np.where(active & ~failed, rn, r)
        active = active & ~failed & ~done
    ok = ~failed & ~active if converge else ~failed
    return r, ok


def solve_array(  # noqa: PLR0915
    coef: Coefficients,
    t: _Array,
    e: _Array,
    hi: _Array,
    sign: float,
    method: Method = "gated",
) -> tuple[_Array, NDArray[np.int8]]:
    """Vectorised :func:`solve_scalar` on flat arrays.

    Returns ``r`` and the regime per element (0 = no root needed, 1, 2, or 3 =
    Brent needed with ``r = nan``).
    """
    n = t.size
    r = np.full(n, np.nan)
    regime = np.zeros(n, dtype=np.int8)
    todo = (e > 0.0) & (hi > 0.0)
    r[~todo] = 0.0
    if not todo.any():
        return r, regime
    quartic = coef[3] != 0.0
    with np.errstate(all="ignore"):
        al, be, ga, de = _local_array(coef, t, sign)
        rq_all = _next_event_array(al, be, e)
        if coef[2] == 0.0 and not quartic:
            none = todo & (rq_all > hi)
            r[none] = hi[none]
            lin = todo & ~none
            r[lin] = rq_all[lin]
            regime[lin] = 1
            return r, regime
        finite_hi = np.isfinite(hi)
        p_hi = _poly_array(al, be, ga, de, np.where(finite_hi, hi, 0.0))
        none = todo & finite_hi & (p_hi < e)
        r[none] = hi[none]
        work = todo & ~none
        ill = work & _ill_array(al, be, ga, de, hi, quartic=quartic)
        regime[ill] = 3
        work &= ~ill
        eta = np.where(
            np.isfinite(rq_all),
            (np.abs(ga) * rq_all**3 / 3.0 + np.abs(de) * rq_all**4 / 4.0) / e,
            np.inf,
        )
        if method == "newton":
            use1 = work
        elif method == "closed":
            use1 = np.zeros(n, dtype=bool)
        else:
            use1 = work & (eta < ETA_NEWTON)
        use2 = work & ~use1
        use1 &= np.isfinite(rq_all)
        regime[work & ~use1 & ~use2] = 3
        for mask, which in ((use1, 1), (use2, 2)):
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            a_, b_, g_, d_, e_, h_ = (v[idx] for v in (al, be, ga, de, e, hi))
            if which == 1:
                start = np.minimum(rq_all[idx], h_)
                got, ok = _newton_array(
                    (a_, b_, g_, d_), e_, h_, start, steps=MAX_NEWTON, converge=True
                )
            else:
                if quartic:
                    roots = quartic_roots_array(d_ / 4.0, g_ / 3.0, b_ / 2.0, a_, -e_)
                else:
                    roots = cubic_roots_array(g_ / 3.0, b_ / 2.0, a_, -e_)
                limit = np.where(np.isfinite(h_), h_ * (1.0 + 1e-9), np.inf)
                good = (roots >= -1e-12) & (roots <= limit[:, None])
                cand = np.where(good, roots, np.inf).min(axis=1)
                has = np.isfinite(cand)
                start = np.minimum(np.maximum(np.where(has, cand, 0.0), 0.0), h_)
                got, ok = _newton_array(
                    (a_, b_, g_, d_), e_, h_, start, steps=POLISH_STEPS, converge=False
                )
                ok &= has
            noise = _noise_array(a_, b_, g_, d_, got)
            res = np.abs(_poly_array(a_, b_, g_, d_, got) - e_)
            ok &= np.isfinite(got) & (got >= 0.0)
            ok &= res <= RESIDUAL_RTOL * e_ + RESIDUAL_EPS * EPS * noise
            r[idx] = np.where(ok, np.minimum(got, h_), np.nan)
            regime[idx] = np.where(ok, which, 3)
    return r, regime


# --------------------------------------------------------------------------
# Public helpers shared by QuadraticHazard and CubicHazard
# --------------------------------------------------------------------------


def _check_e(e: _Array) -> None:
    if not np.all(np.isfinite(e) & (e >= 0.0)):
        msg = "e must be non-negative and finite"
        raise ValueError(msg)


def _record(counts: RegimeCounts | None, regime: NDArray[np.int8]) -> None:
    if counts is not None:
        counts.newton += int(np.count_nonzero(regime == 1))
        counts.algebraic += int(np.count_nonzero(regime == 2))
        counts.brent += int(np.count_nonzero(regime == 3))


def forward_offsets(  # noqa: PLR0913
    hazard: _Hazard,
    coef: Coefficients,
    t0: ArrayLike,
    e: ArrayLike,
    cap: float = math.inf,
    *,
    method: Method = "gated",
    counts: RegimeCounts | None = None,
) -> _Array:
    """Vectorised offsets ``R`` with ``H(t0 + R) - H(t0) = e``.

    Returns ``cap - t0`` where there is no event before a finite ``cap``.
    """
    a, b = np.broadcast_arrays(
        np.asarray(t0, dtype=np.float64), np.asarray(e, dtype=np.float64)
    )
    _check_e(b)
    tt = np.ascontiguousarray(a).ravel()
    ee = np.ascontiguousarray(b).ravel()
    hi = np.maximum(cap - tt, 0.0) if math.isfinite(cap) else np.full(tt.size, np.inf)
    out, regime = solve_array(coef, tt, ee, hi, 1.0, method)
    for i in np.flatnonzero(regime == 3):
        out[i] = forward_offset(hazard, float(tt[i]), float(ee[i]), cap)
    _record(counts, regime)
    return out.reshape(a.shape)


def backward_offsets(
    hazard: _Hazard,
    coef: Coefficients,
    x: ArrayLike,
    e: ArrayLike,
    *,
    method: Method = "gated",
    counts: RegimeCounts | None = None,
) -> _Array:
    """Vectorised offsets ``D`` in ``[0, x]`` with ``H(x) - H(x - D) = e``."""
    a, b = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(e, dtype=np.float64)
    )
    _check_e(b)
    xx = np.ascontiguousarray(a).ravel()
    ee = np.ascontiguousarray(b).ravel()
    hi = np.maximum(xx, 0.0)
    out, regime = solve_array(coef, xx, ee, hi, -1.0, method)
    for i in np.flatnonzero(regime == 3):
        out[i] = backward_offset(hazard, float(xx[i]), float(ee[i]))
    _record(counts, regime)
    return out.reshape(a.shape)


def forward_time(  # noqa: PLR0913
    hazard: _Hazard,
    coef: Coefficients,
    start: float,
    e: float,
    cap: float,
    *,
    method: Method = "gated",
    counts: RegimeCounts | None = None,
) -> float:
    """Scalar ``min(start + R, cap)``; no numpy on the closed-form path."""
    if not (math.isfinite(e) and e >= 0.0):
        msg = "e must be non-negative and finite"
        raise ValueError(msg)
    hi = max(cap - start, 0.0) if math.isfinite(cap) else math.inf
    r, regime = solve_scalar(coef, start, e, hi, 1.0, method)
    if counts is not None:
        counts.newton += regime == 1
        counts.algebraic += regime == 2
        counts.brent += regime == 3
    if regime == 3:
        r = forward_offset(hazard, start, e, cap)
    return min(start + r, cap)


def backward_time(
    hazard: _Hazard,
    coef: Coefficients,
    start: float,
    e: float,
    *,
    method: Method = "gated",
    counts: RegimeCounts | None = None,
) -> float:
    """Scalar ``max(start - D, 0)``; ``D = start`` if ``H(start) <= e``."""
    if not (math.isfinite(e) and e >= 0.0):
        msg = "e must be non-negative and finite"
        raise ValueError(msg)
    hi = max(start, 0.0)
    d, regime = solve_scalar(coef, start, e, hi, -1.0, method)
    if counts is not None:
        counts.newton += regime == 1
        counts.algebraic += regime == 2
        counts.brent += regime == 3
    if regime == 3:
        d = backward_offset(hazard, start, e)
    return max(start - d, 0.0)
