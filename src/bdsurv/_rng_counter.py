"""Count the random numbers a conditioned simulation draws.

Wall-clock time depends on the machine and on implementation details such as
record copying.
The number of random draws is a machine-independent measure of how much
stochastic work a method does, so it supports a "theoretical efficiency"
comparison of the contour process and multilevel splitting.

Counting is off by default and costs one global check per instrumented site.
Inside a :func:`count_draws` block the generators created by
:func:`make_rng` are wrapped in :class:`CountingGenerator`, which forwards
every call unchanged, so a counted run produces exactly the same output as an
uncounted one.

Each draw is filed under a *role* and a *kind*:

roles
    ``contour`` (the contour excursions), ``dynamics`` (the event simulation of
    each splitting particle) and ``seeding`` (seeds, subsampling and the choice
    of trajectories to keep).
kinds
    ``exponential``, ``uniform``, ``integers`` and ``choice``.
    A ``size=n`` call counts as ``n`` draws; ``choice`` counts its ``size``,
    the number of items requested.

Other work counters (``events``, ``clones``, ``records_copied``, ``attempts``,
``particle_steps``, ``contour_steps``, ``spine_resets``, ``capped_levels``,
``dropped_by_cap``) are recorded with
:func:`bump`.
The state is module-global, so counting is not thread-safe.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

_KINDS = ("exponential", "uniform", "integers", "choice")


@dataclass
class DrawCounts:
    """Random draws, by ``(role, kind)``, and work counters from one counted run."""

    draws: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    extras: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def total_draws(self) -> int:
        """All random numbers drawn."""
        return sum(self.draws.values())

    def by_role(self, role: str) -> int:
        """Return the draws made by one role."""
        return sum(n for (r, _), n in self.draws.items() if r == role)

    def by_kind(self, kind: str) -> int:
        """Return the draws of one kind, over all roles."""
        return sum(n for (_, k), n in self.draws.items() if k == kind)

    def as_row(self) -> dict[str, int]:
        """Flatten to one row: totals, per-role and per-kind counts, the extras."""
        row: dict[str, int] = {"draws_total": self.total_draws}
        for role in ("contour", "dynamics", "seeding"):
            row[f"draws_{role}"] = self.by_role(role)
        for kind in _KINDS:
            row[f"draws_{kind}"] = self.by_kind(kind)
        for name in (
            "events",
            "clones",
            "records_copied",
            "attempts",
            "particle_steps",
            "contour_steps",
            "spine_resets",
            "capped_levels",
            "dropped_by_cap",
        ):
            row[name] = self.extras.get(name, 0)
        return row


_active: DrawCounts | None = None


def _n_items(size: int | tuple[int, ...] | None) -> int:
    return 1 if size is None else int(np.prod(size))


class CountingGenerator:
    """A ``numpy.random.Generator`` stand-in that tallies what it draws."""

    def __init__(self, rng: np.random.Generator, counts: DrawCounts, role: str) -> None:
        """Wrap ``rng``, filing its draws under ``role`` in ``counts``."""
        self._rng = rng
        self._counts = counts
        self._role = role

    def _add(self, kind: str, size: int | tuple[int, ...] | None) -> None:
        self._counts.draws[(self._role, kind)] += _n_items(size)

    def exponential(self, scale: Any = 1.0, size: Any = None) -> Any:
        """Draw as :meth:`numpy.random.Generator.exponential`."""
        self._add("exponential", size)
        return self._rng.exponential(scale, size)

    def uniform(self, low: Any = 0.0, high: Any = 1.0, size: Any = None) -> Any:
        """Draw as :meth:`numpy.random.Generator.uniform`."""
        self._add("uniform", size)
        return self._rng.uniform(low, high, size)

    def random(self, size: Any = None) -> Any:
        """Draw as :meth:`numpy.random.Generator.random` (counted as uniform)."""
        self._add("uniform", size)
        return self._rng.random(size)  # pyright: ignore[reportUnknownVariableType]

    def integers(self, low: Any, high: Any = None, size: Any = None) -> Any:
        """Draw as :meth:`numpy.random.Generator.integers`."""
        self._add("integers", size)
        return self._rng.integers(low, high, size)  # pyright: ignore[reportUnknownVariableType]

    def choice(
        self, a: Any, size: Any = None, replace: bool = True, p: Any = None
    ) -> Any:
        """Draw as :meth:`numpy.random.Generator.choice`."""
        self._add("choice", size)
        return self._rng.choice(a, size, replace, p)  # pyright: ignore[reportUnknownVariableType]

    def __getattr__(self, name: str) -> Any:
        """Forward anything else to the wrapped generator, uncounted."""
        return getattr(self._rng, name)


@contextmanager
def count_draws() -> Generator[DrawCounts]:
    """Count the random draws of the simulations run inside the ``with`` block.

    Yields a :class:`DrawCounts` that fills in as the block runs.
    Blocks may be nested; the outer one resumes counting on exit of the inner.
    """
    global _active  # noqa: PLW0603
    previous = _active
    counts = DrawCounts()
    _active = counts
    try:
        yield counts
    finally:
        _active = previous


def make_rng(seed: int | None, role: str) -> np.random.Generator:
    """Return ``np.random.default_rng(seed)``, counting its draws if counting is on."""
    rng = np.random.default_rng(seed)
    if _active is None:
        return rng
    return cast("np.random.Generator", CountingGenerator(rng, _active, role))


def bump(name: str, n: int = 1) -> None:
    """Add ``n`` to the work counter ``name`` if counting is on."""
    if _active is not None:
        _active.extras[name] += n
