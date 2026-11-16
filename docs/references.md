# References

These are the works the methods in `bdsurv` rest on, with the modules and docs pages that cite each.

- Kendall, D. G. (1948). On the generalized "birth-and-death" process. *Annals of Mathematical Statistics* 19(1), 1-15.

  Used in: `_hazard.py`, `survival.py`, `splitting.py`; `docs/splitting.md`, `docs/contour_conditioned.md`.

- Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical reactions. *Journal of Physical Chemistry* 81(25), 2340-2361.

  Used in: `_forward.py`, `forward.py`; `docs/lifespans.md`.

- Kahn, H. and Harris, T. E. (1951). Estimation of particle transmission by random sampling. *National Bureau of Standards Applied Mathematics Series* 12, 27-30.

  Used in: `splitting.py`, `era_conditioning.py`; `docs/splitting.md`, `docs/era_conditioning.md`.

- Villen-Altamirano, M. and Villen-Altamirano, J. (1991). RESTART: a method for accelerating rare event simulations. In *Queueing, Performance and Control in ATM* (ITC-13), 71-76.

  Used in: `splitting.py`; `docs/splitting.md`.

- Lambert, A. (2010). The contour of splitting trees is a Levy process. *Annals of Probability* 38(1), 348-395.

  TODO(author): confirm source of the contour construction

  Used in: `contour.py`, `contour_conditioned.py`; `docs/contour_process.md`, `docs/contour_conditioned.md`.

- Arley, N. (1943). *On the Theory of Stochastic Processes and Their Application to the Theory of Cosmic Radiation*. Copenhagen: Gads.

  Used in: `docs/contour_conditioned.md`; `README.md` (through the pointer to this page).
