# Benchmarks

`simulator_speed.qmd` times every `bdsurv` simulator.
It is a Quarto notebook that needs only `bdsurv`; Quarto and Jupyter are
tools you run, not dependencies of the package.

```sh
uv run --with jupyter --with pyyaml quarto render benchmarks/simulator_speed.qmd
```

Exact-versus-theory validation of the simulated ages, and the timing of
sampling directly from the age distributions, live in `bdpinf`
(`qmd_scripts/validation/`).

`method_comparison.qmd` is a slide deck comparing the contour process, multilevel
splitting and a JAX simulate-and-keep-survivors baseline (cost against death rate
and horizon, batch throughput, JAX compile cost, and agreement with the exact
population law). It additionally needs `altair`.

```sh
uv run --with jupyter --with pyyaml --with altair quarto render benchmarks/method_comparison.qmd --to revealjs
```

Use `--to html` for a scrolling report instead of slides. Lower the `N_*` constants
at the top of the first cell for a quick draft render.
