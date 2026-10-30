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
