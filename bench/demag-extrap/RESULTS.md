# Demagnetizing-field extrapolation validation

Date: 2026-08-01

Machine: Apple M4 (10 CPU cores, integrated Apple GPU), macOS 15.6, Go 1.26.5

Exact baseline revision: `8727cb8a`

## Decision

The implementation is useful as an **opt-in approximation for solver 4, 5,
or 6**, with exact A/B validation for the intended workload. It must remain off
by default. Solver 2 (Heun) and solver 3 (Bogacki--Shampine) are hard-disabled;
there is no user override. A coarse-step Heun diagnostic diverged materially
(field mean error about 0.83), while the high-order paths remained accurate.

The method follows Eq. (6) of [Lepadatu (2022)](https://arxiv.org/abs/2107.06729):
store accepted exact demagnetizing fields with the zero-displacement term
removed, extrapolate only that non-local component, and add the current local
self term exactly. [Boris2](https://github.com/SerbanL/Boris2) was used as an
open-source implementation cross-check.

## Accuracy and safety results

All comparisons used separate exact and extrapolated processes initialized by
identical scripts.

| Case | Main result |
|---|---|
| Adaptive µMAG standard problem 4, 1 ns | final average-`m` error `1.67e-7`; energy relative error `1.96e-6`; full-field mean/RMS/max error `6.59e-5 / 8.55e-5 / 2.29e-4` |
| Fixed-step trajectory, 128×32, 1 ns | max/RMS average-`m` error `1.17e-5 / 3.83e-6`; max energy error normalized by reference energy scale `1.77e-6` |
| Stepwise field and current reversals | max/RMS trajectory error `2.49e-6 / 2.40e-7`; normalized max/RMS energy error `8.12e-6 / 6.65e-7` |
| 250 GHz sinusoidal field and current | max/RMS trajectory error `5.78e-8 / 1.53e-8`; normalized max/RMS energy error `2.51e-7 / 8.13e-8` |
| Adaptive discontinuous drive | exact/extrapolated paths rejected `73 / 82` attempts; all 82 extrapolated retries were handled; final average-`m` error `1.14e-6`; energy relative error `9.23e-7`; full-field max error `6.19e-6` |
| Geometry + two `Msat` regions + XY PBC | solvers 4/5/6 all passed; 30-step full-field mean errors `3.01e-8 / 3.99e-8 / 2.16e-8` |
| Exact output-query isolation | per-step `B_demag` and `E_demag` queries left solver counters identical (`110` exact, `450` extrapolated) and produced zero final-field difference |
| Low-order guard | solvers 2 and 3 reported the hard-disable reason and performed zero extrapolated evaluations |

The discontinuity test uses step-interpolated data inside one `Steps` call, so
the field/current jumps occur without an intervening history reset. The
adaptive case combines discontinuities, rapid `dt` changes, and genuine step
rejection. A rejected attempt never enters accepted history; its exact
step-start sample is retained only for retry at the restored state.

Output and energy queries run after the manager leaves solver-step scope. They
therefore use the exact convolution and do not append, consume, or reset solver
history. This is covered by both a Go lifecycle test and the GPU A/B test above.

## Indicative performance

These timings were recorded on the machine above, but other large benchmarks
were running during part of the session. Treat them as indicative and repeat
in a quiet system state before publishing a final performance number.

| Workload | Exact | Extrapolated | Median speedup |
|---|---:|---:|---:|
| Fixed DP, 512×512×1, 200 steps | 2.934 s | 1.579 s | 1.86× |
| Adaptive standard problem 4, 1 ns, 3 repeats | 2.363 s | 1.157 s | 2.04× |
| Fixed DP, 128×32×1, 2000 steps | — | — | 2.37× |

The extrapolator keeps at most six three-component single-precision history
fields: 72 bytes per cell at fifth polynomial order. Priming uses exact fields
for the first five accepted steps. Thereafter DP typically replaces six of
seven demagnetizing convolutions per attempt; the step-start field stays exact.

## Fail-closed boundaries

Extrapolation is automatically disabled for:

- solver order below 4 or unsupported/non-explicit solvers;
- finite or time-dependent temperature;
- time-dependent `Msat`;
- any non-zero or time-dependent `NoDemagSpins` mask;
- relaxation/minimization;
- registered post-step callbacks;
- invalid, repeated, or ill-conditioned history times.

History is invalidated by solver, mesh, cell-size, PBC, kernel-accuracy,
geometry, region, `Msat`, `NoDemagSpins`, magnetization, and moving-window
changes. Direct Go code that mutates `M.Buffer()` must call
`ResetDemagExtrapolation()` explicitly.

## Reproduction

The full Go suite, Metal generator/unit/build checks, and the repository's
complete `.go`/`.mx3` GPU regression suite passed with the feature left at its
default `false` value.

```sh
go build -o /tmp/mumax3-demag-extrap ./cmd/mumax3
python3 bench/demag-extrap/run_ab.py --mumax /tmp/mumax3-demag-extrap
python3 bench/demag-extrap/run_std4_adaptive.py --mumax /tmp/mumax3-demag-extrap
python3 bench/demag-extrap/validate_matrix.py --mumax /tmp/mumax3-demag-extrap
python3 bench/demag-extrap/run_stress_ab.py --mumax /tmp/mumax3-demag-extrap
make check-metal
go test -vet=off ./...
```

Enable in an `.mx3` script only after choosing solver 4, 5, or 6:

```go
SetSolver(5)
DemagExtrapolation = true
```

Use `GetDemagExtrapolationStatus()`, `GetDemagExactEvals()`,
`GetDemagExtrapolatedEvals()`, and `GetDemagRejectedAttempts()` for runtime
diagnostics. Because the error is trajectory- and time-step-dependent, these
counters do not replace an exact scientific A/B comparison.
