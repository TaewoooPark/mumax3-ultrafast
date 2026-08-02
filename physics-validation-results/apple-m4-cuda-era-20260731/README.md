# Apple M4 versus upstream CUDA-era physics baselines

## Result

At source commit
[`1c1e7a6d1d25d858f4a55275c12926b856a1e7f5`](https://github.com/TaewoooPark/mumax3-for-mac/commit/1c1e7a6d1d25d858f4a55275c12926b856a1e7f5),
the Apple M4 Metal backend passed all 15 selected tests and all 103 of their
physics assertions.

| Measure | Result |
| --- | ---: |
| Test conformance | **15/15 (100.000%)** |
| Assertion conformance within upstream tolerance | **103/103 (100.000%)** |
| Average-magnetization vector agreement, mean | **99.9923%** |
| Average-magnetization vector agreement, median | **99.9984%** |
| Average-magnetization vector agreement, minimum | **99.9415%** |
| Zero-tolerance assertions matched exactly | **19/19** |
| Positive-tolerance budget used, median | 3.7752% |
| Positive-tolerance budget used, 95th percentile | 87.1792% |
| Positive-tolerance budget used, maximum | 94.8336% |

Vector agreement is defined for each nonzero reference vector as:

```text
agreement (%) = 100 × (1 - ||m_Metal - m_reference||₂ / ||m_reference||₂)
```

The mean, median, and minimum above cover 10 dimensionless average-
magnetization vectors from `standardproblem4`, `standardproblem4_rk56`,
`standardproblem4-3d`, `standardproblem5a`, and `interexchange`. The recorded
inputs and individual results are in
[`magnetization-vectors.csv`](magnetization-vectors.csv). The complete summary
is also available as [`statistics.json`](statistics.json).

## What the percentage means

The 100.000% headline is an acceptance-rate measurement: every observed Metal
value satisfied the original upstream condition
`abs(Metal - reference) <= tolerance`. It does not mean that every
floating-point bit was identical.

The upstream mumax³ tree used for the comparison was CUDA-only. All 15 selected
test scripts are byte-for-byte unchanged relative to
[`f656494b29516bead825b444b1f0b38c6e6c7dbf`](https://github.com/mumax/3/tree/f656494b29516bead825b444b1f0b38c6e6c7dbf/test).
Their embedded references include historical mumax³ regression goldens,
physical or analytic invariants, and explicit independent OOMMF values. This
combination tests backend conformance more broadly than one trajectory alone,
but it is not a controlled run of two GPUs in the same host.

The official throughput benchmark places this M4 between a GTX 1050 mobile and
a GTX 860M. That identifies a similar performance class only. The official GPU
chart publishes throughput, not same-input field arrays from those cards, and
this Apple-silicon Mac cannot execute CUDA. Consequently, a direct,
card-specific GTX 1050/860M pairwise percentage cannot be measured on this
machine without an external NVIDIA host. Correct deterministic physics
references are not expected to change with CUDA card speed, apart from normal
floating-point reduction-order effects.

Thermal-noise trajectories are outside this percentage because Metal uses
Philox while the CUDA backend uses cuRAND's default XORWOW sequence. Equal
seeds therefore test distributional correctness, not sample-by-sample
identity.

## Test cohort

| Test | Assertions | Coverage |
| --- | ---: | --- |
| `standardproblem4.mx3` | 6 | Standard problem 4 relaxation and reversal |
| `standardproblem4-3d.mx3` | 6 | Three-dimensional standard problem 4 |
| `standardproblem4_rk56.mx3` | 6 | Standard problem 4 with RK56 |
| `standardproblem5a.mx3` | 9 | Spin-transfer-torque reversal |
| `demagSmall.mx3` | 9 | Small-grid demagnetizing field |
| `exchange.mx3` | 9 | Exchange field and region symmetry |
| `dmienergy.mx3` | 2 | DMI energy dissipation |
| `fixedlayer.mx3` | 3 | Slonczewski torque against OOMMF |
| `topologicalcharge-skyrmion.mx3` | 4 | Skyrmion topological charge |
| `mfm.mx3` | 1 | MFM field golden value |
| `rkky.mx3` | 1 | RKKY energy |
| `interexchange.mx3` | 3 | Antiferromagnetic inter-region exchange |
| `minimizer.mx3` | 3 | Energy minimization |
| `nodemagspins.mx3` | 36 | Demagnetizing-field masks and regions |
| `energy.mx3` | 5 | Energy terms against OOMMF |
| **Total** | **103** | |

## Measurement method

1. The repository was cloned from the exact source commit shown above into a
   temporary directory.
2. A measurement-only JSON log statement was inserted immediately before the
   existing `Expect` comparison. No solver, kernel, test input, tolerance, or
   reference value was changed.
3. Each selected input was run separately with the Metal backend, `-test`, and
   `-http=""`. Every process exited successfully.
4. The captured `have`, `want`, and `tolerance` triples were evaluated with the
   same absolute-error rule used by mumax³. Positive-tolerance utilization was
   calculated as `abs(have - want) / tolerance`.
5. Consecutive `m[0]`, `m[1]`, and `m[2]` assertions were grouped into the 10
   vectors used for the normalized-L2 calculation.

The maximum tolerance utilization was the first positive skyrmion-number
check: Metal produced `0.9857749633680211` for a reference of `1.0` with a
tolerance of `0.015`, using 94.8336% of that tolerance. This is a
discretized-topological-charge comparison with an analytic target, not a
94.8336% CUDA/Metal similarity score.

## Environment

| Item | Value |
| --- | --- |
| Computer | MacBook Air `Mac16,13` |
| SoC / GPU | Apple M4, integrated 10-core GPU |
| Memory | 32 GB unified memory |
| OS | macOS 15.6 (24G84), arm64 |
| Backend | Metal 3 |
| Go | 1.26.5 |
| Measurement time | 2026-07-31 KST |

The selected upstream test scripts are distributed under the repository's
[GPL-3.0-or-later license](../../LICENSE).
