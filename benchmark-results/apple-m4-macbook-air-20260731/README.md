# Apple M4 MacBook Air — official 4M-cell benchmark

## Result

The 5-run median is **66.02 M cells/s**. If inserted into the 58-CUDA-GPU
ranking actually displayed on the [mumax³ website](https://mumax.github.io/)
when this result was checked, this Mac is **55th out of 59 total entries**:
immediately below the GTX 1050 (mobile) and above the GTX 860M.

| Rank after insertion | Device | Throughput |
| ---: | --- | ---: |
| 54 | GTX 1050 (mobile), CUDA | 72.61 M cells/s |
| **55** | **Apple M4 10-core GPU, Metal** | **66.02 M cells/s** |
| 56 | GTX 860M, CUDA | 55.28 M cells/s |

The M4 reaches 90.9% of the GTX 1050 mobile result and 119.4% of the GTX 860M
result. It is faster than 4 and slower than 54 of the 58 CUDA entries shown on
the website.

This is a sound result for a fanless, integrated 10-core GPU running a new
compatibility backend: the full 4M-cell workload completes and run-to-run
variation is only 0.49%. It is not high-end CUDA-class performance. The score
is 22.1% of the website list's 298.83 M cells/s median and far below modern
discrete GPUs, so large production sweeps remain substantially faster on
recent NVIDIA hardware.

## Measurements

| Run | Throughput |
| ---: | ---: |
| 1 | 65.860759 M cells/s |
| 2 | 66.432432 M cells/s |
| 3 | 66.018737 M cells/s |
| 4 | 65.698946 M cells/s |
| 5 | 66.543828 M cells/s |
| **Median** | **66.018737 M cells/s** |
| Mean | 66.110940 M cells/s |
| Range | 65.698946–66.543828 M cells/s |

Raw console logs are in `run-01.log` through `run-05.log`. Each corresponding
`run-NN.out/benchmark.txt` is the value written by mumax³ itself.
[`statistics.json`](statistics.json) contains the machine-readable summary.

## Method

The source is upstream
[`bench/bench.mx3`](https://github.com/mumax/3/blob/f656494b29516bead825b444b1f0b38c6e6c7dbf/bench/bench.mx3).
The checked-in [`benchmark-4m.mx3`](benchmark-4m.mx3) restricts only the
original `e=5..13` size sweep to `e=11`, the 2,048×2,048 point used by the
website's “2D simulations containing 4 million cells” chart. Material
parameters, solver 2, kernel/time-step warm-up, 100 measured steps, `Neval`,
wall-clock timing, and `N² × Neval / wall` throughput formula are unchanged.
The GUI was disabled during timing with `-http=""`.

The five runs were consecutive, with the Mac connected to AC power and Low
Power Mode off. The first run had to create the demagnetizing kernel cache;
that happens before the benchmark starts its timer.

| Environment item | Value |
| --- | --- |
| Computer | MacBook Air `Mac16,13` |
| SoC / GPU | Apple M4, 10 CPU cores, integrated 10-core GPU |
| Memory | 32 GB unified memory |
| OS | macOS 15.6 (24G84), arm64 |
| Backend | Metal 3 |
| Go | 1.26.5 |
| mumax³ | 3.12, runtime-reported commit `f133509f`; equivalent rewritten commit `120ec5ea` |
| Binary SHA-256 | `275a4cb842f1cd7c040c1706e90fb7e15d263f41ef705929714be3f23f794bbb` |
| Measurement time | 2026-07-31 02:53:08–02:54:21 KST |

The immutable raw logs retain the commit identifier printed by the benchmark
binary. Repository history maintenance subsequently rewrote that identifier;
`120ec5ea` is the equivalent source commit. The recorded binary SHA-256 remains
the authoritative executable identity.

Re-run from this directory with:

```bash
for run_number in 01 02 03 04 05; do
  mumax3 -http="" -f -o "run-${run_number}.out" benchmark-4m.mx3 \
    2>&1 | tee "run-${run_number}.log"
done
```

## Ranking provenance and caveat

The comparison uses the exact
[`gpus.svg`](https://mumax.github.io/gpus.svg) served by the official site,
SHA-256
`7d5d0c44fbe68bad4d61c079f5f6ee86517a1485535c647ab36b7c005efc6143`.
That chart contains 58 CUDA entries; the numeric source used to generate it is
preserved as [`website-gpu-ranking.txt`](website-gpu-ranking.txt).

The upstream source tree's
[`bench/gpus.txt`](https://github.com/mumax/3/blob/master/bench/gpus.txt) was
updated after the website chart and currently contains two additional GPUs.
Against that newer 60-entry source table, the same M4 result would be 57th out
of 61 after insertion. The headline rank uses the published website chart,
because that is the comparison requested.

This is a cross-backend comparison, not a controlled GPU laboratory test.
The scripts and throughput formula match, but the CUDA measurements were
submitted from different machines and dates, while this result uses Metal,
unified memory, and this fork's implementation.
