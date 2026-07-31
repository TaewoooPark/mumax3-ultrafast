# Benchmark results

This directory stores reproducible runs of the
[official mumax³ benchmark](https://github.com/mumax/3/blob/master/bench/bench.mx3),
clearly labeled Apple Silicon projections, and the CUDA GPU data published on
the [mumax³ website](https://mumax.github.io/).

![Ranked mumax3 benchmark comparison](apple-silicon-vs-cuda.svg)

- Blue: verified Apple Metal measurements with raw evidence.
- Orange: calculated Apple Silicon projections, not measurements.
- Gray: all 58 entries in the frozen mumax³ CUDA website snapshot checked on
  2026-07-31.

The combined ordering is a visual comparison, not an official mumax³ ranking.
Throughput is measured in millions of cells per second for the same
4,194,304-cell problem; higher is faster.

## Verified measurements

- [`apple-m4-macbook-air-20260731/`](apple-m4-macbook-air-20260731/):
  Apple M4 MacBook Air, 10-core GPU, 32 GB unified memory, Metal backend,
  five-run median **66.02 M cells/s**.

[`registry.json`](registry.json) is the chart's authoritative measurement
index. Every measurement links to its statistics and evidence. New accepted
measurements can replace matching estimates without deleting the projection
inputs or historical evidence.

## Projection and chart data

- [`apple-silicon-estimates.json`](apple-silicon-estimates.json) contains the
  projection formula, assumptions, chip inputs, and source mapping.
- [`website-gpu-ranking.txt`](apple-m4-macbook-air-20260731/website-gpu-ranking.txt)
  preserves the 58 CUDA values used for the published comparison.
- [`tools/chart/`](tools/chart/) validates the source IDs, projection anchor,
  CUDA entry count, registry paths, and positive inputs before rendering the
  SVG.

Regenerate the deterministic chart from the repository root:

```bash
go run ./benchmark-results/tools/chart
```

The full formula, limitations, and primary references are disclosed in the
collapsible **Estimated benchmark calculation method and references** section
of the [main README](../README.md#verified-examples-and-performance).

## Contribute a measurement

Measured results from physical Apple Silicon Macs are welcome. Copy the
submission template, run five consecutive headless trials, add the result to
the registry, regenerate the SVG, and open a pull request. See the
[benchmark submission guide](submissions/) for the exact commands, privacy
rules, required files, and review checklist.
