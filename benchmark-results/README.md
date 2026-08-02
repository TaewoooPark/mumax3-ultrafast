# Benchmark documentation

This directory contains the reproducible performance evidence kept out of the
project's main README: the measured Apple Metal run, the frozen CUDA comparison
data published by mumax³, clearly labeled Apple Silicon projections, the chart
generator, and the protocol for contributing physical-device measurements.

## Comparison chart

![Ranked mumax3 benchmark comparison](apple-silicon-vs-cuda.svg)

- Blue bars are verified Apple Metal measurements with raw evidence.
- Orange bars are calculated Apple Silicon projections, not measurements.
- Gray bars are all 58 CUDA entries in the mumax³ website snapshot checked on
  2026-07-31.

The combined ordering is a visualization, not an official mumax³ ranking.
Every value uses the same 4,194,304-cell benchmark convention and is reported
in millions of cells per second; higher is faster.

## Verified M4 measurement

The physical anchor is an Apple M4 MacBook Air with a 10-core integrated GPU
and 32 GB of unified memory. Five consecutive runs of the official 4M-cell
workload produced a median of **66.018737 M cells/s**.

| Run | Throughput |
| ---: | ---: |
| 1 | 65.860759 M cells/s |
| 2 | 66.432432 M cells/s |
| 3 | 66.018737 M cells/s |
| 4 | 65.698946 M cells/s |
| 5 | 66.543828 M cells/s |
| **Median** | **66.018737 M cells/s** |

The complete environment, exact input, raw logs, output records, executable
hash, run-to-run statistics, and ranking provenance are in the
[measurement record](apple-m4-macbook-air-20260731/). The authoritative
machine-readable index is [`registry.json`](registry.json).

## Price-class CUDA context used in the main README

The main README deliberately shows one ratio rather than the tested laptop's
absolute score or inserted rank. Its CUDA reference is the RTX 4050 Laptop GPU:
Apple launched the 15-inch M4 MacBook Air family from USD 1,199, while Lenovo
listed its 16-inch IdeaPad Pro 5i family—configurable up to an RTX 4050 Laptop
GPU—from USD 1,149.99. These are overlapping notebook-family launch ranges,
not matched prices for the exact RAM, storage, GPU-power, or CPU configurations.
See the [Apple launch announcement](https://www.apple.com/newsroom/2025/03/apple-introduces-the-new-macbook-air-with-the-m4-chip-and-a-sky-blue-color/)
and the [Lenovo launch announcement](https://news.lenovo.com/pressroom/press-releases/new-ai-pc-experiences-thinkpad-ideapad-laptops-intel-core-ultra-processors/).

The relative throughput is calculated directly from the measured M4 median and
the frozen official CUDA dataset:

```text
relative throughput (%) = 100 × 66.0187366513 / published CUDA throughput
```

| Main README reference | Published CUDA result | M4 Metal ÷ CUDA |
| --- | ---: | ---: |
| RTX 4050 (mobile) | 175.971374 M cells/s | **37.5%** |

This remains notebook-segment context rather than a controlled price/performance
study. The host systems, power limits, cooling, software stacks, and purchase
dates differ. The complete set of source values is preserved in
[`website-gpu-ranking.txt`](apple-m4-macbook-air-20260731/website-gpu-ranking.txt),
which was extracted from the [mumax³ GPU chart](https://mumax.github.io/gpus.svg).

## Official benchmark method

The source workload is upstream
[`bench/bench.mx3`](https://github.com/mumax/3/blob/f656494b29516bead825b444b1f0b38c6e6c7dbf/bench/bench.mx3).
The checked-in
[`benchmark-4m.mx3`](apple-m4-macbook-air-20260731/benchmark-4m.mx3) changes
only the original size sweep from `e=5..13` to `e=11`, selecting the
2,048×2,048 problem used by the website's 4-million-cell chart. Material
parameters, solver 2, warm-up, 100 measured steps, `Neval`, wall-clock timing,
and the `N² × Neval / wall` throughput formula are unchanged. The Web UI was
disabled with `-http=""` to avoid timing an interactive browser session.

The five physical runs were consecutive, on AC power, with Low Power Mode off.
The first run created the demagnetizing-kernel cache before the benchmark timer
started. See the measurement record for the exact rerun command.

## Estimated Apple Silicon calculation

The orange bars use a deliberately simple balanced-roofline projection. The
only empirical calibration point is the measured 10-core M4 median:

```text
T₀ = 66.0187366513 million cells/s
B₀ = 120 GB/s

Rcompute   = target graphics-performance proxy / M4 graphics-performance proxy
Rbandwidth = target unified-memory bandwidth / 120 GB/s

Ttarget = T₀ × min(Rcompute, Rbandwidth)
```

`Rcompute` comes from Apple's published non-AI graphics-performance ratios.
For a binned GPU in the same chip family, the ratio is scaled linearly by active
GPU-core count. `Rbandwidth` uses Apple's published unified-memory bandwidth.
Taking the lower value prevents nominal shader capacity and memory bandwidth
from both being counted as if either could independently determine throughput.
Neural Engine, Neural Accelerator, and ray-tracing-only gains are excluded
because this backend uses ordinary FP32 Metal kernels and MPSGraph FFTs.

The formula, chip inputs, source identifiers, assumptions, and projected values
are machine-readable in
[`apple-silicon-estimates.json`](apple-silicon-estimates.json). The limiting
`min(compute, bandwidth)` structure follows the
[Roofline performance model](https://doi.org/10.1145/1498765.1498785).

### Projection limitations

An orange value is a chip-level estimate for the same workload, not a measured
Mac result. The model assumes unchanged backend efficiency, FFT selection,
fixed overhead, cooling, and sustained clocks. Different MacBook Air, MacBook
Pro, iMac, Mac mini, and Mac Studio enclosures can therefore produce different
measurements with the same chip. Projections are intended to prioritize real
testing, not replace it.

The gray CUDA values are also historical submissions from different machines
and dates. Although the workload and throughput convention match, this remains
a cross-backend comparison rather than a controlled GPU laboratory experiment.

## Sources

- mumax³: [official benchmark input](https://github.com/mumax/3/blob/master/bench/bench.mx3),
  [published GPU chart](https://mumax.github.io/gpus.svg), and
  [project website](https://mumax.github.io/)
- Metal: [Metal Performance Primitives Programming Guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
  and [GPU memory-bandwidth measurement](https://developer.apple.com/documentation/xcode/measuring-the-gpus-use-of-memory-bandwidth)
- M1 family: [M1 Pro and M1 Max](https://www.apple.com/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/)
  and [M1 Ultra](https://www.apple.com/newsroom/2022/03/apple-unveils-m1-ultra-the-worlds-most-powerful-chip-for-a-personal-computer/)
- M2 family: [M2](https://www.apple.com/ie/newsroom/2022/06/apple-unveils-m2-with-breakthrough-performance-and-capabilities/),
  [M2 Pro and M2 Max](https://www.apple.com/newsroom/2023/01/apple-unveils-macbook-pro-featuring-m2-pro-and-m2-max/),
  and [M2 Ultra](https://www.apple.com/uk/newsroom/2023/06/apple-introduces-m2-ultra/)
- M3 family: [M3, M3 Pro, and M3 Max](https://www.apple.com/ae/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/),
  [M3 Pro/Max specifications](https://support.apple.com/en-us/117737),
  [M3 Ultra](https://www.apple.com/mu/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/),
  and [M3 Ultra specifications](https://support.apple.com/en-us/122211)
- M4 family: [M4, M4 Pro, and M4 Max](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/)
  and [M4 Pro/Max specifications](https://support.apple.com/en-us/121554)
- M5 family: [M5](https://www.apple.com/ca/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/),
  [M5 Pro and M5 Max](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/),
  and [current M5-family specifications](https://www.apple.com/macbook-pro/specs/)
- Base-chip configurations: [M1 Air](https://support.apple.com/en-us/111883),
  [M2 Air](https://support.apple.com/en-us/111867),
  [M3 Air](https://support.apple.com/en-us/118551),
  [M4 Air](https://support.apple.com/en-us/122209), and
  [current M5 Air specifications](https://www.apple.com/macbook-air/specs/)

## Reproduce the chart

[`tools/chart/`](tools/chart/) validates source identifiers, the projection
anchor, the CUDA entry count, registry paths, and positive numeric inputs before
rendering the deterministic SVG. From the repository root, run:

```bash
go run ./benchmark-results/tools/chart
```

## Contribute a physical measurement

Measurements from other Apple Silicon Macs are welcome. Copy the provided
template, run five consecutive headless trials, add the result to the registry,
regenerate the chart, and open a pull request. The
[benchmark submission guide](submissions/) contains the exact commands,
privacy rules, required files, and review checklist. An accepted measurement
replaces the matching orange projection with a blue bar while retaining its
projection inputs and evidence history.
