<!-- markdownlint-disable MD033 -->

# mumax³ for macOS

**The CUDA-based micromagnetic simulator, rebuilt to run natively on Apple
Silicon.**

Upstream [mumax³](https://github.com/mumax/3) is built around NVIDIA CUDA,
which is unavailable on current Macs. This fork replaces that GPU execution
layer with a native Metal implementation, so an Apple-silicon Mac can run
ordinary `.mx3` simulations locally without an NVIDIA GPU, CUDA, a virtual
machine, or a remote Linux host.

The port changes the hardware backend, not the physical model. The `.mx3`
language, high-level Go solver, material terms, integration methods, and output
formats remain mumax³-compatible.

> [!IMPORTANT]
> The Metal backend supports Apple Silicon (M1 or newer) on macOS 14 or newer.
> Intel Macs are not supported. The original CUDA backend remains available for
> NVIDIA-equipped Linux and Windows systems.

## Install on a Mac

On a fresh supported Mac, open Terminal and run:

```bash
/bin/bash -c "$(/usr/bin/curl -fsSL https://raw.githubusercontent.com/TaewoooPark/mumax3-for-mac/master/install-macos.sh)"
```

The installer checks the machine and shell architecture, opens Apple's Command
Line Tools installer when necessary, installs native Homebrew and a compatible
Go toolchain if missing, clones and builds this Metal port, configures the
binary path, and finishes with `mumax3 -test`. It is safe to rerun after an
interruption because completed prerequisites are reused.

When installing from an existing checkout, run:

```bash
./install-macos.sh
```

Open a new Terminal after installation so the updated path is loaded.

## Run a simulation

The default mode runs the simulation with the live Web UI:

```bash
mumax3 example.mx3
```

The UI is served at <http://127.0.0.1:35367>, and simulation output is written
to `example.out/`. If a browser does not open automatically, open that address
manually.

For a headless benchmark, script, or batch job, disable the UI explicitly:

```bash
mumax3 -http="" example.mx3
```

Both modes execute the same simulation and produce the same output. The full
Xcode application and an offline Metal compiler are not required; the shader
library can compile through the system Metal runtime.

## How the native port works

| Upstream CUDA component | macOS implementation | Compatibility preserved |
| --- | --- | --- |
| CUDA compute kernels | Metal compute shaders and an Objective-C++ runtime bridge | Existing mumax³ kernel contracts and FP32 model |
| cuFFT | MPSGraph FFT | Packed R2C/C2R layout used by the demagnetizing-field convolution |
| `cudaMalloc` and CUDA streams | Apple unified-memory buffers, command batching, and one ordered Metal queue | Buffer semantics and execution ordering |
| cuRAND thermal noise | Philox4x32-10 with Box–Muller normal generation | Seeded reproducibility on Metal and validated noise statistics |
| CUDA-only platform bindings | `darwin/arm64` build tags and compatibility shims | The `.mx3` language and high-level Go API |

The implementation also uses X-contiguous 32-wide SIMD tiles suited to Apple
GPUs and runtime shader compilation so the project works with Command Line
Tools alone.

## Physics fidelity

The physical equations and solvers were not reimplemented independently; the
native backend executes the existing mumax³ model through Metal. Validation
against unchanged upstream CUDA-era regression references produced:

| Validation measure | Result |
| --- | ---: |
| Non-thermal upstream physics tests passed | **15/15** |
| Assertions within the original upstream tolerances | **103/103 (100.000%)** |
| Mean average-magnetization vector agreement | **99.9923%** |
| Minimum average-magnetization vector agreement | **99.9415%** |
| Zero-tolerance assertions matched exactly | **19/19** |
| Official mumax³ example-page simulations completed | **15/15** |

The 100% result is tolerance conformance, not a claim of bit-for-bit identity.
Parallel GPU reductions can differ in their last floating-point bits. Thermal
simulations use Philox instead of cuRAND's XORWOW sequence, so the same seed is
reproducible on Metal but does not generate the same sample-by-sample trajectory
as CUDA; the statistical behavior is validated instead.

See the [complete physics-validation method and data](physics-validation-results/apple-m4-cuda-era-20260731/)
and the [official examples, outputs, provenance, and GPL license](examples-and-results/RESULTS.md).

## Performance at a glance

For compact price-class context, the measured base M4 Metal result is compared
with the RTX 4050 Laptop GPU in the official mumax³ 4-million-cell benchmark.
Both appeared in consumer notebook families with overlapping launch-price
ranges, although this is not an exact configuration- or price-normalized test.

| Comparable notebook CUDA reference | M4 Metal throughput ÷ CUDA throughput |
| --- | ---: |
| RTX 4050 (mobile) | **37.5%** |

The measured machine was a fanless, base 10-core-GPU M4 MacBook Air, so this is
one conservative hardware datapoint rather than an Apple Silicon performance
ceiling. Absolute results, raw runs, the full CUDA comparison, projections,
limitations, and the measurement-submission protocol are kept in the
[benchmark documentation](benchmark-results/).

## Upstream and license

This project is derived from [mumax³](https://github.com/mumax/3), whose design
and verification are described in the
[original paper](https://doi.org/10.1063/1.4899186). NVIDIA CUDA users should
follow the [official mumax³ installation documentation](https://mumax.github.io/download.html).
This fork is distributed under the [GNU GPL v3 or later](LICENSE). The
[project notices](NOTICE) preserve the upstream CUDA linking permission and
identify the July 2026 macOS modifications. Licenses and attribution for
Random123, Go, SVGo, and Freetype-Go are collected in the
[third-party notices](THIRD_PARTY_NOTICES.md).

## Creator

I am **Taewoo Park**, and I am an undergraduate physics student at the Korea
Advanced Institute of Science and Technology (KAIST). Since October 2025, I
have conducted experimental spintronics research on magnetic domain wall motion
and neuromorphic computing applications at the
[KAIST Ultrafast Spin Dynamics Laboratory (USDL)](https://spintronics.kaist.ac.kr/),
led by **Professor Kab Jin Kim**. From June 2023 through March 2024, I studied
domain wall motion through theoretical modeling and micromagnetic simulation in
the KAIST Quantum Spin Dynamics Laboratory under **Professor Se Kwon Kim**.

<a href="https://taewoopark.com"><img src="https://img.shields.io/badge/-taewoopark.com-000000?style=for-the-badge&logo=safari&logoColor=white" alt="Personal site"></a>
<a href="mailto:ptw151125@kaist.ac.kr"><img src="https://img.shields.io/badge/-Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email"></a>
