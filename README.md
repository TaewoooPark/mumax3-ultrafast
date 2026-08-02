<!-- markdownlint-disable MD033 -->

# mumax³ ultrafast

**The fastest micromagnetic simulator on a Mac.**

<p align="center">
  <b>English</b> ·
  <a href="./README.ko.md">한국어</a> ·
  <a href="./README.zh.md">中文</a> ·
  <a href="./README.ja.md">日本語</a>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/TaewoooPark/mumax3-ultrafast?style=flat-square&logo=github&logoColor=white&labelColor=000000&color=333333" alt="GitHub stars">
  <img src="https://img.shields.io/github/last-commit/TaewoooPark/mumax3-ultrafast?style=flat-square&labelColor=000000&color=333333" alt="Last commit">
  <img src="https://img.shields.io/github/languages/top/TaewoooPark/mumax3-ultrafast?style=flat-square&labelColor=000000&color=333333" alt="Top language">
  &nbsp;
  <img src="https://img.shields.io/badge/Apple%20Silicon-000000?style=flat-square&logo=apple&logoColor=white&labelColor=000000" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/Metal-000000?style=flat-square&logo=apple&logoColor=white&labelColor=000000" alt="Metal">
  <img src="https://img.shields.io/badge/Go-000000?style=flat-square&logo=go&logoColor=white&labelColor=000000" alt="Go">
  <img src="https://img.shields.io/badge/Objective--C++-000000?style=flat-square&labelColor=000000&color=000000" alt="Objective-C++">
  &nbsp;
  <img src="https://img.shields.io/badge/mumax³%20compatible-000000?style=flat-square&labelColor=000000&color=000000" alt="mumax3 compatible">
  <img src="https://img.shields.io/badge/15%2F15%20upstream%20tests-000000?style=flat-square&labelColor=000000&color=000000" alt="15/15 upstream tests">
  <img src="https://img.shields.io/badge/No%20NVIDIA%20required-000000?style=flat-square&labelColor=000000&color=000000" alt="No NVIDIA required">
  <img src="https://img.shields.io/badge/GPL--3.0--or--later-000000?style=flat-square&labelColor=000000&color=000000" alt="GPL-3.0-or-later">
</p>

Upstream [mumax³](https://github.com/mumax/3) is built around NVIDIA CUDA, so it does not run on a Mac at all. This project replaces the GPU execution layer with a native **Metal** implementation and then optimizes it — and on an M4 MacBook Air it now runs **3.3× to 30× faster** than every other micromagnetic simulator that runs on a Mac, using **5.9× to 26× less energy** for the same physics.

The port changes the hardware backend, not the physical model. The `.mx3` language, the high-level Go solver, material terms, integration methods, and output formats stay mumax³-compatible — and the results agree with the CUDA-era references to within the original upstream tolerances.

> *"Fastest. Most efficient. And the only one that actually uses the GPU."*

[**taewoopark.com** — author site](https://taewoopark.com)

<p align="center">
  <img src="./docs/bench/hero.svg" alt="Three-panel comparison of every micromagnetic simulator that runs on Apple Silicon, measured on one M4 MacBook Air. Speed: mumax3-ultrafast at 136M cell-evaluations per second, 3.3x to 18.5x ahead of OOMMF, magnum.np and MicroMagnetic.jl. Energy: 22.5M cell-evaluations per joule, 6x to 26x ahead. Size: 84M cells, tied with OOMMF and 5x ahead of every other tool." width="100%">
</p>

---

## Why this exists

A physicist with a Mac has, until now, had three options: keep a Linux box with an NVIDIA card, rent one, or run on the CPU and wait.

- **mumax³** and **mumax+** are CUDA-only. They do not run on Apple hardware.
- **MicroMagnetic.jl** lists Apple GPUs among its supported backends. Version 0.5.0 **does not run on one.** Its own `examples/std4.jl`, unmodified, fails at `Sim()` — the package defaults to `Float64` and Apple GPUs have no double-precision hardware. Forcing `set_precision(Float32)` gets past that and then still fails inside GPU kernels: `gpu_fill_kernel!` for a plain exchange-only LLG step, and `newell_f` in the demag tensor. Every path reaches a double inside a Metal kernel.
- **magnum.np** has no Apple GPU path at all: `magnumnp/__init__.py` selects `cuda:N` or `cpu`, forces `float64` at import, and builds the demag tensor in double precision. Reaching MPS took three source patches.
- **OOMMF** has no GPU backend for any vendor.

So the Mac has been a second-class machine for micromagnetics. It no longer has to be.

---

## What the measurements say

One M4 MacBook Air (10 GPU cores, 32 GB, fanless), one session, every simulator solving **the identical problem**: demag + exchange, Heun integration at a fixed 1e-13 s step, the same number of steps per mesh. Every arm therefore performs exactly two effective-field evaluations per step — mumax³ reports `Neval = 2×steps`, OOMMF reports `energy_calc_count = 2×steps + 1`.

### Speed — fastest at every mesh, against every arm

| cell-evaluations / s | 128² | 256² | 512² | 1024² | 2048² |
| --- | ---: | ---: | ---: | ---: | ---: |
| **mumax3-ultrafast** | **1.61×10⁸** | **2.21×10⁸** | **1.36×10⁸** | **1.27×10⁸** | **1.17×10⁸** |
| OOMMF (CPU, 8 threads) | 3.17×10⁷ | 3.91×10⁷ | 4.14×10⁷ | 3.41×10⁷ | 2.85×10⁷ |
| magnum.np (MPS, patched) | 1.11×10⁷ | 2.28×10⁷ | 1.78×10⁷ | 2.63×10⁷ | 3.02×10⁷ |
| MicroMagnetic.jl (CPU) | 1.85×10⁷ | 1.62×10⁷ | 1.29×10⁷ | 1.03×10⁷ | 8.22×10⁶ |
| magnum.np (CPU, as shipped) | 7.41×10⁶ | 7.36×10⁶ | 7.36×10⁶ | 5.49×10⁶ | 4.97×10⁶ |
| MicroMagnetic.jl (Metal) | — | — | — | — | — |

**3.28× to 29.97× faster than every external tool, at every size measured.**

<p align="center">
  <img src="./docs/bench/speed.svg" alt="Throughput against mesh size for every micromagnetic simulator that runs on a Mac. mumax3-ultrafast leads at every mesh from 128 squared to 2048 squared; MicroMagnetic.jl's Metal backend cannot run at any size." width="100%">
</p>

<sub>MicroMagnetic.jl appears twice because its two backends behave differently: the CPU backend runs and is benchmarked normally, while the Metal backend does not run at all. magnum.np likewise appears twice — `MPS` is its best case after three source patches, `CPU` is what `pip install` gives you.</sub>

### Energy — the gap is even wider

<p align="center">
  <img src="./docs/bench/energy.svg" alt="Energy for one identical simulation: 512² mesh, 2000 Heun steps, 1.049 billion cell-evaluations per arm, measured with powermetrics. mumax3-ultrafast delivers 22.51 million cell-evaluations per joule, 5.9 to 26 times more than any other simulator." width="100%">
</p>

This is the axis nobody can settle against NVIDIA, because published comparisons divide measured throughput by a *nominal board TDP* and the answer moves by 3× depending on what fraction of TDP the card really draws. Restricting the question to one machine removes the assumption entirely: every arm ran on the same chip with `powermetrics` sampling the real GPU, CPU and combined rails.

| | wall | combined power | energy | **M cell-evals / J** |
| --- | ---: | ---: | ---: | ---: |
| **mumax3-ultrafast** | **7.6 s** | 6.10 W | **46.6 J** | **22.51** |
| magnum.np (MPS, patched) | 43.4 s | 6.38 W | 277.1 J | 3.78 |
| OOMMF (CPU, 8 threads) | 25.2 s | 16.42 W | 414.2 J | 2.53 |
| MicroMagnetic.jl (CPU) | 86.1 s | 6.51 W | 560.6 J | 1.87 |
| magnum.np (CPU, as shipped) | 143.7 s | 8.47 W | 1218.1 J | 0.86 |

<sub>512² mesh, 2000 Heun steps = 1.049×10⁹ cell-evaluations in every arm. Machine idle floor during the run: 0.195 W combined, 0.002 W GPU.</sub>

OOMMF is only 3.3× behind on time but burns **16.4 W** doing it, against 6.1 W on the GPU — so on energy it falls **8.9×** behind. A fanless laptop does the same physics for the power of a light bulb filament.

### Capacity — 83.9 million cells on a laptop

<p align="center">
  <img src="./docs/bench/capacity.svg" alt="Largest problem that fits on a 32 GB MacBook Air. mumax3-ultrafast and OOMMF both reach 83.9 million cells; magnum.np and MicroMagnetic.jl stop at 16.8 million; MicroMagnetic.jl's Metal backend cannot run." width="100%">
</p>

**83,886,080 cells** (8192 × 10240, 23.69 GiB peak) complete without paging out — at 4 nm cells that is a **32.8 × 41.0 µm** film, a whole patterned device rather than a corner of one. That is **5.0× more than any other GPU-capable tool on the platform**.

OOMMF reaches the same 83.9M — and mumax3-ultrafast is **4.5× faster there** (29.2 s against 131.1 s). Both fail at 104.9M on this machine.

### Physics — four independent codes, one answer

<p align="center">
  <img src="./docs/bench/physics.svg" alt="Relative deviation of the mean in-plane magnetization from mumax3-ultrafast across four independently written simulators. Every code agrees to better than 7.1e-5." width="100%">
</p>

A speed number is worthless if the answer is wrong. Every arm above was required to agree before its timing counted. After the same number of steps, all four independently written codes — three languages, three backends — return the same mean magnetization:

| ⟨mₓ⟩ after equal steps | 512² | 1024² | 2048² |
| --- | ---: | ---: | ---: |
| mumax3-ultrafast | 0.994939 | 0.9950321 | 0.9950429 |
| OOMMF | 0.994942 | 0.9950322 | 0.9950370 |
| MicroMagnetic.jl (CPU) | 0.994939 | 0.9950319 | 0.9950369 |
| magnum.np (MPS) | 0.994939 | 0.9950319 | 0.9950370 |

Worst disagreement anywhere in the sweep: **7.1×10⁻⁵**.

---

## Physics fidelity

The equations and solvers were **not reimplemented**. The native backend executes the existing mumax³ physical model through Metal, so the model you are citing is still mumax³. Validation against unchanged upstream CUDA-era regression references:

| Validation measure | Result |
| --- | ---: |
| Non-thermal upstream physics tests passed | **15 / 15** |
| Assertions within the original upstream tolerances | **103 / 103 (100.000%)** |
| Mean average-magnetization vector agreement | **99.9923%** |
| Minimum average-magnetization vector agreement | **99.9415%** |
| Zero-tolerance assertions matched exactly | **19 / 19** |
| Official mumax³ example-page simulations completed | **15 / 15** |

The 100% figure is tolerance conformance, not a claim of bit-for-bit identity — parallel GPU reductions can differ in their last floating-point bits. Thermal simulations use Philox instead of cuRAND's XORWOW, so a seed is reproducible on Metal but does not reproduce CUDA's sample-by-sample trajectory; the statistical behaviour is validated instead.

All 15 official example inputs are byte-for-byte identical to the upstream template, and their outputs, logs, and provenance are committed.

- [Complete physics-validation method and data](physics-validation-results/apple-m4-cuda-era-20260731/)
- [Official examples, outputs, provenance, and license](examples-and-results/RESULTS.md)

---

## Install

On a supported Mac, open Terminal and run:

```bash
/bin/bash -c "$(/usr/bin/curl -fsSL https://raw.githubusercontent.com/TaewoooPark/mumax3-ultrafast/main/install-macos.sh)"
```

The installer checks the machine and shell architecture, opens Apple's Command Line Tools installer when necessary, installs native Homebrew and a compatible Go toolchain if missing, clones and builds the project, configures the binary path, and finishes with `mumax3 -test`. It is safe to rerun after an interruption.

From an existing checkout:

```bash
./install-macos.sh
```

Open a new Terminal afterwards so the updated path is loaded.

> [!IMPORTANT]
> Apple Silicon (M1 or newer) on macOS 14 or newer. Intel Macs are not supported. The original CUDA backend remains available for NVIDIA-equipped Linux and Windows systems.

## Run a simulation

```bash
mumax3 example.mx3            # live Web UI at http://127.0.0.1:35367
mumax3 -http="" example.mx3   # headless, for benchmarks and batch jobs
```

Output lands in `example.out/`. Both modes execute the same simulation and produce the same output. The full Xcode application and an offline Metal compiler are not required — the shader library compiles through the system Metal runtime.

---

## How the native port works

| Upstream CUDA component | macOS implementation | Compatibility preserved |
| --- | --- | --- |
| CUDA compute kernels | Metal compute shaders and an Objective-C++ runtime bridge | Existing mumax³ kernel contracts and FP32 model |
| cuFFT | MPSGraph FFT, plus a vendored VkFFT path for eligible 2D transforms | Packed R2C/C2R layout used by the demag convolution |
| `cudaMalloc` and CUDA streams | Apple unified-memory buffers, command batching, one ordered Metal queue | Buffer semantics and execution ordering |
| cuRAND thermal noise | Philox4x32-10 with Box–Muller normal generation | Seeded reproducibility on Metal, validated noise statistics |
| CUDA-only platform bindings | `darwin/arm64` build tags and compatibility shims | The `.mx3` language and high-level Go API |

The implementation uses X-contiguous 32-wide SIMD tiles suited to Apple GPUs, keeps the GPU resident across blocking drains, splits large demag transforms one axis at a time to avoid MPSGraph's collapsed scheduling region, and compiles shaders at runtime so Command Line Tools alone are enough.

---

## Tuning

### Latency-bound runs

At research mesh sizes an Apple GPU is often idle waiting for the host rather than short of arithmetic: at 128×128 the host takes longer to encode a Dormand-Prince step than the GPU takes to run it. Three knobs address that, and none change what the default build does.

```go
SpeculativeStep = true   // overlap host encoding with GPU execution — 1.53× measured
MinimizeOnGPU   = true   // keep minimize()'s BB step size on the device — 1.58× measured
```

```sh
mumax3 -j 3 sweep_*.mx3   # N queued inputs per GPU — 2.53× aggregate for three minimize() jobs
```

`SpeculativeStep` still enforces `MaxErr`, but the rejection lands one step late, so the sequence of time steps differs from the exact controller and the trajectory diverges at the ~1% level after a few thousand steps. It closes itself under `FixDt`, finite `Temp`, `relax()`, `DemagExtrapolation`, and post-step hooks. `MinimizeOnGPU` keeps every descent bit-identical; only the convergence check is one iteration late, so a minimization can stop one iteration further along. Validate either against a default run for your own problem.

### Metal FFT backend

The default automatically uses the vendored VkFFT backend for eligible 2D demag transforms whose padded dimensions are powers of two no larger than 512×512 and whose active data is a strict prefix; MPSGraph handles everything else. `MUMAX3_METAL_FFT_BACKEND=mps` keeps all plans on MPSGraph; `vkfft` extends VkFFT to the padded 1024×1024 tier.

### Optional demag extrapolation

High-order demagnetizing-field extrapolation accelerates demag-heavy solver 4/5/6 workloads, but it is an approximation and is **off by default**:

```go
SetSolver(5)
DemagExtrapolation = true
```

Unsupported solvers and unsafe model states fail closed to exact convolution. The error depends on trajectory and time step, so a successful benchmark on another problem is not an accuracy guarantee — see the [validation results](bench/demag-extrap/RESULTS.md) and [A/B instructions](bench/demag-extrap/README.md) first.

### Sizing

Throughput per cell is not flat in problem size. It peaks at 256² on the measured M4, and at and below 128² a fixed per-evaluation overhead of about 172–187 µs dominates, where a wider GPU cannot help at all. `bench/curve.txt` has the measured curve.

---

## Reproduce the benchmark

Everything above is reproducible from this repository. Nothing is modelled.

```bash
./bench/capacity.sh name=/path/to/binary   # Metal limits, B/cell, vmmap split, ceiling
python3 bench/crosstool_svg.py             # re-render the README charts from the data
```

The cross-tool harnesses — one per simulator, all solving the same problem with the same integrator — live in [`crosstool/`](crosstool/): `run_crosstool.py` (speed and physics), `run_energy.py` (powermetrics), `run_capacity.py` (ceilings). Method notes, including the two patches magnum.np needs to reach the Apple GPU and what was tried before concluding MicroMagnetic.jl's Metal backend cannot run, are in each script's docstring.

Measurement hygiene that materially changed results, recorded so others avoid them: swap must be watched through `vm_stat`'s **`Swapouts`**, not `Pageouts` — on Apple Silicon the compressor leaves `Pageouts` nearly still while writing gigabytes — and mumax³'s per-geometry demag kernel cache reaches 8–10 GB per entry at these sizes, so it has to be dropped between runs.

---

## Scope

Stated plainly, because a benchmark that hides its edges is worth nothing:

- **Against NVIDIA on raw speed, this loses.** On mumax³'s own 62-GPU benchmark table an M4 lands between a GTX 1650 mobile and a GTX 970. The claim here is scoped to Apple Silicon deliberately — it is the range in which every axis can be measured on one machine with no TDP assumption and no cross-machine normalization.
- **Capacity ties OOMMF.** Both reach 83.9M cells and both fail at 104.9M. mumax3-ultrafast is 4.5× faster at that size; it is not larger.
- **The margin over the previous release narrows with mesh size** — 5.4× at 128², 1.38× at 2048². This round's work targeted the latency-bound regime where research meshes actually sit.

---

## Upstream and license

Derived from [mumax³](https://github.com/mumax/3), whose design and verification are described in the [original paper](https://doi.org/10.1063/1.4899186). NVIDIA CUDA users should follow the [official mumax³ installation documentation](https://mumax.github.io/download.html).

Distributed under the [GNU GPL v3 or later](LICENSE). The [project notices](NOTICE) preserve the upstream CUDA linking permission and identify the macOS modifications. Licenses and attribution for Random123, Go, SVGo, and Freetype-Go are collected in the [third-party notices](THIRD_PARTY_NOTICES.md).

Comparison targets are cited as measured, at the versions installed during the run: [MicroMagnetic.jl](https://github.com/ww1g11/MicroMagnetic.jl) 0.5.0 with Metal.jl 1.10 ([arXiv:2406.16064](https://arxiv.org/abs/2406.16064)), [magnum.np](https://gitlab.com/magnum.np/magnum.np) 2.2.0, and [OOMMF](https://math.nist.gov/oommf/) 2.0b0.

---

## Creator

I am **Taewoo Park**, an undergraduate physics student at the Korea Advanced Institute of Science and Technology (KAIST). Since October 2025 I have conducted experimental spintronics research on magnetic domain wall motion and neuromorphic computing applications at the [KAIST Ultrafast Spin Dynamics Laboratory (USDL)](https://spintronics.kaist.ac.kr/), led by **Professor Kab Jin Kim**. From June 2023 through March 2024 I studied domain wall motion through theoretical modeling and micromagnetic simulation in the KAIST Quantum Spin Dynamics Laboratory under **Professor Se Kwon Kim**.

<p align="center">
  <a href="https://github.com/TaewoooPark"><img src="https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github&logoColor=white&cacheSeconds=3600" alt="GitHub"></a>
  <a href="https://x.com/theoverstrcture"><img src="https://img.shields.io/badge/-X-000000?style=for-the-badge&logo=x&logoColor=white&cacheSeconds=3600" alt="X (Twitter)"></a>
  <a href="https://www.linkedin.com/in/taewoo-park-427a05352"><img src="https://img.shields.io/badge/-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white&cacheSeconds=3600" alt="LinkedIn"></a>
  <a href="https://taewoopark.com"><img src="https://img.shields.io/badge/-taewoopark.com-000000?style=for-the-badge&logo=safari&logoColor=white&cacheSeconds=3600" alt="Personal site"></a>
  <a href="mailto:ptw151125@kaist.ac.kr"><img src="https://img.shields.io/badge/-Email-D14836?style=for-the-badge&logo=gmail&logoColor=white&cacheSeconds=3600" alt="Email"></a>
</p>

<p align="center"><sub>Same equations. Same scripts. Same answers. On Apple Silicon.</sub></p>
