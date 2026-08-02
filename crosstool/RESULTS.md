# Cross-tool benchmark — results

Every micromagnetic simulator that runs on a Mac, measured on one machine in one
session. This is the record the README charts are drawn from; the harnesses that
produced it are in this directory.

**Machine.** Apple M4, 10 GPU cores, 32 GB unified memory, fanless MacBook Air,
macOS 15.6 (24G84), on AC power.

**Problem, identical in every arm.** `n × n × 1` cells of 4 nm,
`Ms = 8×10⁵ A/m`, `A = 1.3×10⁻¹¹ J/m`, `α = 0.02`, demagnetizing + exchange only,
`m₀ = normalize(1, 0.1, 0)`, **Heun** integration at a **fixed 1×10⁻¹³ s step**,
the same number of steps per mesh in every arm.

The fixed step is what makes the comparison mean anything: under adaptive control
a tool can finish sooner by taking different steps. With the step pinned, every
arm performs exactly two effective-field evaluations per step — verified, not
assumed. mumax³ reports `Neval = 2 × steps`; OOMMF reports
`energy_calc_count = 2 × steps + 1`.

Start-up is excluded everywhere. mumax³, magnum.np and MicroMagnetic.jl time an
inner loop after a two-step warm-up. OOMMF has no internal timer, so it is run at
N and 2N steps and the per-step cost is taken as the slope.

**Versions as tested.** mumax3-ultrafast `c6a8c5d6` · MicroMagnetic.jl 0.5.0 with
Metal.jl 1.10 · magnum.np 2.2.0 with PyTorch 2.13.0 · OOMMF 2.0b0 (8 threads) ·
Julia 1.12.6 · Python 3.14.2 · Go 1.26.5.

---

## 1. Throughput — cell-evaluations per second

| | 128² | 256² | 512² | 1024² | 2048² |
| --- | ---: | ---: | ---: | ---: | ---: |
| **mumax3-ultrafast** | **1.611×10⁸** | **2.205×10⁸** | **1.360×10⁸** | **1.268×10⁸** | **1.170×10⁸** |
| OOMMF (CPU, 8 threads) | 3.173×10⁷ | 3.914×10⁷ | 4.143×10⁷ | 3.414×10⁷ | 2.850×10⁷ |
| magnum.np (MPS, patched) | 1.107×10⁷ | 2.277×10⁷ | 1.779×10⁷ | 2.632×10⁷ | 3.020×10⁷ |
| MicroMagnetic.jl (CPU) | 1.847×10⁷ | 1.618×10⁷ | 1.288×10⁷ | 1.030×10⁷ | 8.219×10⁶ |
| magnum.np (CPU, as shipped) | 7.410×10⁶ | 7.358×10⁶ | 7.362×10⁶ | 5.490×10⁶ | 4.967×10⁶ |
| MicroMagnetic.jl (Metal) | cannot run | cannot run | cannot run | cannot run | cannot run |

Speed-up of mumax3-ultrafast:

| | 128² | 256² | 512² | 1024² | 2048² |
| --- | ---: | ---: | ---: | ---: | ---: |
| over OOMMF | 5.08× | 5.63× | 3.28× | 3.71× | 4.11× |
| over magnum.np (MPS) | 14.55× | 9.68× | 7.64× | 4.82× | 3.87× |
| over MicroMagnetic.jl (CPU) | 8.72× | 13.63× | 10.55× | 12.32× | 14.23× |
| over magnum.np (CPU) | 21.74× | 29.97× | 18.47× | 23.10× | 23.55× |

**Range: 3.28× to 29.97×.**

## 2. Energy — one identical simulation

512² mesh, 2000 Heun steps = **1.049×10⁹ cell-evaluations in every arm**. GPU, CPU
and combined rails sampled with `powermetrics` at 2 Hz for the whole run; energy
is mean power × measured wall time. Machine idle floor during the run: 0.195 W
combined, 0.002 W GPU, 0.0% GPU residency.

| | wall | GPU W | CPU W | combined W | energy | **M cell-evals / J** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **mumax3-ultrafast** | **7.64 s** | 5.89 | 0.21 | 6.10 | **46.6 J** | **22.51** |
| magnum.np (MPS, patched) | 43.42 s | 5.67 | 0.71 | 6.38 | 277.1 J | 3.78 |
| OOMMF (CPU, 8 threads) | 25.23 s | 0.00 | 16.42 | 16.42 | 414.2 J | 2.53 |
| MicroMagnetic.jl (CPU) | 86.14 s | 0.00 | 6.51 | 6.51 | 560.6 J | 1.87 |
| magnum.np (CPU, as shipped) | 143.67 s | 0.01 | 8.47 | 8.48 | 1218.1 J | 0.86 |

**5.95× to 26.1× less energy for the same physics.** OOMMF is only 3.3× behind on
time but draws 16.42 W against 6.10 W, so it falls 8.9× behind on energy.

## 3. Capacity — largest mesh that completes without paging out

A run that pages out has left the single-device regime, so it is killed and
recorded as not fitting. Swap is watched through `vm_stat`'s `Swapouts`.

| | largest mesh | cells | peak footprint |
| --- | --- | ---: | ---: |
| **mumax3-ultrafast** | 8192 × 10240 | **83,886,080** | 23.69 GiB |
| OOMMF (CPU, 8 threads) | 8192 × 10240 | 83,886,080 | — |
| magnum.np (MPS, patched) | 4096 × 4096 | 16,777,216 | 7.47 GiB |
| magnum.np (CPU, as shipped) | 4096 × 4096 | 16,777,216 | 6.59 GiB |
| MicroMagnetic.jl (CPU) | 4096 × 4096 | 16,777,216 | 9.69 GiB |
| MicroMagnetic.jl (Metal) | — | 0 | cannot run |

Both mumax3-ultrafast and OOMMF fail at 10240 × 10240 = 104,857,600 cells, so
capacity is a tie between them — and 5.0× over every other tool. At the size
where they tie, mumax3-ultrafast is **4.5× faster** (29.2 s against 131.1 s).

OOMMF's peak footprint is not measurable this way because `boxsi` forks a solver
child, so `/usr/bin/time` only sees the launcher.

Measured on this machine: **≈300 bytes per cell** of total process footprint
(298.5 at 4096², 321.3 at 6144², 296.7 at 8192², 303.3 at the ceiling).
`vmmap` splits that into ≈200 B/cell of Metal buffers and ≈102 B/cell of Go heap,
the latter dominated by the 2×-padded demag kernel held host-side.

Metal device limits read from the API: `recommendedMaxWorkingSetSize` = 21.33 GiB
(exactly ⅔ of RAM, and advisory — the 23.69 GiB run exceeded it and was fine),
`maxBufferLength` = 16 GiB (RAM/2, never the binding constraint here).

## 4. Physics — the agreement gate

Every arm had to agree before its timing counted. Mean in-plane magnetization
after the same number of steps:

| ⟨mₓ⟩ | 128² | 256² | 512² | 1024² | 2048² |
| --- | ---: | ---: | ---: | ---: | ---: |
| mumax3-ultrafast | 0.990229 | 0.994200 | 0.994939 | 0.9950321 | 0.9950429 |
| OOMMF (CPU, 8 threads) | 0.990299 | 0.994216 | 0.994942 | 0.9950322 | 0.9950370 |
| MicroMagnetic.jl (CPU) | 0.990217 | 0.994198 | 0.994939 | 0.9950319 | 0.9950369 |
| magnum.np (MPS) | 0.990202 | 0.994195 | 0.994939 | 0.9950319 | 0.9950370 |
| magnum.np (CPU) | 0.990202 | 0.994195 | 0.994939 | 0.9950319 | 0.9950369 |

Worst relative disagreement anywhere in the sweep: **7.1×10⁻⁵**.

## 5. Why two arms per competitor

**MicroMagnetic.jl 0.5.0 does not run on an Apple GPU at all.** Three attempts,
most generous first:

| What was run | Where it fails |
| --- | --- |
| `examples/std4.jl`, the package's own example, unmodified | `Sim()` — `create_zeros` asks Metal for a `Float64` array |
| `set_precision(Float32)`, public `run_until`, exchange only, no demag | `gpu_fill_kernel!` — "unsupported use of double value" |
| `set_precision(Float32)`, public `relax()` + `add_demag()` | `gpu_tensors_kernel_xx!` → `newell_f` (demag.jl:134) |

Apple GPUs have no double-precision hardware and every path reaches a `Float64`
inside a Metal kernel. `newell_f` is declared
`newell_f(x::Float64, y::Float64, z::Float64)::Float64`; converting the 23
affected signatures to `Float32` is not enough because the numeric literals
promote everything back. The package was restored to pristine after the attempt.
`probe_micromagnetic_metal.jl` re-runs the three cases.

**magnum.np has no Apple GPU path as shipped.** `magnumnp/__init__.py` selects
`cuda:N` or `cpu` and calls `torch.set_default_dtype(torch.float64)` at import;
`demag.py` then forces `float64` again to build the demag tensor. The `MPS` arm
overrides device and dtype after import and builds the tensor on the CPU before
moving it to the GPU. Both arms are reported: `MPS` is magnum.np's best case,
`CPU` is what `pip install` gives you.

Separately, magnum.np 2.2.0's `Heun` path is broken on every device —
`LLGSolver.step()` calls `self._solver.step(state.t, state.m, dt, state=state)`
while `Heun.step()` is declared `step(self, state, dt)`, and `Heun` is absent
from the solver list in `LLGSolver`'s own docstring. The harness drives
magnum.np's own `llg.dm()` from a Heun step written here, so the physics and the
field terms remain entirely magnum.np's.

## 6. Measurement hygiene

Three mistakes that changed results before they were caught:

- **Swap must be watched through `vm_stat`'s `Swapouts`, not `Pageouts`.** On
  Apple Silicon the memory compressor records swap under `Swapouts` and leaves
  `Pageouts` nearly still — 10.8M pages against 9.6k in one run — so a guard on
  `Pageouts` never fires and a thrashing mesh is scored as a fit.
- **mumax³ caches its demag kernel per geometry**, and at these sizes one entry
  is 8–10 GB. Left to accumulate it filled a 926 GB volume and killed a run.
- **A tool can look memory-limited when the machine is still under pressure from
  the previous arm.** OOMMF appeared to fail at 4096², then passed the same size
  on a settled machine. Every ceiling here was confirmed on a quiet machine.
