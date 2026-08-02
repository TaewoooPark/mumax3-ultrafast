# Cross-tool benchmark — every micromagnetic simulator that runs on a Mac

These harnesses produce the speed, energy, capacity and physics-agreement
numbers in the top-level README. One machine, one session, one problem.

**The measured numbers are in [`RESULTS.md`](RESULTS.md).** This file documents
how they were obtained.

## The problem, identical in every arm

`n × n × 1` cells of 4 nm, `Ms = 8×10⁵ A/m`, `A = 1.3×10⁻¹¹ J/m`, `α = 0.02`,
demagnetizing + exchange fields only, `m₀ = normalize(1, 0.1, 0)`, **Heun**
integration at a **fixed 1×10⁻¹³ s step**.

The fixed step is the point. Under adaptive control a tool can finish sooner by
taking different steps, and the comparison would measure solver policy instead
of implementation. With a pinned step every arm performs exactly two
effective-field evaluations per step, and that was verified rather than assumed:
mumax³ reports `Neval = 2 × steps`, OOMMF reports
`energy_calc_count = 2 × steps + 1`.

Start-up is excluded everywhere. mumax³, magnum.np and MicroMagnetic.jl time an
inner loop after a two-step warm-up. OOMMF has no internal timer, so it is run at
N and 2N steps and the per-step cost is the slope, which cancels process start,
problem load and demag tensor construction exactly.

## Scripts

| Script | Produces |
| --- | --- |
| `run_crosstool.py` | throughput sweep and the ⟨mₓ⟩ agreement gate, 128² → 2048² |
| `run_energy.py` | joules per simulation, `powermetrics` at 2 Hz (needs `sudo`) |
| `run_capacity.py` | largest mesh each tool completes without paging out |
| `bench_mumax3.mx3`, `bench_magnumnp.py`, `bench_micromagnetic.jl`, `bench_oommf.mif` | the per-tool arms |

```bash
python3 -m venv venv && ./venv/bin/pip install torch magnumnp
julia -e 'using Pkg; Pkg.add(["MicroMagnetic","Metal"])'
# OOMMF: download the source and build with `tclsh oommf.tcl pimake`

python3 run_crosstool.py --sizes 128,256,512,1024,2048
sudo -v && python3 run_energy.py      # sudo needs a real TTY
python3 run_capacity.py
```

`MUMAX_WORK` points at a directory holding the mumax³ binaries to compare
(`ultrafast`, `formac-bin`); `bench/capacity.sh` in the parent repository builds
them from any two commits.

## What the competitors needed

Recorded because these are findings, not workarounds chosen to flatter a result.

**MicroMagnetic.jl 0.5.0 — its Metal backend does not run at all.** Apple GPUs
have no double-precision hardware, and every path into the package reaches a
`Float64` inside a Metal kernel. Three attempts, from most to least generous:

| What was run | Where it fails |
| --- | --- |
| `examples/std4.jl`, the package's own example, unmodified | `Sim()` — `create_zeros` asks Metal for a `Float64` array |
| `set_precision(Float32)`, public `run_until`, **exchange only, no demag** | `gpu_fill_kernel!` — "unsupported use of double value" |
| `set_precision(Float32)`, public `relax()` + `add_demag()` (the std4 idiom) | `gpu_tensors_kernel_xx!` → `newell_f` (demag.jl:134) |

So demag is one blocker, not the only one. `newell_f` is declared
`newell_f(x::Float64, y::Float64, z::Float64)::Float64` and `init_demag` casts
`dx, dy, dz` to `Float64`; converting those 23 signatures to `Float32` is still
not enough, because the numeric literals inside the Newell functions promote
everything back. Making the package work on Apple hardware means a
parametric-precision pass over the numeric core, not a configuration change. The
package was restored to pristine after the attempt, and the CPU backend — which
runs correctly and agrees with every other code — is what the benchmark reports.

**magnum.np — no Apple GPU path as shipped.** `magnumnp/__init__.py` selects
`cuda:N` or `cpu`, and calls `torch.set_default_dtype(torch.float64)` at import.
`demag.py` then sets `float64` explicitly to build the demag tensor ("always use
double precision"). `bench_magnumnp.py --device mps` overrides the device and
dtype after import and builds the tensor on the CPU before moving it to the GPU,
which is the only way to get a demag field on Apple hardware. Both arms are
reported: `mps` is magnum.np's best case, `cpu` is what `pip install` gives you.

Separately, magnum.np 2.2.0's `Heun` path is broken on every device —
`LLGSolver.step()` calls `self._solver.step(state.t, state.m, dt, state=state)`
while `Heun.step()` is declared `step(self, state, dt)`, and `Heun` is absent
from the solver list in `LLGSolver`'s own docstring. The harness drives
magnum.np's own `llg.dm()` from a Heun step written here, so the physics and the
field terms are entirely magnum.np's.

## Measurement hygiene

Two mistakes that changed results before they were caught:

- **Swap must be watched through `vm_stat`'s `Swapouts`, not `Pageouts`.** On
  Apple Silicon the memory compressor records swap under `Swapouts` and leaves
  `Pageouts` nearly still — 10.8M pages against 9.6k in one run — so a guard on
  `Pageouts` never fires and a thrashing mesh is scored as a fit.
- **mumax³ caches its demag kernel per geometry**, and at these sizes one entry
  is 8–10 GB. Left to accumulate it filled a 926 GB volume and killed a run. The
  capacity harness drops the cache after every run and checks free space before
  each size.

A third, subtler one: a tool can look memory-limited when the machine is simply
still under pressure from the previous arm. OOMMF appeared to fail at 4096²,
then passed the same size on a settled machine. Every ceiling here was confirmed
on a quiet machine.
