#!/usr/bin/env python
"""magnum.np arm of the Apple Silicon cross-tool benchmark.

Same problem as every other arm: n x n x 1 cells of 4 nm, Ms = 8e5 A/m,
A = 1.3e-11 J/m, alpha = 0.02, demag + exchange only, m0 = normalize(1, 0.1, 0),
Heun at a fixed 1e-13 s step.

Two things had to be worked around, and both are reportable facts about running
magnum.np on a Mac rather than choices made to flatter another tool.

magnum.np picks its device in __init__.py as cuda:N or cpu - there is no Apple
path - and it calls torch.set_default_dtype(torch.float64) at import. Apple GPUs
have no float64 at all, so MPS needs both an explicit device override and a dtype
override applied after the import. --device mps does that; without it the package
runs on the CPU, which is what a user gets out of the box.

LLGSolver.step() calls self._solver.step(state.t, state.m, dt, state=state) while
Heun.step() is declared step(self, state, dt), so the Heun path raises TypeError
on any device. Heun is also absent from the solver list in LLGSolver's own
docstring. The loop below therefore calls llg.dm() - magnum.np's own LLG right
hand side, with magnum.np's own demag and exchange - from a Heun step written
here, which is the same integrator mumax3 runs as setsolver(2).
"""
import argparse, time, sys

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, required=True)
p.add_argument("--steps", type=int, default=100)
p.add_argument("--dt", type=float, default=1e-13)
p.add_argument("--device", choices=["mps", "cpu"], default="mps")
a = p.parse_args()

import torch, magnumnp
if a.device == "mps":
    torch.set_default_dtype(torch.float32)          # after import: magnum.np forces float64
    magnumnp.device = torch.device("mps")
    magnumnp.complex_dtype = torch.complex64
    torch.set_default_device("mps")
    sync = torch.mps.synchronize

    # demag.py:161 forces float64 for the demag tensor ("always use double
    # precision") before casting back. Apple GPUs have no float64, so that
    # allocation cannot happen on MPS at all. Build the tensor on the CPU, where
    # magnum.np gets the double precision it asks for, then move it to the GPU.
    # Without this, magnum.np cannot compute a demag field on Apple hardware.
    import magnumnp.field_terms.demag as _demag
    _orig_init_N = _demag.DemagField._init_N

    def _init_N_on_cpu(self, state):
        torch.set_default_device("cpu")
        try:
            N = _orig_init_N(self, state)
        finally:
            torch.set_default_device("mps")
        return [[t.to("mps") for t in row] for row in N]

    _demag.DemagField._init_N = _init_N_on_cpu
else:
    sync = lambda: None

from magnumnp import Mesh, State, DemagField, ExchangeField, LLGSolver, Heun, normalize
import magnumnp.common.logging as L
L.set_log_level(40)

n = a.n
mesh = Mesh((n, n, 1), (4e-9, 4e-9, 4e-9))
state = State(mesh)
state.material = {"Ms": 8e5, "A": 1.3e-11, "alpha": 0.02}
m = torch.zeros(n, n, 1, 3)
m[..., 0] = 1.0
m[..., 1] = 0.1
state.m = m / m.norm(dim=-1, keepdim=True)
state.t = 0.0

llg = LLGSolver([DemagField(), ExchangeField()], solver=Heun)

def heun(dt):
    """One Heun step on magnum.np's own LLG right hand side."""
    t0, m0 = state.t, state.m.clone()
    k1 = llg.dm(t0, m0, state)
    k2 = llg.dm(t0 + dt, m0 + dt * k1, state)
    state.m = m0 + dt * (k1 + k2) / 2.0
    state.t = t0 + dt
    normalize(state.m)

for _ in range(2):        # warm up: FFT plans, kernel assembly, lazy init
    heun(a.dt)
sync()

t0 = time.time()
for _ in range(a.steps):
    heun(a.dt)
sync()
wall = time.time() - t0

cells = n * n
print(f"RESULT {cells} {a.steps} {wall:.6f} {state.m[..., 0].mean().item():.8f} "
      f"{cells*2*a.steps/wall:.6e}")
