# MicroMagnetic.jl arm of the Apple Silicon cross-tool benchmark.
#
# Same problem as every other arm: n x n x 1 cells of 4 nm, Ms = 8e5 A/m,
# A = 1.3e-11 J/m, alpha = 0.02, demag + exchange only, m0 = normalize(1, 0.1, 0),
# Heun at a fixed 1e-13 s step - the same integrator mumax3 runs as setsolver(2)
# and the same one the magnum.np arm drives, so every arm performs exactly two
# effective-field evaluations per step.
#
# `using Metal` selects the Metal backend on import, so the CPU arm has to ask
# for the CPU back explicitly.

using MicroMagnetic
using Metal
using Printf

n       = parse(Int,     ARGS[1])
nsteps  = parse(Int,     ARGS[2])
dt      = parse(Float64, ARGS[3])
backend = length(ARGS) >= 4 ? ARGS[4] : "metal"

if backend == "metal"
    set_backend("metal")
    MicroMagnetic.set_precision(Float32)   # Apple GPUs have no Float64 at all
else
    set_backend("cpu")
    MicroMagnetic.set_precision(Float64)   # the package's own default on CPU
end

mesh = FDMesh(nx=n, ny=n, nz=1, dx=4e-9, dy=4e-9, dz=4e-9)
sim  = Sim(mesh, driver="LLG", integrator="Heun", save_data=false)
set_Ms(sim, 8e5)
add_exch(sim, 1.3e-11)
add_demag(sim)
sim.driver.alpha = 0.02
sim.driver.gamma = 2.211e5
init_m0(sim, (1.0, 0.1, 0.0))

integ = sim.driver.integrator
integ.step = dt

sync() = backend == "metal" ? Metal.synchronize() : nothing

for _ in 1:2                                   # warm up: FFT plans, demag tensor
    MicroMagnetic.advance_step(sim, integ)
end
sync()

t0 = time()
for _ in 1:nsteps
    MicroMagnetic.advance_step(sim, integ)
end
sync()
wall = time() - t0

spin  = Array(sim.spin)
mx    = sum(@view spin[1:3:end]) / (n * n)
cells = n * n
@printf("RESULT %d %d %.6f %.8f %.6e\n", cells, nsteps, wall, mx, cells * 2 * nsteps / wall)
