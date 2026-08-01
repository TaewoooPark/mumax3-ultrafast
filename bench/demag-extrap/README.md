# Demagnetizing-field extrapolation A/B benchmark

This harness compares the exact demagnetizing-field path with the experimental
matched-order polynomial extrapolation path in separate processes.

```sh
go build -o /tmp/mumax3-demag-extrap ./cmd/mumax3
python3 bench/demag-extrap/run_ab.py \
  --mumax /tmp/mumax3-demag-extrap \
  --out /tmp/mumax3-demag-extrap-results
```

The performance case uses fixed-dt Dormand--Prince and intentionally performs
no per-step table or field output. Such output evaluates `B_demag` exactly by
design and would measure I/O/query policy rather than solver acceleration.

The accuracy case records average magnetization and total energy at every
fixed step. `report.json` contains:

- median exact and extrapolated wall time and their ratio;
- exact/extrapolated demag evaluation counts;
- maximum and RMS absolute error of the `(mx,my,mz)` trajectory;
- maximum/RMS absolute total-energy error, normalized by the trajectory's
  maximum absolute reference energy;
- pointwise relative energy error as a diagnostic (it can spike when total
  energy crosses zero and should not be used alone as a merge gate).

This is an A/B gate, not a universal accuracy proof. Repeat it with the target
geometry, cell size, material parameters, field/current drive and time-step
range before using extrapolation for production results. The implementation is
opt-in and automatically fails closed for finite temperature, time-dependent
`Msat`, `NoDemagSpins`, post-step callbacks, relaxation/minimization and
unsupported solvers. The production default also requires solver order 4 or
higher: only solver 4 (RK4), 5 (Dormand--Prince), and 6 (Fehlberg) are allowed.
That guard is deliberately not user-tunable because coarse-step Heun testing
was unstable. `validate_matrix.py` exercises all three allowed solvers.

Additional checks:

```sh
python3 bench/demag-extrap/run_std4_adaptive.py --mumax /tmp/mumax3-demag-extrap
python3 bench/demag-extrap/validate_matrix.py --mumax /tmp/mumax3-demag-extrap
python3 bench/demag-extrap/run_stress_ab.py --mumax /tmp/mumax3-demag-extrap
```

`run_stress_ab.py` compares exact and extrapolated trajectories under stepwise
field/current discontinuities, a fast sinusoidal field/current drive, adaptive
step rejection and rapid `dt` changes. It also runs identical extrapolated
trajectories with and without per-step exact `B_demag`/`E_demag` queries and
checks that solver counters and final fields are unchanged.

Method reference: S. Lepadatu, “Speeding Up Explicit Numerical Evaluation
Methods for Micromagnetic Simulations Using Demagnetizing Field Polynomial
Extrapolation,” IEEE Transactions on Magnetics 58 (2022),
doi:10.1109/TMAG.2022.3159849.
