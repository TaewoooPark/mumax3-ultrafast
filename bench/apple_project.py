#!/usr/bin/env python3
"""Project MuMax3's 4.19M cell throughput across Apple Silicon.

Writes bench/apple.txt, which bench/apple_svg.py then renders. Only the M4 row
is a measurement; everything else comes out of the model below.

Why not simply scale by quoted bandwidth
----------------------------------------
Quoted bandwidth alone puts M1 Ultra (800 GB/s) ahead of M4 Max (546 GB/s) and
would leave the M5 parts out entirely. The first result is misleading: a GPU can
only pull as much bandwidth as it can keep requests in flight for, and an M1 core
does that less well than an M4 core. Ultra parts have a second problem, in that
their figure is the sum across two dies joined by UltraFusion, which a single
strided workload does not see in full.

The model therefore takes the smaller of two limits.

Limit 1, the memory system:

    spec_bandwidth * MEMORY_EFFICIENCY * die_penalty

MEMORY_EFFICIENCY comes from measured GPU STREAM numbers for the base chips in
Kunkel et al., "Apple vs. Oranges: Evaluating the Apple Silicon M-Series SoCs for
HPC Performance and Efficiency" (arXiv:2502.05317): M1 60/67, M2 91/100,
M3 92/100, M4 100/120, i.e. 90%, 91%, 92% and 83%. The paper concludes that all
of them reach roughly 85% of peak, which is the value used here.

Limit 2, the GPU width:

    gpu_cores * per_core_bandwidth(generation)

per_core_bandwidth is derived from each generation's base chip, where the memory
system is the binding limit and so the achieved figure divided by core count is a
lower bound on what one core can pull. It rises across generations, from
7.25 GB/s on M1 to 13.1 GB/s on M5, which is what stops an old wide part from
being credited with a modern part's efficiency.

Throughput is then the achievable bandwidth times a constant calibrated so the
model reproduces the measured M4 result exactly.

Caveats worth repeating when quoting these numbers
--------------------------------------------------
  - Everything except M4 is unmeasured. Treat the ordering as more trustworthy
    than the absolute values.
  - per_core_bandwidth is a lower bound taken from memory-bound base chips, so
    the wide parts could do better than shown if their cores are not in fact the
    limit.
  - The Ultra die penalty is an estimate, not a measurement.
  - This is the 4.19M cell operating point, where the simulation is bandwidth
    bound. Wider GPUs help the small-mesh, dispatch-bound regime more than they
    help here, so do not read this table as a general speedup ranking.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Re-measured on this machine from final commit 081c74da with the isolated
# 2048x2048 point of bench/bench.mx3 (solver 2): seven consecutive fresh
# processes, each with the benchmark's kernel/solver warm-up. The median was
# 1.0497623298479466e8 cells*evals/s (7.99095925 s for 100 steps); the seven-run
# range was 1.03604e8..1.05704e8 and CV 0.72%. Session-local repetition matters:
# the same binary has historically differed by as much as 15% hours apart.
MEASURED_CHIP = "Apple M4"
MEASURED_THROUGHPUT = 1.0497623298479466e08

# Fraction of quoted bandwidth a GPU streaming kernel reaches (arXiv:2502.05317).
MEMORY_EFFICIENCY = 0.85
# Extra derate for two-die parts, where the quoted figure is the sum of both
# dies and a single strided workload does not see all of it. Estimated.
ULTRA_DIE_PENALTY = 0.85

# name -> (quoted GB/s, GPU cores, generation, is_ultra)
# Core counts are the top configuration of each part.
CHIPS = [
    ("Apple M1",          68.25,  8, "M1", False),
    ("Apple M1 Pro",     200.0,  16, "M1", False),
    ("Apple M1 Max",     400.0,  32, "M1", False),
    ("Apple M1 Ultra",   800.0,  64, "M1", True),
    ("Apple M2",         100.0,  10, "M2", False),
    ("Apple M2 Pro",     200.0,  19, "M2", False),
    ("Apple M2 Max",     400.0,  38, "M2", False),
    ("Apple M2 Ultra",   800.0,  76, "M2", True),
    ("Apple M3",         102.4,  10, "M3", False),
    ("Apple M3 Pro",     153.6,  18, "M3", False),
    ("Apple M3 Max 30c", 300.0,  30, "M3", False),
    ("Apple M3 Max 40c", 409.6,  40, "M3", False),
    ("Apple M3 Ultra",   819.3,  80, "M3", True),
    ("Apple M4",         120.0,  10, "M4", False),
    ("Apple M4 Pro",     273.0,  20, "M4", False),
    ("Apple M4 Max 32c", 410.0,  32, "M4", False),
    ("Apple M4 Max 40c", 546.0,  40, "M4", False),
    ("Apple M5",         153.6,  10, "M5", False),
    ("Apple M5 Pro",     307.0,  20, "M5", False),
    ("Apple M5 Max 32c", 460.0,  32, "M5", False),
    ("Apple M5 Max 40c", 614.0,  40, "M5", False),
]

# Base chip of each generation, used to derive per-core bandwidth.
BASE = {"M1": 68.25, "M2": 100.0, "M3": 102.4, "M4": 120.0, "M5": 153.6}
BASE_CORES = {"M1": 8, "M2": 10, "M3": 10, "M4": 10, "M5": 10}


def per_core_bandwidth(generation):
    return MEMORY_EFFICIENCY * BASE[generation] / BASE_CORES[generation]


def achievable_bandwidth(spec, cores, generation, is_ultra):
    memory_limit = spec * MEMORY_EFFICIENCY * (ULTRA_DIE_PENALTY if is_ultra else 1.0)
    width_limit = cores * per_core_bandwidth(generation)
    return min(memory_limit, width_limit), memory_limit, width_limit


def main():
    reference = next(c for c in CHIPS if c[0] == MEASURED_CHIP)
    reference_bw, _, _ = achievable_bandwidth(*reference[1:])
    scale = MEASURED_THROUGHPUT / reference_bw

    rows = []
    for name, spec, cores, generation, is_ultra in CHIPS:
        bandwidth, memory_limit, width_limit = achievable_bandwidth(
            spec, cores, generation, is_ultra
        )
        rows.append(
            {
                "name": name,
                "spec": spec,
                "cores": cores,
                "bandwidth": bandwidth,
                "limit": "memory" if memory_limit <= width_limit else "gpu-width",
                "throughput": bandwidth * scale,
                "status": "measured" if name == MEASURED_CHIP else "projected",
            }
        )
    rows.sort(key=lambda row: row["throughput"])

    lines = [
        "# MuMax3 throughput at the 4.194304e6 cell (2048x2048) operating point",
        "# of bench/bench.mx3, the same point recorded in bench/gpus.txt.",
        "#",
        "# Generated by bench/apple_project.py. Do not edit by hand; that script",
        "# documents the model and its caveats. Only the Apple M4 row is measured.",
        "#",
        f"# Calibration: {MEASURED_CHIP} measured at {MEASURED_THROUGHPUT:.4g}"
        f" cells*evals/s,",
        f"# giving {scale:.4g} cells*evals/s per GB/s of achievable bandwidth.",
        "#",
        "# columns: spec_GB_s  achievable_GB_s  binding_limit  throughput  status  \"name\"",
    ]
    for row in rows:
        lines.append(
            f'{row["spec"]:7.1f} {row["bandwidth"]:8.1f} {row["limit"]:>9s}'
            f'  {row["throughput"]:.4g}  {row["status"]:9s} "{row["name"]}"'
        )
    path = os.path.join(HERE, "apple.txt")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"wrote {path}")
    print(
        f"{'chip':18s} {'spec':>7s} {'achv':>7s} {'limit':>10s} "
        f"{'M cell-evals/s':>14s}"
    )
    for row in rows:
        print(
            f'{row["name"]:18s} {row["spec"]:7.1f} {row["bandwidth"]:7.1f} '
            f'{row["limit"]:>10s} {row["throughput"]/1e6:10.1f}'
            f'{"   <- measured" if row["status"] == "measured" else ""}'
        )


if __name__ == "__main__":
    main()
