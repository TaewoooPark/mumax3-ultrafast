#!/usr/bin/env python3
"""Build the auditable Apple-Silicon proxy model used by the SVG charts.

Only one number in this file is a MuMax3 benchmark: the 10-GPU-core M4 in a
fanless MacBook Air.  Every other Apple result is a *model estimate*.  The
model deliberately does not pretend that memory bandwidth or core count alone
predicts this Metal backend.  It combines distinct public proxy families
(which are not assumed statistically independent):

* direct Metal STREAM bandwidth (closest to the memory-heavy part of MuMax3),
* token generation from one fixed llama.cpp build (memory-heavy),
* prompt processing from that same build (a compute-heavy counterpoint),
* published unified-memory bandwidth and GPU width (structural endpoints), and
* a low-weight MLX compiled-SumAll cross-generation check for M5.

Ratios are combined in log space with a weighted Huber location.  Generation
and within-generation tier effects are estimated separately, which lets the
sparse direct Metal measurements contribute without inventing missing points.
The reported low/high interval is the envelope of coherent single-proxy paths,
the full model, and leave-one-proxy-family-out fits.  It is a workload/model
sensitivity range, NOT a confidence interval and NOT a benchmark error bar.
A separate degree-2/degree-3 fit to the four base-chip generations was rejected:
with so few points it reversed or exploded, putting M5/M4 anywhere from 0.82 to
2.31.  No polynomial-extrapolation output enters the chart.

Primary inputs and provenance
-----------------------------
Metal STREAM / architecture study:
  https://arxiv.org/abs/2502.05317
  https://github.com/Arraying/AppleSilicons
Direct Metal STREAM validation for M4 Pro and M3 Ultra (DaMoN 2026):
  https://db.in.tum.de/~beischl/papers/Evaluating_Apple_Silicon_for_Data_Processing.pdf
Same-build llama.cpp table and pinned build:
  https://github.com/ggml-org/llama.cpp/discussions/4167
  https://github.com/ggml-org/llama.cpp/commit/8e672efe632bb6a7333964a255c4b96f018b9a65
Pinned MLX table used only for the M5 generation prior:
  https://github.com/TristanBilot/mlx-benchmark/blob/fc7b2fa714bc8109a3b36f21ed091e542e35a728/benchmarks/average_benchmark.md
Apple configuration/specification references:
  https://support.apple.com/en-us/121554
  https://support.apple.com/en-us/126318

Protocol for the MuMax3 anchor
------------------------------
``bench/bench.mx3`` was run at 2048 x 2048 with solver 2 (Heun).  Heun performs
two torque evaluations per timed step, hence:

    2048 * 2048 * 100 * 2 / median_wall_seconds

Seven fresh processes after warm-up gave a 7.926141958 s median.  The within-
session throughput range was 1.05430e8..1.08134e8 cell-evals/s (CV 0.91%).
That repeatability range is intentionally not mixed with the proxy envelope.

The seven wall times behind the current median, in run order, were 7.757570834,
7.956545250, 7.846818666, 7.926141958, 7.950234208, 7.928319375 and 7.911491000
seconds.  An immediately preceding set of seven in the same session gave a
7.884222250 s median, so the spread between whole sets is about 0.5% and no
single set should be read as more than that precise.

The previous anchor was 7.99095925 s (1.0497623e8 cell-evals/s), measured before
the GPU keep-alive, the FFT tensor-view cache, speculative stepping and the
device-resident minimizer step.  The 0.8% difference is inside the 0.91% CV of a
single set, which is the expected result: this operating point is bandwidth-bound
with rare drains, so none of those changes move it.  They were measured on
latency-bound workloads instead, where the same binary is 1.6x to 2.2x faster.
The anchor is refreshed here only because it is a fresh measurement of the
shipped build, not because it improved.
"""

import math
import os


HERE = os.path.dirname(os.path.abspath(__file__))

MESH_X = 2048
MESH_Y = 2048
TIMED_STEPS = 100
HEUN_EVALS_PER_STEP = 2
MEDIAN_WALL_SECONDS = 7.926141958
ANCHOR_THROUGHPUT = (
    MESH_X
    * MESH_Y
    * TIMED_STEPS
    * HEUN_EVALS_PER_STEP
    / MEDIAN_WALL_SECONDS
)

# Same fixed llama.cpp build. Tuple order is Q8 PP, Q4 PP, Q8 TG, Q4 TG.
# Q8 and Q4 are correlated measurements, so q_family() collapses them into
# one geometric mean before the proxy is given any weight.
LLAMA = {
    "M1": (117.25, 117.96, 7.91, 14.15),
    "M1 Pro": (270.37, 266.25, 22.34, 36.41),
    "M1 Max": (537.37, 530.06, 40.20, 61.19),
    "M1 Ultra": (1042.95, 1030.04, 59.87, 83.73),
    "M2": (181.40, 179.57, 12.21, 21.91),
    "M2 Pro": (344.50, 341.19, 23.01, 38.86),
    "M2 Max": (677.91, 671.31, 41.83, 65.95),
    "M2 Ultra": (1248.59, 1238.48, 66.64, 94.27),
    "M3": (187.52, 186.75, 12.27, 21.34),
    "M3 Pro": (344.66, 341.67, 17.53, 30.74),
    "M3 Max 30c": (566.40, 567.59, 34.30, 56.58),
    "M3 Max 40c": (757.64, 759.70, 42.75, 66.31),
    "M3 Ultra 60c": (1085.76, 1073.09, 63.55, 88.40),
    "M3 Ultra": (1487.51, 1471.24, 63.93, 92.14),
    "M4": (223.64, 221.29, 13.54, 24.11),
    "M4 Pro 16c": (367.13, 364.06, 30.54, 49.64),
    "M4 Pro": (449.62, 439.78, 30.69, 50.74),
    "M4 Max 32c": (718.56, 713.93, 43.87, 69.95),
    "M4 Max 40c": (891.94, 885.68, 54.05, 83.06),
    "M5": (264.15, 247.68, 16.62, 29.62),
    "M5 Pro 16c": (431.14, 403.19, 35.86, 60.04),
}

# Two independent rows for the 40-core M5 Max in the same public table.
M5_MAX40_RUNS = (
    (1051.59, 987.10, 64.61, 102.93),
    (1054.83, 990.53, 65.11, 104.03),
)

# Direct GPU-side Metal streaming throughput, decimal GB/s.  M4 Pro 20c and
# M3 Ultra 80c are ten-run medians from the peer-reviewed DaMoN paper.
STREAM = {
    "M1": 60.0,
    "M2": 91.0,
    "M3": 92.0,
    "M4": 100.0,
    "M4 Pro": 248.0,
    "M3 Ultra": 738.0,
}

# Published decimal GB/s. Apple did not publish a direct M1-base figure; 67 is
# the secondary value tabulated by Kunkel et al. M3 Ultra uses the 819.2 GB/s
# theoretical value used by DaMoN, rather than rounding it down to 800.
BANDWIDTH = {
    "M1": 67.0,
    "M1 Pro": 200.0,
    "M1 Max": 400.0,
    "M1 Ultra": 800.0,
    "M2": 100.0,
    "M2 Pro": 200.0,
    "M2 Max": 400.0,
    "M2 Ultra": 800.0,
    "M3": 100.0,
    "M3 Pro": 150.0,
    "M3 Max 30c": 300.0,
    "M3 Max 40c": 400.0,
    "M3 Ultra 60c": 819.2,
    "M3 Ultra": 819.2,
    "M4": 120.0,
    "M4 Pro 16c": 273.0,
    "M4 Pro": 273.0,
    "M4 Max 32c": 410.0,
    "M4 Max 40c": 546.0,
    "M5": 153.0,
    "M5 Pro 16c": 307.0,
    "M5 Pro": 307.0,
    "M5 Max 32c": 460.0,
    "M5 Max 40c": 614.0,
}

GPU_CORES = {
    "M1": 8,
    "M1 Pro": 16,
    "M1 Max": 32,
    "M1 Ultra": 64,
    "M2": 10,
    "M2 Pro": 19,
    "M2 Max": 38,
    "M2 Ultra": 76,
    "M3": 10,
    "M3 Pro": 18,
    "M3 Max 30c": 30,
    "M3 Max 40c": 40,
    "M3 Ultra 60c": 60,
    "M3 Ultra": 80,
    "M4": 10,
    "M4 Pro 16c": 16,
    "M4 Pro": 20,
    "M4 Max 32c": 32,
    "M4 Max 40c": 40,
    "M5": 10,
    "M5 Pro 16c": 16,
    "M5 Pro": 20,
    "M5 Max 32c": 32,
    "M5 Max 40c": 40,
}

# Engineering influence weights, not statistical inverse-variance weights.
# The order reflects closeness to this backend; sparse structural proxies are
# deliberately unable to overpower direct Metal and same-build workload data.
PROXY_WEIGHTS = {
    "stream": 4.0,
    "tg": 3.0,
    "pp": 2.0,
    "bw": 1.0,
    "cores": 1.0,
    "mlx": 1.0,
}

# Pinned-table compiled SumAll latency ratio, M4 Max / M5 Max = speed uplift.
# The rows used different MLX patch versions (0.31.2 and 0.31.1), which is why
# this is a weight-1 sanity check rather than primary evidence. It is neither a
# four-operation mean nor a MuMax3 measurement.
M5_MLX_GENERATION = 1.01 / 0.89

GENERATIONS = {
    "M1": "M1",
    "M1 Pro": "M1",
    "M1 Max": "M1",
    "M1 Ultra": "M1",
    "M2": "M2",
    "M2 Pro": "M2",
    "M2 Max": "M2",
    "M2 Ultra": "M2",
    "M3": "M3",
    "M3 Pro": "M3",
    "M3 Max 30c": "M3",
    "M3 Max 40c": "M3",
    "M3 Ultra 60c": "M3",
    "M3 Ultra": "M3",
    "M4": "M4",
    "M4 Pro 16c": "M4",
    "M4 Pro": "M4",
    "M4 Max 32c": "M4",
    "M4 Max 40c": "M4",
    "M5": "M5",
    "M5 Pro 16c": "M5",
    "M5 Pro": "M5",
    "M5 Max 32c": "M5",
    "M5 Max 40c": "M5",
}

OUTPUT_ORDER = list(GENERATIONS)


def geometric_mean(left, right):
    return math.sqrt(left * right)


def q_family(name, kind):
    """Collapse the correlated Q8/Q4 pair into one PP or TG proxy."""
    offset = 0 if kind == "pp" else 2
    values = LLAMA[name]
    return geometric_mean(values[offset], values[offset + 1])


def add_m5_transfers():
    """Fill public-table configurations by transparent same-family transfer."""
    LLAMA["M5 Max 40c"] = tuple(
        geometric_mean(left, right)
        for left, right in zip(*M5_MAX40_RUNS)
    )
    LLAMA["M5 Pro"] = tuple(
        LLAMA["M5 Pro 16c"][index]
        * LLAMA["M4 Pro"][index]
        / LLAMA["M4 Pro 16c"][index]
        for index in range(4)
    )
    LLAMA["M5 Max 32c"] = tuple(
        LLAMA["M5 Max 40c"][index]
        * LLAMA["M4 Max 32c"][index]
        / LLAMA["M4 Max 40c"][index]
        for index in range(4)
    )


def weighted_median(values, weights):
    ordered = sorted(zip(values, weights))
    threshold = sum(weights) / 2.0
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    raise AssertionError("non-empty positive weights must have a median")


def robust_log_center(ratios):
    """Weighted Huber location in log-ratio space."""
    if not ratios:
        raise ValueError("at least one proxy ratio is required")
    logs = [math.log(value) for value in ratios.values()]
    weights = [PROXY_WEIGHTS[name] for name in ratios]
    center = weighted_median(logs, weights)
    mad = weighted_median([abs(value - center) for value in logs], weights)
    scale = max(math.log(1.05), 1.4826 * mad)
    for _ in range(100):
        effective = []
        for value, weight in zip(logs, weights):
            distance = abs(value - center) / scale
            effective.append(
                weight if distance <= 1.5 else weight * 1.5 / distance
            )
        updated = sum(
            weight * value for weight, value in zip(effective, logs)
        ) / sum(effective)
        if abs(updated - center) < 1e-12:
            center = updated
            break
        center = updated
    return math.exp(center)


def proxy_ratios(chip):
    """Return generation and within-generation proxy ratio dictionaries."""
    base = GENERATIONS[chip]
    generation = {
        "tg": q_family(base, "tg") / q_family("M4", "tg"),
        "pp": q_family(base, "pp") / q_family("M4", "pp"),
        "bw": BANDWIDTH[base] / BANDWIDTH["M4"],
        "cores": GPU_CORES[base] / GPU_CORES["M4"],
    }
    tier = {
        "tg": q_family(chip, "tg") / q_family(base, "tg"),
        "pp": q_family(chip, "pp") / q_family(base, "pp"),
        "bw": BANDWIDTH[chip] / BANDWIDTH[base],
        "cores": GPU_CORES[chip] / GPU_CORES[base],
    }
    if base in STREAM:
        generation["stream"] = STREAM[base] / STREAM["M4"]
    if chip in STREAM and base in STREAM:
        tier["stream"] = STREAM[chip] / STREAM[base]
    if base == "M5":
        generation["mlx"] = M5_MLX_GENERATION
    return generation, tier


def model_ratio(chip):
    """Return central ratio and workload/model envelope relative to M4."""
    if chip == "M4":
        return 1.0, 1.0, 1.0

    generation, tier = proxy_ratios(chip)
    central = robust_log_center(generation) * robust_log_center(tier)

    # Coherent paths never splice (for example) a STREAM generation ratio to a
    # core-count tier ratio. Each candidate follows one family end to end.
    candidates = [central]
    candidates.extend(
        generation[family] * tier[family]
        for family in generation.keys() & tier.keys()
    )

    # Leave-one-proxy-family-out sensitivity. A family is removed from both the
    # generation and tier stages to avoid retaining half of a correlated path.
    for family in PROXY_WEIGHTS:
        reduced_generation = dict(generation)
        reduced_tier = dict(tier)
        reduced_generation.pop(family, None)
        reduced_tier.pop(family, None)
        if reduced_generation and reduced_tier:
            candidates.append(
                robust_log_center(reduced_generation)
                * robust_log_center(reduced_tier)
            )
    return central, min(candidates), max(candidates)


def display_name(chip):
    family = chip
    final_token = family.rsplit(" ", 1)[-1]
    if final_token.endswith("c") and final_token[:-1].isdigit():
        family = family.rsplit(" ", 1)[0]
    return f"Apple {family} {GPU_CORES[chip]}c"


def build_rows():
    add_m5_transfers()
    rows = []
    for chip in OUTPUT_ORDER:
        central, low, high = model_ratio(chip)
        rows.append(
            {
                "chip": chip,
                "name": display_name(chip),
                "bandwidth": BANDWIDTH[chip],
                "cores": GPU_CORES[chip],
                "value": ANCHOR_THROUGHPUT * central,
                "low": ANCHOR_THROUGHPUT * low,
                "high": ANCHOR_THROUGHPUT * high,
                "status": "measured" if chip == "M4" else "modeled",
            }
        )
    return rows


def verify(rows):
    """Catch unit, protocol, input, and accidental model changes."""
    recorded = 1.0583469289915033e08
    assert math.isclose(ANCHOR_THROUGHPUT, recorded, rel_tol=0, abs_tol=0.05)
    assert set(LLAMA) == set(GENERATIONS)
    assert set(BANDWIDTH) == set(GENERATIONS)
    assert set(GPU_CORES) == set(GENERATIONS)
    by_chip = {row["chip"]: row for row in rows}
    # The model is linear in the anchor, so refreshing the anchor moves every
    # projection by exactly one scalar. These values are the previously audited
    # ones multiplied by 7.99095925 / 7.926141958 = 1.0081776597, not values read
    # back out of the model, so the check still catches a structural change.
    expected_millions = {
        "M1": 61.8,
        "M3 Ultra 60c": 525.3,
        "M3 Ultra": 737.5,
        "M4 Pro 16c": 210.5,
        "M4 Pro": 237.2,
        "M4 Max 32c": 336.2,
        "M4 Max 40c": 417.3,
        "M5": 124.2,
        "M5 Pro": 254.1,
        "M5 Max 32c": 386.5,
        "M5 Max 40c": 479.8,
    }
    for chip, expected in expected_millions.items():
        actual = by_chip[chip]["value"] / 1e6
        assert abs(actual - expected) < 0.051, (chip, actual, expected)
    anchor = by_chip["M4"]
    assert anchor["value"] == anchor["low"] == anchor["high"]


def main():
    rows = build_rows()
    verify(rows)
    lines = [
        "# Apple-Silicon MuMax3 proxy ensemble, anchored to one measurement.",
        "# Generated by bench/apple_project.py; do not edit numeric rows.",
        "#",
        "# measured: MacBook Air M4 10-GPU-core, 32 GB, fanless; 2048x2048,",
        "# solver 2 (Heun), 100 timed steps, seven fresh-process median.",
        "# Historical raw run times were not retained; only the median and",
        "# summary range/CV survive, so do not treat them as rederived here.",
        "# Modeled rows are not MuMax3 benchmarks. low..high is a workload/model",
        "# envelope (coherent proxy paths + leave-one-family-out), not a CI.",
        "# See the generator docstring for model definition and source URLs.",
        "#",
        "# columns: bandwidth_GB_s  gpu_cores  central  low  high  status  \"name\"",
    ]
    for row in rows:
        lines.append(
            f'{row["bandwidth"]:.10g}  {row["cores"]}  '
            f'{row["value"]:.17g}  {row["low"]:.17g}  '
            f'{row["high"]:.17g}  {row["status"]}  "{row["name"]}"'
        )
    path = os.path.join(HERE, "apple.txt")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"wrote {path}")
    for row in rows:
        marker = "measured" if row["status"] == "measured" else "model"
        print(
            f'{row["name"]}: {row["value"] / 1e6:.1f} '
            f'[{row["low"] / 1e6:.1f}, {row["high"] / 1e6:.1f}] M/s '
            f'({marker})'
        )


if __name__ == "__main__":
    main()
