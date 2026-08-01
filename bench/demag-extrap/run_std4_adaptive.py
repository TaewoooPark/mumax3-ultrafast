#!/usr/bin/env python3
"""Adaptive µMAG standard-problem-4 A/B end-state and timing check."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import struct
import subprocess
import tempfile
from pathlib import Path


RESULT_RE = re.compile(
    r"DEMAG_STD4_RESULT\s+wall\s+([0-9.eE+-]+)\s+steps\s+([0-9]+)"
    r"\s+m\s+\[([0-9.eE+\- ]+)\]\s+energy\s+([0-9.eE+-]+)"
    r"\s+exact\s+([0-9]+)\s+extrapolated\s+([0-9]+)"
)


def script_text(enabled: bool, duration: float) -> str:
    return f"""// Adaptive µMAG standard problem 4(a), generated A/B case.
SetGridSize(128, 32, 1)
SetCellSize(500e-9/128, 125e-9/32, 3e-9)
Msat = 1600e3
Aex = 13e-12
E_total.get()
Msat = 800e3
alpha = 0.02
m = uniform(1, 0.1, 0)
Relax()
B_ext = vector(-24.6e-3, 4.3e-3, 0)
DemagExtrapolation = {str(enabled).lower()}
steps0 := step
start := now()
Run({duration:.17g})
wall := since(start).Seconds()
energy := E_total.get()
Save(m)
print("DEMAG_STD4_RESULT", "wall", wall, "steps", step-steps0, "m", m.average(), "energy", energy, "exact", GetDemagExactEvals(), "extrapolated", GetDemagExtrapolatedEvals())
"""


def run(binary: Path, root: Path, label: str, enabled: bool, duration: float) -> dict[str, object]:
    script = root / f"{label}.mx3"
    out = root / f"{label}.out"
    script.write_text(script_text(enabled, duration))
    proc = subprocess.run(
        [str(binary), "-f", "-o", str(out), str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (root / f"{label}.log").write_text(proc.stdout)
    if proc.returncode:
        raise RuntimeError(f"{label} failed ({proc.returncode}); see {root / f'{label}.log'}")
    match = RESULT_RE.search(proc.stdout)
    if not match:
        raise RuntimeError(f"missing result marker for {label}; see {root / f'{label}.log'}")
    return {
        "wall_s": float(match.group(1)),
        "steps": int(match.group(2)),
        "m": [float(value) for value in match.group(3).split()],
        "energy_J": float(match.group(4)),
        "exact": int(match.group(5)),
        "extrapolated": int(match.group(6)),
        "out": str(out),
    }


def read_binary4_ovf(path: Path) -> list[tuple[float, float, float]]:
    raw = path.read_bytes()
    marker = b"# Begin: Data Binary 4\n"
    start = raw.index(marker) + len(marker)
    control = struct.unpack_from("=f", raw, start)[0]
    if control != 1234567.0:
        raise RuntimeError(f"unexpected OVF control number in {path}: {control}")
    end = raw.index(b"# End: Data Binary 4", start + 4)
    payload = raw[start + 4 : end]
    values = struct.unpack(f"={len(payload) // 4}f", payload)
    if len(values) % 3:
        raise RuntimeError(f"non-vector OVF payload in {path}")
    return list(zip(values[0::3], values[1::3], values[2::3]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mumax", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--duration", type=float, default=1e-9)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    root = args.out or Path(tempfile.mkdtemp(prefix="mumax3-demag-std4-ab."))
    root.mkdir(parents=True, exist_ok=True)
    binary = args.mumax.resolve()

    all_results: dict[str, list[dict[str, object]]] = {"exact": [], "extrap": []}
    for repeat in range(args.repeats):
        for enabled, label in ((False, "exact"), (True, "extrap")):
            all_results[label].append(run(binary, root, f"{label}_{repeat}", enabled, args.duration))

    exact = all_results["exact"][-1]
    extrap = all_results["extrap"][-1]
    m_error = math.sqrt(sum((a - b) ** 2 for a, b in zip(exact["m"], extrap["m"])))
    energy_abs = abs(float(exact["energy_J"]) - float(extrap["energy_J"]))
    exact_field = read_binary4_ovf(Path(str(exact["out"])) / "m000000.ovf")
    extrap_field = read_binary4_ovf(Path(str(extrap["out"])) / "m000000.ovf")
    if len(exact_field) != len(extrap_field):
        raise RuntimeError("final OVF grids differ")
    field_errors = [
        math.sqrt(sum((a - b) ** 2 for a, b in zip(exact_cell, extrap_cell)))
        for exact_cell, extrap_cell in zip(exact_field, extrap_field)
    ]
    exact_times = [float(result["wall_s"]) for result in all_results["exact"]]
    extrap_times = [float(result["wall_s"]) for result in all_results["extrap"]]
    report = {
        "duration_s": args.duration,
        "exact": all_results["exact"],
        "extrap": all_results["extrap"],
        "median_exact_s": statistics.median(exact_times),
        "median_extrap_s": statistics.median(extrap_times),
        "speedup": statistics.median(exact_times) / statistics.median(extrap_times),
        "final_average_m_abs_error": m_error,
        "final_energy_abs_error_J": energy_abs,
        "final_energy_rel_error": energy_abs / max(abs(float(exact["energy_J"])), 1e-30),
        "final_field_mean_abs_error": statistics.fmean(field_errors),
        "final_field_rms_abs_error": math.sqrt(statistics.fmean(value * value for value in field_errors)),
        "final_field_max_abs_error": max(field_errors, default=0.0),
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"artifacts: {root}")


if __name__ == "__main__":
    main()
