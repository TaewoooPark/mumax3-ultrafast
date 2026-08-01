#!/usr/bin/env python3
"""A/B stress checks for discontinuous drives, adaptive rejection, and output queries."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path


RESULT_RE = re.compile(
    r"DEMAG_STRESS_RESULT\s+wall\s+([0-9.eE+-]+)\s+steps\s+([0-9]+)"
    r"\s+m\s+\[([0-9.eE+\- ]+)\]\s+energy\s+([0-9.eE+-]+)"
    r"\s+dt\s+([0-9.eE+-]+)\s+exact\s+([0-9]+)"
    r"\s+extrapolated\s+([0-9]+)\s+rejected\s+([0-9]+)\s+nevals\s+([0-9]+)"
)


def load_helpers():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("demag_ab_helpers", here / "run_ab.py")
    assert spec and spec.loader
    ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ab)
    spec = importlib.util.spec_from_file_location("demag_ovf_helpers", here / "run_std4_adaptive.py")
    assert spec and spec.loader
    ovf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ovf)
    return ab.compare_trajectories, ovf.read_binary4_ovf


COMPARE_TRAJECTORIES, READ_OVF = load_helpers()


def common_input(enabled: bool) -> str:
    return f"""SetGridSize(64, 32, 1)
SetCellSize(4e-9, 4e-9, 4e-9)
SetGeom(Ellipse(240e-9, 112e-9))
Msat = 800e3
Aex = 13e-12
alpha = 0.04
xi = 0.05
Pol = 0.65
m = vortex(1, 1)
SetSolver(5)
DemagExtrapolation = {str(enabled).lower()}
"""


def script_text(
    *,
    scenario: str,
    enabled: bool,
    drive_file: Path,
    with_output_query: bool = False,
) -> str:
    common = common_input(enabled)
    drive_path = str(drive_file.resolve()).replace("\\", "\\\\").replace('"', '\\"')
    if scenario == "discontinuous_drive":
        drive = f"""fieldDrive := FunctionFromDatafile("{drive_path}", 0, 1, "step")
currentDrive := FunctionFromDatafile("{drive_path}", 0, 2, "step")
B_ext = vector(8e-3, fieldDrive(t), 0)
J = vector(currentDrive(t), 0, 0)
FixDt = 5e-14
TableAdd(E_total)
TableAutoSave(FixDt)
start := now()
Steps(600)
"""
    elif scenario == "fast_time_drive":
        drive = """f := 250e9
B_ext = vector(35e-3*sin(2*pi*f*t), 25e-3*cos(2*pi*f*t), 5e-3*sin(4*pi*f*t))
J = vector(7e11*sin(2*pi*f*t), 0, 0)
FixDt = 2e-14
TableAdd(E_total)
TableAutoSave(FixDt)
start := now()
Steps(600)
"""
    elif scenario == "adaptive_rejection":
        drive = f"""fieldDrive := FunctionFromDatafile("{drive_path}", 0, 1, "step")
currentDrive := FunctionFromDatafile("{drive_path}", 0, 2, "step")
B_ext = vector(10e-3, fieldDrive(t), 0)
J = vector(currentDrive(t), 0, 0)
FixDt = 0
MinDt = 1e-16
MaxDt = 2e-12
MaxErr = 2e-8
start := now()
Run(30e-12)
"""
    elif scenario == "output_query":
        query = "TableAdd(B_demag)\nTableAdd(E_demag)\nTableAutoSave(FixDt)\n" if with_output_query else ""
        drive = f"""B_ext = vector(-20e-3, 4e-3, 1e-3)
FixDt = 1e-13
{query}start := now()
Steps(80)
"""
    else:
        raise ValueError(scenario)

    return common + drive + """wall := since(start).Seconds()
energy := E_total.get()
Save(m)
print("DEMAG_STRESS_RESULT", "wall", wall, "steps", step, "m", m.average(), "energy", energy, "dt", dt.get(), "exact", GetDemagExactEvals(), "extrapolated", GetDemagExtrapolatedEvals(), "rejected", GetDemagRejectedAttempts(), "nevals", NEval.get())
"""


def run(binary: Path, root: Path, label: str, text: str) -> dict[str, object]:
    script = root / f"{label}.mx3"
    out = root / f"{label}.out"
    script.write_text(text)
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
        "final_dt_s": float(match.group(5)),
        "exact": int(match.group(6)),
        "extrapolated": int(match.group(7)),
        "rejected": int(match.group(8)),
        "nevals": int(match.group(9)),
        "out": str(out),
    }


def compare_fields(a: Path, b: Path) -> dict[str, float]:
    av = READ_OVF(a)
    bv = READ_OVF(b)
    if len(av) != len(bv):
        raise RuntimeError("OVF grids differ")
    errors = [math.sqrt(sum((x - y) ** 2 for x, y in zip(ac, bc))) for ac, bc in zip(av, bv)]
    return {
        "field_mean_abs_error": sum(errors) / max(len(errors), 1),
        "field_rms_abs_error": math.sqrt(sum(x * x for x in errors) / max(len(errors), 1)),
        "field_max_abs_error": max(errors, default=0.0),
    }


def compare_results(exact: dict[str, object], extrap: dict[str, object]) -> dict[str, float]:
    m_error = math.sqrt(sum((a - b) ** 2 for a, b in zip(exact["m"], extrap["m"])))
    energy_abs = abs(float(exact["energy_J"]) - float(extrap["energy_J"]))
    return {
        "average_m_abs_error": m_error,
        "energy_abs_error_J": energy_abs,
        "energy_rel_error": energy_abs / max(abs(float(exact["energy_J"])), 1e-30),
        **compare_fields(
            Path(str(exact["out"])) / "m000000.ovf",
            Path(str(extrap["out"])) / "m000000.ovf",
        ),
    }


def annotate_dp_rejections(result: dict[str, object], *, enabled: bool) -> None:
    # Dormand--Prince uses FSAL on the exact path (1 + 6 evaluations per
    # attempt) and deliberately refreshes k1 on the extrapolated path (7 per
    # attempt). This independent inference cross-checks the manager counter.
    nevals = int(result["nevals"])
    attempts = nevals // 7 if enabled else max(nevals - 1, 0) // 6
    result["inferred_rejected_attempts"] = attempts - int(result["steps"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mumax", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.out or Path(tempfile.mkdtemp(prefix="mumax3-demag-stress."))
    root.mkdir(parents=True, exist_ok=True)
    binary = args.mumax.resolve()

    drive_file = root / "step_drive.tsv"
    drive_file.write_text(
        "0,0,0\n"
        "5e-12,0.12,8e11\n"
        "10e-12,-0.10,-8e11\n"
        "15e-12,0.08,0\n"
        "20e-12,-0.06,5e11\n"
        "25e-12,0,0\n"
        "30e-12,0,0\n"
    )

    report: dict[str, object] = {"scenarios": {}}
    for scenario in ("discontinuous_drive", "fast_time_drive", "adaptive_rejection"):
        exact = run(binary, root, f"{scenario}_exact", script_text(
            scenario=scenario, enabled=False, drive_file=drive_file
        ))
        extrap = run(binary, root, f"{scenario}_extrap", script_text(
            scenario=scenario, enabled=True, drive_file=drive_file
        ))
        annotate_dp_rejections(exact, enabled=False)
        annotate_dp_rejections(extrap, enabled=True)
        scenario_report: dict[str, object] = {
            "exact": exact,
            "extrap": extrap,
            **compare_results(exact, extrap),
        }
        if scenario != "adaptive_rejection":
            scenario_report.update(COMPARE_TRAJECTORIES(
                Path(str(exact["out"])) / "table.txt",
                Path(str(extrap["out"])) / "table.txt",
            ))
        report["scenarios"][scenario] = scenario_report

    # Run the same extrapolated trajectory with and without exact output-side
    # B_demag/E_demag queries. Equal solver counters and end states demonstrate
    # that those queries neither consume nor append extrapolation history.
    no_query = run(binary, root, "output_query_none", script_text(
        scenario="output_query", enabled=True, drive_file=drive_file, with_output_query=False
    ))
    with_query = run(binary, root, "output_query_exact", script_text(
        scenario="output_query", enabled=True, drive_file=drive_file, with_output_query=True
    ))
    annotate_dp_rejections(no_query, enabled=True)
    annotate_dp_rejections(with_query, enabled=True)
    report["output_query_nonpollution"] = {
        "without_queries": no_query,
        "with_queries": with_query,
        "solver_counters_equal": (
            no_query["exact"] == with_query["exact"]
            and no_query["extrapolated"] == with_query["extrapolated"]
            and no_query["rejected"] == with_query["rejected"]
        ),
        **compare_results(no_query, with_query),
    }

    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"artifacts: {root}")


if __name__ == "__main__":
    main()
