#!/usr/bin/env python3
"""Energy per cell-evaluation for each simulator, measured on one machine.

This is the axis that could not be settled against NVIDIA: every published
comparison there divides measured throughput by a nominal board TDP, and the
answer moves by 3x depending on what fraction of TDP the card actually draws.

Restricting the question to Apple Silicon removes that problem entirely. Every
arm runs on the same chip, and powermetrics reports the real GPU, CPU and
combined rails while the workload runs. Nothing is assumed.

Protocol: one mesh, the same fixed number of Heun steps for every arm, so the
work is identical; powermetrics samples at 2 Hz for the whole run; energy is mean
power times the measured wall time; efficiency is cell-evaluations per joule.

An idle capture is taken first and reported alongside, because the machine is not
at zero when nothing is running - a browser or Spotlight on the GPU shows up in
the same rails. The raw captures are kept so the subtraction can be checked.

Needs sudo: Apple exposes GPU power only through powermetrics.
"""
import json, os, re, statistics, subprocess, sys, time

CT = os.path.dirname(os.path.abspath(__file__))
MW = os.environ.get("MUMAX_WORK",
     "/Users/taewoopark/personal/mumax3-for-mac/mumax3-ultrafast/bench/capacity-20260803-002617/work")
JULIA = "/opt/homebrew/bin/julia"
PY = os.path.join(CT, "venv", "bin", "python")
OUT = os.path.join(CT, "energy")
N = int(os.environ.get("ENERGY_N", "512"))
STEPS = int(os.environ.get("ENERGY_STEPS", "2000"))


def mumax3_cmd(binary):
    src = f"""setcellsize(4e-9, 4e-9, 4e-9)
setgridsize({N}, {N}, 1)
msat  = 800e3
aex   = 13e-12
alpha = 0.02
setsolver(2)
fixdt = 1e-13
m = uniform(1, 0.1, 0)
steps(2)
steps({STEPS})
print("RESULT done")
"""
    scr = os.path.join(CT, "_energy.mx3"); open(scr, "w").write(src)
    od = os.path.join(CT, "_oden"); subprocess.run(f"rm -rf {od}", shell=True)
    return f"{binary} -s -f -http '' -o {od} {scr}"


def oommf_cmd():
    tpl = open(os.path.join(CT, "bench_oommf.mif")).read()
    src = (tpl.replace("NNN", str(N)).replace("TTT", f"{STEPS*1e-13:.6e}"))
    os.makedirs(os.path.join(CT, "oommf_run"), exist_ok=True)
    open(os.path.join(CT, "oommf_run", "energy.mif"), "w").write(src)
    return f"cd {CT}/oommf && tclsh oommf.tcl boxsi -threads 8 -kill all ../oommf_run/energy.mif"


# OOMMF has no internal timer and its process start is counted here, so the step
# count is large enough (see ENERGY_STEPS) that start-up is a few percent of the
# window rather than a quarter of it.
ARMS = [
    ("mumax3-ultrafast",      lambda: mumax3_cmd(f"{MW}/ultrafast")),
    ("mumax3-for-mac",        lambda: mumax3_cmd(f"{MW}/formac-bin")),
    ("OOMMF CPU 8t",          oommf_cmd),
    ("magnum.np MPS",         lambda: f"{PY} {CT}/bench_magnumnp.py --n {N} --steps {STEPS} --device mps"),
    ("magnum.np CPU",         lambda: f"{PY} {CT}/bench_magnumnp.py --n {N} --steps {STEPS} --device cpu"),
    ("MicroMagnetic.jl CPU",  lambda: f"{JULIA} {CT}/bench_micromagnetic.jl {N} {STEPS} 1e-13 cpu"),
]


def pm_start(path):
    f = open(path, "w")
    p = subprocess.Popen(["sudo", "powermetrics", "--samplers", "gpu_power,cpu_power", "-i", "500"],
                         stdout=f, stderr=subprocess.DEVNULL)
    time.sleep(2)
    return p, f


def pm_stop(p, f):
    subprocess.run(["sudo", "pkill", "-x", "powermetrics"], capture_output=True)
    try: p.wait(timeout=10)
    except Exception: p.kill()
    f.close()


def pm_mean(path):
    txt = open(path, errors="ignore").read()
    out = {}
    for key, label in (("GPU Power", "gpu"), ("CPU Power", "cpu"), ("Combined Power", "combined")):
        v = [float(m) / 1000 for m in re.findall(rf"{re.escape(key)}[^:]*:\s*([0-9.]+)\s*mW", txt)]
        if v: out[label] = statistics.mean(v)
    r = [float(m) for m in re.findall(r"GPU HW active residency:\s*([0-9.]+)", txt)]
    if r: out["residency_median"] = statistics.median(r)
    return out


def main():
    if subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode != 0:
        print("sudo is not cached. Run  sudo -v  first, then re-run this.", file=sys.stderr)
        sys.exit(1)
    os.makedirs(OUT, exist_ok=True)
    e = dict(os.environ); e["PATH"] = "/opt/homebrew/bin:" + e.get("PATH", "")
    res = {}

    p, f = pm_start(os.path.join(OUT, "idle.txt")); time.sleep(20); pm_stop(p, f)
    res["idle"] = pm_mean(os.path.join(OUT, "idle.txt"))
    print(f"idle: {res['idle']}")

    cells = N * N
    print(f"\n{'arm':24} {'wall s':>8} {'GPU W':>7} {'CPU W':>7} {'comb W':>7} "
          f"{'J':>9} {'M cell-evals/J':>15}")
    for name, mk in ARMS:
        cap = os.path.join(OUT, name.replace(" ", "_").replace(".", "") + ".txt")
        p, f = pm_start(cap)
        t0 = time.time()
        subprocess.run(mk(), shell=True, env=e, capture_output=True, timeout=3600)
        wall = time.time() - t0
        pm_stop(p, f)
        m = pm_mean(cap)
        comb = m.get("combined", 0.0)
        evals = cells * 2 * STEPS
        j = comb * wall
        res[name] = dict(wall=wall, evals=evals, **m, joules=j,
                         m_evals_per_joule=(evals / j / 1e6) if j else 0)
        print(f"{name:24} {wall:8.2f} {m.get('gpu',0):7.2f} {m.get('cpu',0):7.2f} "
              f"{comb:7.2f} {j:9.1f} {evals/j/1e6 if j else 0:15.2f}", flush=True)
        json.dump(res, open(os.path.join(CT, "results_energy.json"), "w"), indent=1)
    print(f"\nmesh {N}^2, {STEPS} Heun steps, {cells*2*STEPS:.3e} cell-evaluations per arm")


if __name__ == "__main__":
    main()
