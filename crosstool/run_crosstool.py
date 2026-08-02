#!/usr/bin/env python3
"""Cross-tool micromagnetic benchmark on Apple Silicon.

Every arm solves the identical problem: n x n x 1 cells of 4 nm, Ms = 8e5 A/m,
A = 1.3e-11 J/m, alpha = 0.02, demag + exchange only, m0 = normalize(1, 0.1, 0),
Heun (predictor-corrector, 2 effective-field evaluations per step) at a fixed
1e-13 s timestep. The fixed step matters: with adaptive control a tool could
finish sooner by taking different steps, and the comparison would measure solver
policy rather than implementation. Every arm was checked to perform exactly two
field evaluations per step - mumax3 reports Neval = 2*steps, OOMMF reports
energy_calc_count = 2*steps + 1.

Startup is excluded everywhere. mumax3, magnum.np and MicroMagnetic.jl time an
inner loop after a two-step warm-up. OOMMF has no internal timer, so it is run at
N and 2N steps and the per-step cost is taken as the slope, which cancels process
start, problem load and demag tensor construction exactly.

Arms, and why each is configured the way it is:

  mumax3-ultrafast   this fork, HEAD
  mumax3-for-mac     b18bc5ea, the previous release of the same line
  magnum.np MPS      needs three patches to reach the Apple GPU at all: device is
                     cuda-or-cpu in __init__.py, float64 is forced at import, and
                     the demag tensor is built in float64 which Apple GPUs cannot
                     allocate. Patched here so it gets its best case.
  magnum.np CPU      what a Mac user actually gets from pip install, unpatched
  MicroMagnetic.jl   CPU only: its Metal backend cannot compile the demag tensor
                     kernel, because newell_f and the whole Newell path are typed
                     Float64 and Apple GPUs have no double precision
  OOMMF              CPU, 8 threads. No GPU backend exists for any vendor.
"""
import argparse, json, os, re, subprocess, sys, time

CT = os.path.dirname(os.path.abspath(__file__))
MW = os.environ.get("MUMAX_WORK",
     "/Users/taewoopark/personal/mumax3-for-mac/mumax3-ultrafast/bench/capacity-20260803-002617/work")
JULIA = "/opt/homebrew/bin/julia"
PY = os.path.join(CT, "venv", "bin", "python")


def sh(cmd, timeout=3600, cwd=None, env=None):
    e = dict(os.environ); e["PATH"] = "/opt/homebrew/bin:" + e.get("PATH", "")
    if env: e.update(env)
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, cwd=cwd, env=e)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def parse_result(out):
    for ln in out.splitlines():
        if ln.startswith("RESULT"):
            p = ln.split()
            return dict(cells=int(p[1]), steps=int(p[2]), wall=float(p[3]),
                        mx=float(p[4]), rate=float(p[5]))
    return None


# ------------------------------------------------------------------- arms ---

def run_mumax3(binary, n, steps):
    tpl = open(os.path.join(CT, "bench_mumax3.mx3")).read()
    src = tpl.replace("NNN", str(n)).replace("SSS", str(steps))
    scr = os.path.join(CT, "_m3.mx3"); open(scr, "w").write(src)
    od = os.path.join(CT, "_od3")
    sh(f"rm -rf {od}")
    rc, out = sh(f"{binary} -s -f -http '' -o {od} {scr}")
    f = os.path.join(od, "res.txt")
    if rc != 0 or not os.path.exists(f):
        return None, out[-300:]
    p = open(f).read().split()
    return dict(cells=int(float(p[0])), steps=int(float(p[1])), wall=float(p[2]),
                mx=float(p[3]), rate=float(p[4])), ""


def run_magnumnp(n, steps, device):
    rc, out = sh(f"{PY} {CT}/bench_magnumnp.py --n {n} --steps {steps} --device {device}")
    return parse_result(out), out[-300:]


def run_micromagnetic(n, steps, backend):
    rc, out = sh(f"{JULIA} {CT}/bench_micromagnetic.jl {n} {steps} 1e-13 {backend}")
    return parse_result(out), out[-400:]


def run_oommf(n, steps):
    """Slope of wall time against step count, so startup cancels exactly."""
    tpl = open(os.path.join(CT, "bench_oommf.mif")).read()
    walls = {}
    mx = None
    for S in (steps, 2 * steps):
        src = tpl.replace("NNN", str(n)).replace("TTT", f"{S*1e-13:.6e}")
        mif = os.path.join(CT, "oommf_run", "b.mif")
        os.makedirs(os.path.join(CT, "oommf_run"), exist_ok=True)
        open(mif, "w").write(src)
        odt = os.path.join(CT, "oommf_run", "oommf_bench.odt")
        if os.path.exists(odt): os.remove(odt)
        t0 = time.time()
        rc, out = sh("tclsh oommf.tcl boxsi -threads 8 -kill all ../oommf_run/b.mif",
                     cwd=os.path.join(CT, "oommf"))
        walls[S] = time.time() - t0
        if rc != 0: return None, out[-300:]
        if os.path.exists(odt):
            cols, last = None, None
            for ln in open(odt):
                if ln.startswith("# Columns:"):
                    cols = [a or b for a, b in re.findall(r'\{([^}]*)\}|(\S+)', ln[10:])]
                elif not ln.startswith("#") and ln.strip():
                    last = ln.split()
            if cols and last:
                d = dict(zip(cols, last))
                if S == steps:
                    mx = float(d.get("Oxs_TimeDriver::mx", "nan"))
    per_step = (walls[2 * steps] - walls[steps]) / steps
    if per_step <= 0: return None, "non-positive slope"
    wall = per_step * steps
    cells = n * n
    return dict(cells=cells, steps=steps, wall=wall, mx=mx if mx is not None else float("nan"),
                rate=cells * 2 * steps / wall), ""


ARMS = [
    ("mumax3-ultrafast", lambda n, s: run_mumax3(f"{MW}/ultrafast", n, s)),
    ("mumax3-for-mac",   lambda n, s: run_mumax3(f"{MW}/formac-bin", n, s)),
    ("magnum.np MPS",    lambda n, s: run_magnumnp(n, s, "mps")),
    ("magnum.np CPU",    lambda n, s: run_magnumnp(n, s, "cpu")),
    ("MicroMagnetic.jl Metal", lambda n, s: run_micromagnetic(n, s, "metal")),
    ("MicroMagnetic.jl CPU",   lambda n, s: run_micromagnetic(n, s, "cpu")),
    ("OOMMF CPU 8t",     lambda n, s: run_oommf(n, s)),
]

# Every arm runs the SAME number of steps at each mesh. Giving the slow arms
# fewer steps would let them amortize their warm-up over a shorter timed window
# and quietly change the comparison, so they simply take longer instead.
STEPS = {128: 400, 256: 300, 512: 200, 1024: 100, 2048: 40, 4096: 15}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="128,256,512,1024,2048")
    ap.add_argument("--out", default=os.path.join(CT, "results.json"))
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]

    results = {}
    print(f"{'mesh':>7} {'arm':26} {'steps':>6} {'wall s':>9} {'cell-evals/s':>14} {'<mx>':>12}")
    for n in sizes:
        for name, fn in ARMS:
            steps = STEPS[n]
            r, err = fn(n, steps)
            key = f"{n}|{name}"
            if r:
                results[key] = r
                print(f"{n:5d}^2 {name:26} {r['steps']:6d} {r['wall']:9.3f} "
                      f"{r['rate']:14.4e} {r['mx']:12.7f}", flush=True)
            else:
                results[key] = {"error": err}
                short = err.strip().splitlines()[-1][:60] if err.strip() else "failed"
                print(f"{n:5d}^2 {name:26} {'-':>6} {'FAILED':>9} {short:>28}", flush=True)
        json.dump(results, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
