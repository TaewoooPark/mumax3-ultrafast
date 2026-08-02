#!/usr/bin/env python3
"""Largest problem each simulator can hold on this Mac.

Same problem as the throughput sweep, two steps only - this measures whether a
mesh can be set up and integrated at all, not how fast.

"Runs" means: completes, and does not page out. A run that swaps has left the
single-device regime the question is about, so it is killed and recorded as not
fitting. Swap is watched through vm_stat's Swapouts, not Pageouts: on Apple
Silicon the memory compressor records swap under Swapouts and leaves Pageouts
almost still, so a guard on Pageouts never fires.

mumax3 caches its demag kernel per geometry and at these sizes one entry runs to
several GB, so the cache is dropped after every run and free disk is checked
before each size.
"""
import json, os, re, subprocess, sys, time

CT = os.path.dirname(os.path.abspath(__file__))
MW = os.environ.get("MUMAX_WORK",
     "/Users/taewoopark/personal/mumax3-for-mac/mumax3-ultrafast/bench/capacity-20260803-002617/work")
JULIA = "/opt/homebrew/bin/julia"
PY = os.path.join(CT, "venv", "bin", "python")
SWAP_ABORT = 16384          # pages, 256 MB at 16 KiB
TIMEOUT = 900


def swapouts():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    m = re.search(r"Swapouts:\s+(\d+)", out)
    return int(m.group(1)) if m else 0


def free_gib():
    out = subprocess.run(["df", "-g", "/"], capture_output=True, text=True).stdout
    return int(out.splitlines()[1].split()[3])


def drop_kernel_cache():
    t = os.environ.get("TMPDIR", "/tmp").rstrip("/")
    subprocess.run(f"rm -rf {t}/mumax3kernel_*", shell=True)


def guarded(cmd, cwd=None):
    """Run cmd, killing it if it starts paging out. -> (verdict, wall, peak_gib)"""
    e = dict(os.environ); e["PATH"] = "/opt/homebrew/bin:" + e.get("PATH", "")
    so0 = swapouts()
    t0 = time.time()
    p = subprocess.Popen(f"/usr/bin/time -l {cmd}", shell=True, cwd=cwd, env=e,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    verdict = None
    while True:
        if p.poll() is not None:
            break
        if time.time() - t0 > TIMEOUT:
            p.kill(); verdict = "TIMEOUT"; break
        if swapouts() - so0 > SWAP_ABORT:
            p.kill(); verdict = "SWAP-KILLED"; break
        time.sleep(1)
    out, err = p.communicate()
    wall = time.time() - t0
    m = re.search(r"(\d+)\s+peak memory footprint", err or "")
    peak = int(m.group(1)) / 2**30 if m else 0.0
    if verdict is None:
        verdict = "OK" if (p.returncode == 0 and "RESULT" in (out or "") + (err or "")) else f"FAIL rc={p.returncode}"
    return verdict, wall, peak, (out or "") + (err or "")


def mumax3_cmd(binary, nx, ny):
    tpl = open(os.path.join(CT, "bench_mumax3.mx3")).read()
    src = (tpl.replace("NNN*NNN", f"{nx}*{ny}").replace("NNN, NNN", f"{nx}, {ny}")
              .replace("NNN", str(nx)).replace("SSS", "2"))
    # rebuild cleanly for rectangular meshes
    src = f"""setcellsize(4e-9, 4e-9, 4e-9)
setgridsize({nx}, {ny}, 1)
msat  = 800e3
aex   = 13e-12
alpha = 0.02
setsolver(2)
fixdt = 1e-13
m = uniform(1, 0.1, 0)
steps(2)
print("RESULT", {nx}*{ny})
"""
    scr = os.path.join(CT, "_cap.mx3"); open(scr, "w").write(src)
    od = os.path.join(CT, "_odcap"); subprocess.run(f"rm -rf {od}", shell=True)
    return f"{binary} -s -f -http '' -o {od} {scr}"


# The non-mumax3 harnesses take a single square dimension, so those arms walk the
# square rungs only. They fail well below the rectangular sizes anyway.
ARMS = {
    "mumax3-ultrafast": (lambda nx, ny: mumax3_cmd(f"{MW}/ultrafast", nx, ny), True),
    "mumax3-for-mac":   (lambda nx, ny: mumax3_cmd(f"{MW}/formac-bin", nx, ny), True),
    "magnum.np MPS":    (lambda nx, ny: f"{PY} {CT}/bench_magnumnp.py --n {nx} --steps 2 --device mps", False),
    "magnum.np CPU":    (lambda nx, ny: f"{PY} {CT}/bench_magnumnp.py --n {nx} --steps 2 --device cpu", False),
    "MicroMagnetic.jl CPU": (lambda nx, ny: f"{JULIA} {CT}/bench_micromagnetic.jl {nx} 2 1e-13 cpu", False),
}

LADDER = [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192), (8192, 10240), (16384, 8192)]


def main():
    results = {}
    print(f"{'arm':24} {'mesh':>13} {'cells':>12} {'verdict':>13} {'wall s':>8} {'peak GiB':>9}")
    for arm, (mk, rect_ok) in ARMS.items():
        best = None
        for nx, ny in LADDER:
            if not rect_ok and nx != ny:
                continue
            if free_gib() < 25:
                print(f"{arm:24} stopping: only {free_gib()} GiB disk free"); break
            cells = nx * ny
            v, wall, peak, out = guarded(mk(nx, ny))
            print(f"{arm:24} {f'{nx}x{ny}':>13} {cells:12d} {v:>13} {wall:8.1f} {peak:9.2f}", flush=True)
            results[f"{arm}|{nx}x{ny}"] = dict(cells=cells, verdict=v, wall=wall, peak_gib=peak)
            drop_kernel_cache()
            if v == "OK":
                best = (nx, ny, cells, peak)
            else:
                if "assertion" in out:
                    a = re.search(r"failed assertion `([^']*)'", out)
                    if a: print(f"{'':24}   -> {a.group(1)[:70]}")
                break
        results[f"{arm}|CEILING"] = best
        print(f"{arm:24} CEILING: {best}\n", flush=True)
    json.dump(results, open(os.path.join(CT, "results_capacity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
