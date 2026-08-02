#!/bin/bash
#
# Capacity and memory characterisation for a mumax3 build on Apple Silicon.
#
#   ./bench/capacity.sh  name=/path/to/binary  [name2=/path/to/other] ...
#   ./bench/capacity.sh                                # builds HEAD and measures it
#
# Answers four questions that the throughput suite cannot, and that a projection
# model should not be trusted to answer:
#
#   1. What are the device's real limits?  Two exist and they differ:
#      recommendedMaxWorkingSetSize (advisory - measured runs exceed it and are
#      fine) and maxBufferLength (per allocation, RAM/2 on the machines seen).
#   2. How many bytes does a cell cost, and where do they go?  A linear fit over
#      several meshes separates per-cell storage from fixed process cost, and
#      vmmap splits the result into Metal buffers and Go heap. That split matters
#      on unified memory, where host-side allocations compete with GPU ones.
#   3. Is a large mesh still physics?  A uniformly out-of-plane film approaches a
#      known analytic demag limit as it widens, so the check gets sharper exactly
#      where the capacity claim lives.
#   4. Where is the ceiling?  Measured as the largest mesh that completes without
#      paging out, not derived from a bytes-per-cell model.
#
# Nothing here is timing-sensitive: byte counts and allocation ceilings do not
# care about machine load, so this is safe to run on a busy machine.
#
# Two hazards this handles, both learned the hard way:
#
#   Swap is watched through vm_stat's Swapouts. On Apple Silicon the memory
#   compressor records swap there and leaves Pageouts nearly still - 10.8M pages
#   against 9.6k in one run - so a guard on Pageouts silently never fires and a
#   thrashing mesh gets scored as a fit.
#
#   mumax3 caches the demag kernel per geometry, and at these sizes one entry is
#   8-10 GB: 16384x20480 padded, six symmetric components, four bytes each. The
#   cache is dropped after every run and free space is checked before each size,
#   because letting it accumulate filled a 926 GB volume and killed a run.

set -u -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$REPO/bench/capacity-$STAMP"
RESULTS="$OUT/results.txt"
WORK="$OUT/work"
SWAP_PAGES_ABORT="${SWAP_PAGES_ABORT:-16384}"     # 256 MB at 16 KiB pages
DISK_MIN_GIB="${DISK_MIN_GIB:-25}"
RUN_TIMEOUT="${RUN_TIMEOUT:-600}"
mkdir -p "$WORK"

say()   { printf '%s\n' "$*" | tee -a "$RESULTS"; }
head1() { say ""; say "================================================================"; say "$*"; say "================================================================"; }
note()  { printf '  %s\n' "$*" >&2; }

swapouts()    { vm_stat | awk '/Swapouts/ {gsub(/[^0-9]/,"",$NF); print $NF}'; }
freegib()     { df -g / | awk 'NR==2 {print $4}'; }
drop_kcache() { rm -rf "${TMPDIR%/}"/mumax3kernel_* 2>/dev/null; true; }
trap 'drop_kcache; git worktree prune 2>/dev/null' EXIT INT TERM

# ------------------------------------------------------------- which builds ---

declare -a NAMES=() BINS=()
if [ "$#" -eq 0 ]; then
  note "no binaries given; building HEAD"
  go build -o "$WORK/head" ./cmd/mumax3 || { echo "build failed" >&2; exit 1; }
  NAMES+=("head"); BINS+=("$WORK/head")
else
  for a in "$@"; do
    n="${a%%=*}"; b="${a#*=}"
    [ -x "$b" ] || { echo "not executable: $b" >&2; exit 1; }
    NAMES+=("$n"); BINS+=("$b")
  done
fi

# Run one script under /usr/bin/time -l, killed if it starts paging out.
# Echoes: rc wall swapdelta_pages peak_footprint_bytes
run_guarded() {  # binary script
  local bin="$1" scr="$2" od="$WORK/od"
  rm -rf "$od"; : > "$WORK/run.err"
  local so0 t0 t1 rc pid i so
  so0=$(swapouts); t0=$(python3 -c 'import time;print(time.time())')
  ( /usr/bin/time -l "$bin" -s -f -http "" -o "$od" "$scr" ) >"$WORK/run.out" 2>"$WORK/run.err" </dev/null &
  pid=$!; rc=124
  for i in $(seq 1 "$RUN_TIMEOUT"); do
    if ! kill -0 "$pid" 2>/dev/null; then wait "$pid"; rc=$?; break; fi
    sleep 1
    if [ $((i % 2)) = 0 ]; then
      so=$(swapouts)
      if [ $((so - so0)) -gt "$SWAP_PAGES_ABORT" ]; then
        kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rc=125; break
      fi
    fi
  done
  [ "$rc" = 124 ] && { kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; }
  t1=$(python3 -c 'import time;print(time.time())')
  local fp; fp=$(awk '/peak memory footprint/ {print $1}' "$WORK/run.err" | tail -1)
  printf '%s %s %s %s\n' "$rc" "$(python3 -c "print(f'{$t1-$t0:.1f}')")" "$(( $(swapouts) - so0 ))" "${fp:-0}"
}

verdict() {  # rc has_output
  case "$1" in
    0)   [ "$2" = 1 ] && echo OK || echo "NO OUTPUT" ;;
    124) echo TIMEOUT ;;
    125) echo SWAP-KILLED ;;
    *)   echo "CRASH rc=$1" ;;
  esac
}

step_script() {  # nx ny steps outfile
  cat > "$4" <<EOF
setcellsize(4e-9, 4e-9, 4e-9)
setgridsize($1, $2, 1)
msat  = 800e3
aex   = 13e-12
alpha = 0.02
setsolver(2)
m = uniform(1, 0, 0)
start := now()
neval0 := Neval.get()
steps($3)
wall := since(start).Seconds()
fprintln("point.txt", $1*$2, $1*$2*(Neval.get()-neval0)/wall, wall)
EOF
}

# =========================================================== 1. device ========

head1 "mumax3 capacity - $STAMP"
say "  chip:   $(sysctl -n machdep.cpu.brand_string)"
say "  macOS:  $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
say "  commit: $(git rev-parse --short HEAD)"
for i in "${!NAMES[@]}"; do say "  build:  ${NAMES[$i]} -> ${BINS[$i]}"; done
say "  load:   $(sysctl -n vm.loadavg | awk '{print $2}')   disk free: $(freegib) GiB"

head1 "1. Metal device limits, from the API"
cat > "$WORK/mtl.swift" <<'EOF'
import Metal
import Foundation
let d = MTLCreateSystemDefaultDevice()!
let ram = Double(ProcessInfo.processInfo.physicalMemory)
func gb(_ b: Double) -> String { String(format: "%.2f GiB", b/1073741824.0) }
print("  device                        \(d.name)")
print("  hasUnifiedMemory              \(d.hasUnifiedMemory)")
print("  physicalMemory                \(gb(ram))")
print("  recommendedMaxWorkingSetSize  \(gb(Double(d.recommendedMaxWorkingSetSize)))  = \(String(format: "%.4f", Double(d.recommendedMaxWorkingSetSize)/ram)) x RAM   (advisory)")
print("  maxBufferLength               \(gb(Double(d.maxBufferLength)))  = \(String(format: "%.4f", Double(d.maxBufferLength)/ram)) x RAM   (per allocation)")
print("RAW \(d.recommendedMaxWorkingSetSize) \(d.maxBufferLength) \(ProcessInfo.processInfo.physicalMemory)")
EOF
swift "$WORK/mtl.swift" > "$WORK/mtl.txt" 2>"$WORK/mtl.err" \
  && grep -v '^RAW' "$WORK/mtl.txt" | tee -a "$RESULTS" \
  || say "  Metal query failed - see $WORK/mtl.err"
read -r _ WSS MAXBUF RAMB <<<"$(grep '^RAW' "$WORK/mtl.txt" 2>/dev/null || echo 'RAW 0 0 0')"

# ====================================================== 2. bytes per cell =====

head1 "2. Bytes per cell"
say "  Solver 2 (Heun), steps(2), 4 nm cells. Padded lengths are all 2^a*3^b, so"
say "  no point measures FFT padding pathology. Sizes are interleaved across"
say "  builds so drift in machine memory state hits every build equally."
say ""
printf '  %-9s %11s' "mesh" "cells" | tee -a "$RESULTS" >/dev/null
say "$(printf '  %-9s %11s%s' 'mesh' 'cells' "$(for n in "${NAMES[@]}"; do printf '%18s' "$n RSS MB"; done)")"
for n in "${NAMES[@]}"; do : > "$WORK/bpc-$n.txt"; done
for m in 512 768 1024 1536 2048 3072 4096; do
  cells=$((m * m)); step_script "$m" "$m" 2 "$WORK/bpc.mx3"
  row="$(printf '  %-9s %11d' "${m}^2" "$cells")"; ok=1
  for i in "${!NAMES[@]}"; do
    read -r rc w sd fp <<<"$(run_guarded "${BINS[$i]}" "$WORK/bpc.mx3")"
    rss=$(awk '/maximum resident set size/ {print $1}' "$WORK/run.err" | tail -1); rss="${rss:-0}"
    if [ "$rc" != 0 ]; then row="$row$(printf '%18s' 'FAILED')"; ok=0
    else
      row="$row$(python3 -c "print(f'{$rss/1048576:18.1f}')")"
      printf '%d %d %d\n' "$cells" "$rss" "$fp" >> "$WORK/bpc-${NAMES[$i]}.txt"
    fi
  done
  say "$row"
  drop_kcache
done
say ""
python3 - "$WORK" "${NAMES[@]}" <<'PY' | tee -a "$RESULTS"
import sys, os
work, names = sys.argv[1], sys.argv[2:]
def fit(path, col):
    xs, ys = [], []
    for line in open(path):
        p = line.split(); xs.append(float(p[0])); ys.append(float(p[col]))
    n = len(xs)
    if n < 2: return None
    mx, my = sum(xs)/n, sum(ys)/n
    den = sum((x-mx)**2 for x in xs)
    if not den: return None
    s = sum((x-mx)*(y-my) for x, y in zip(xs, ys))/den
    b = my - s*mx
    sr = sum((y-(s*x+b))**2 for x, y in zip(xs, ys)); st = sum((y-my)**2 for y in ys)
    return s, b, (1-sr/st if st else float('nan')), n
print("  Linear fit   bytes = slope * cells + fixed")
print(f"    {'build':14} {'metric':10} {'B/cell':>9} {'fixed MB':>10} {'R^2':>10} {'n':>3}")
res = {}
for nm in names:
    p = os.path.join(work, f"bpc-{nm}.txt")
    if not os.path.exists(p): continue
    for metric, col in (("maxRSS", 1), ("footprint", 2)):
        f = fit(p, col)
        if not f: continue
        s, b, r2, n = f; res[(nm, metric)] = s
        print(f"    {nm:14} {metric:10} {s:9.1f} {b/1048576:10.1f} {r2:10.6f} {n:3d}")
if len(names) > 1:
    print()
    base = names[0]
    for metric in ("maxRSS", "footprint"):
        a = res.get((base, metric))
        if not a: continue
        for nm in names[1:]:
            b = res.get((nm, metric))
            if b: print(f"  {metric:10}: {nm} {b:.1f} B/cell vs {base} {a:.1f}  -> {a/b:.4f}x ({100*(a-b)/a:+.2f}%)")
PY

# ============================================== 3. where the bytes go =========

head1 "3. Where the bytes go - vmmap attribution"
say "  On unified memory a host-side allocation costs exactly as much as a GPU one."
say "  IOAccelerator(graphics) is Metal; VM_ALLOCATE is the Go arena."
say ""
say "  build          mesh       cells      Metal B/cell   Go heap B/cell   total B/cell"
for i in "${!NAMES[@]}"; do
  for m in 2048 4096; do
    cells=$((m * m)); step_script "$m" "$m" 400 "$WORK/vm.mx3"
    rm -rf "$WORK/od-vm"
    ( "${BINS[$i]}" -s -f -http "" -o "$WORK/od-vm" "$WORK/vm.mx3" ) >/dev/null 2>&1 </dev/null &
    pid=$!
    for _ in $(seq 1 60); do
      sleep 1
      rss=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' ')
      [ -n "$rss" ] && [ "$rss" -gt $((cells / 40)) ] && break
      kill -0 $pid 2>/dev/null || break
    done
    sleep 3
    vmmap -summary $pid > "$WORK/vmmap-${NAMES[$i]}-$m.txt" 2>&1
    kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
    python3 - "$WORK/vmmap-${NAMES[$i]}-$m.txt" "$cells" "${NAMES[$i]}" "$m" <<'PY' | tee -a "$RESULTS"
import re, sys
txt = open(sys.argv[1], errors="ignore").read(); cells = int(sys.argv[2])
def dirty(label):
    for line in txt.splitlines():
        if line.startswith(label):
            n = re.findall(r'([\d.]+[KMG])', line)
            if len(n) < 3: return 0.0
            u = n[2][-1]; return float(n[2][:-1])*{'K':1024,'M':1048576,'G':1073741824}[u]
    return 0.0
g, h = dirty("IOAccelerator (graphics)"), dirty("VM_ALLOCATE")
if g or h:
    print(f"  {sys.argv[3]:14} {sys.argv[4]+'^2':>6} {cells:11d} {g/cells:15.1f} {h/cells:16.1f} {(g+h)/cells:14.1f}")
else:
    print(f"  {sys.argv[3]:14} {sys.argv[4]+'^2':>6} {cells:11d}   vmmap parse failed")
PY
    drop_kcache
  done
done

# ================================================= 4. validity at scale =======

head1 "4. Physics validity at scale"
say "  A uniformly out-of-plane film has Nz -> 1 as it widens, so"
say "  <B_demag,z> -> -mu0*Msat = -1.005310 T. A wider mesh must land closer."
say ""
say "$(printf '  %-9s %11s%s' 'mesh' 'cells' "$(for n in "${NAMES[@]}"; do printf '%22s' "$n <B_demag,z>"; done)")"
for m in 512 1024 2048 4096 8192; do
  cells=$((m * m))
  cat > "$WORK/demag.mx3" <<EOF
setcellsize(4e-9, 4e-9, 4e-9)
setgridsize($m, $m, 1)
msat = 800e3
aex  = 13e-12
setsolver(2)
m = uniform(0, 0, 1)
fprintln("demag.txt", B_demag.average())
EOF
  row="$(printf '  %-9s %11d' "${m}^2" "$cells")"
  for i in "${!NAMES[@]}"; do
    read -r rc w sd fp <<<"$(run_guarded "${BINS[$i]}" "$WORK/demag.mx3")"
    v=""; [ "$rc" = 0 ] && v=$(tr -d '[]' < "$WORK/od/demag.txt" 2>/dev/null | awk '{print $3}')
    row="$row$(python3 -c "
v='''$v'''.strip()
print(f'{float(v):22.7f}' if v else f'{\"FAILED\":>22}')")"
  done
  say "$row"
  drop_kcache
done
say ""
say "  target -1.0053096 T"

# ======================================================== 5. the ceiling ======

head1 "5. Ceiling - largest mesh that completes without paging out"
say "  Guard: Swapouts (not Pageouts - the compressor does not touch Pageouts)."
say "  Kernel cache dropped between sizes; sizes skipped below $DISK_MIN_GIB GiB free."
say ""
say "  build          mesh              cells   result         wall s   swap dp   peak fp GiB   B/cell"
for i in "${!NAMES[@]}"; do
  best="none"; bestfp=0
  while read -r nx ny; do
    [ -z "$nx" ] && continue
    free=$(freegib)
    if [ "$free" -lt "$DISK_MIN_GIB" ]; then say "  ${NAMES[$i]}: stopping, only ${free} GiB disk free"; break; fi
    cells=$((nx * ny)); step_script "$nx" "$ny" 2 "$WORK/cap.mx3"
    read -r rc w sd fp <<<"$(run_guarded "${BINS[$i]}" "$WORK/cap.mx3")"
    has=0; [ -s "$WORK/od/point.txt" ] && has=1
    v=$(verdict "$rc" "$has")
    [ "$v" = OK ] && { best="${nx}x${ny} = $cells"; bestfp="$fp"; }
    python3 -c "print(f'  {\"${NAMES[$i]}\":14} {\"${nx}x${ny}\":>13} {$cells:11d}   {\"$v\":13} {\"$w\":>7} {\"$sd\":>8} {$fp/2**30:13.2f} {($fp/$cells if $fp else 0):8.1f}')" | tee -a "$RESULTS"
    [ "$v" != OK ] && grep -m1 -oE "failed assertion \`[^']*'|cannot allocate|out of memory" "$WORK/run.err" 2>/dev/null | sed 's/^/                 /' | tee -a "$RESULTS"
    drop_kcache
    [ "$v" != OK ] && break
  done <<'EOD'
2048 2048
4096 4096
8192 8192
8192 10240
8192 12288
16384 8192
EOD
  say "  ${NAMES[$i]} CEILING: $best  ($(python3 -c "print(f'{$bestfp/2**30:.2f} GiB')"))"
  say ""
done

head1 "6. After"
say "  disk free: $(freegib) GiB (kernel cache dropped)"
say "  swap used: $(sysctl -n vm.swapusage | sed 's/.*used = \([0-9.]*M\).*/\1/')"
say ""
say "Wrote: $RESULTS"
note "Done -> $RESULTS"
