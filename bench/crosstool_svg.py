#!/usr/bin/env python3
"""Render the cross-tool benchmark charts used in README.md.

Styling follows bench/apple-crossover.svg so every chart in this repository reads
as one set: white ground, #202124 text over #5f6368 supporting text, #e4e6e9
grid, and the blue family reserved for this project's own line - #0b4f9e for
mumax3-ultrafast, #4b9ae8 for the release before it - with external tools in
neutral grey. Each series carries a <title> so hovering names it.

Every number is measured, not modelled. Provenance:

  speed / physics   crosstool/run_crosstool.py - all arms, one M4 MacBook Air,
                    same session, identical Heun work at a fixed 1e-13 s step,
                    same step count per mesh.
  energy            crosstool/run_energy.py - powermetrics at 2 Hz, 512^2 with
                    2000 steps = 1.049e9 cell-evaluations per arm.
  capacity          crosstool/run_capacity.py - largest mesh that completes
                    without paging out.

  python3 bench/crosstool_svg.py        # writes docs/bench/*.svg
"""
import math
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "bench")

FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
INK, MUT, GRID, AXIS = "#202124", "#5f6368", "#e4e6e9", "#5f6368"
HERO_C, PREV_C, OTH_C, FAIL_C = "#0b4f9e", "#4b9ae8", "#9aa0a6", "#c5221f"
HERO = "mumax3-ultrafast"

SIZES = [128, 256, 512, 1024, 2048]

# cell-evaluations per second
SPEED = {
    "mumax3-ultrafast":       [1.611e8, 2.205e8, 1.360e8, 1.268e8, 1.170e8],
    "OOMMF (CPU, 8 threads)": [3.173e7, 3.914e7, 4.143e7, 3.414e7, 2.850e7],
    "magnum.np (MPS)":        [1.107e7, 2.277e7, 1.779e7, 2.632e7, 3.020e7],
    "MicroMagnetic.jl (CPU)": [1.847e7, 1.618e7, 1.288e7, 1.030e7, 8.219e6],
    "magnum.np (CPU)":        [7.410e6, 7.358e6, 7.362e6, 5.490e6, 4.967e6],
}

# 512^2, 2000 Heun steps, 1.049e9 cell-evaluations; name, M evals/J, J, wall s, W
ENERGY = [
    ("mumax3-ultrafast",        22.51,   46.6,   7.64,  6.10),
    ("magnum.np (MPS)",          3.78,  277.1,  43.42,  6.38),
    ("OOMMF (CPU, 8 threads)",   2.53,  414.2,  25.23, 16.42),
    ("MicroMagnetic.jl (CPU)",   1.87,  560.6,  86.14,  6.51),
    ("magnum.np (CPU)",          0.86, 1218.1, 143.67,  8.47),
]

# largest mesh completing without swap, 32 GB M4 MacBook Air
CAPACITY = [
    ("mumax3-ultrafast",         83886080, "8192 x 10240"),
    ("OOMMF (CPU, 8 threads)",   83886080, "8192 x 10240"),
    ("magnum.np (MPS)",          16777216, "4096 x 4096"),
    ("magnum.np (CPU)",          16777216, "4096 x 4096"),
    ("MicroMagnetic.jl (CPU)",   16777216, "4096 x 4096"),
    ("MicroMagnetic.jl (Metal)",        0, "cannot run"),
]

# <mx> after the same number of steps
PHYSICS = {
    "mumax3-ultrafast":       [0.990229, 0.994200, 0.994939, 0.9950321, 0.9950429],
    "OOMMF (CPU, 8 threads)": [0.990299, 0.994216, 0.994942, 0.9950322, 0.9950370],
    "MicroMagnetic.jl (CPU)": [0.990217, 0.994198, 0.994939, 0.9950319, 0.9950369],
    "magnum.np (MPS)":        [0.990202, 0.994195, 0.994939, 0.9950319, 0.9950370],
    "magnum.np (CPU)":        [0.990202, 0.994195, 0.994939, 0.9950319, 0.9950369],
}

BOLD = ' font-weight="600"'


def colour(name):
    return HERO_C if name == HERO else OTH_C


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def head(w, h, title, subtitle_lines, x0=78):
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
         f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
         f'<text x="{x0}" y="30" font-family="{FONT}" font-size="15" font-weight="600" '
         f'fill="{INK}">{esc(title)}</text>']
    for i, line in enumerate(subtitle_lines):
        s.append(f'<text x="{x0}" y="{48+i*15}" font-family="{FONT}" font-size="11" '
                 f'fill="{MUT}">{esc(line)}</text>')
    return s


def txt(x, y, body, size=10, fill=None, anchor="start", bold=False):
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill or MUT}"{a}{BOLD if bold else ""}>{body}</text>')


def frame(s, L, T, pw, ph):
    s.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T+ph}" stroke="{AXIS}" stroke-width="1"/>')
    s.append(f'<line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" stroke="{AXIS}" stroke-width="1"/>')


def xticks(s, lx, T, ph):
    for i, n in enumerate(SIZES):
        anc = "end" if i == len(SIZES) - 1 else "middle"
        s.append(txt(lx(i), T + ph + 18, f"{n}²", anchor=anc))


def spread(ys, gap, bottom=None):
    """Push labels apart, then lift the stack if it overflows the plot."""
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    if bottom is not None and ys and ys[-1] > bottom:
        over = ys[-1] - bottom
        ys = [y - over for y in ys]
    return ys


# ------------------------------------------------------------------- hero ---

# Short names for the three-panel summary. Same five arms, same order in every
# panel, so a reader compares one row of bars straight across.
HERO_ARMS = ["mumax3-ultrafast", "OOMMF (CPU)", "magnum.np (MPS)",
             "MicroMagnetic.jl (CPU)", "magnum.np (CPU)"]

HERO_PANELS = [
    ("Speed", "cell-evaluations per second, 512²",
     [1.360e8, 4.143e7, 1.779e7, 1.288e7, 7.362e6],
     lambda v: f"{v/1e6:,.0f}M"),
    ("Energy", "cell-evaluations per joule, same 512² run",
     [22.51e6, 2.53e6, 3.78e6, 1.87e6, 0.86e6],
     lambda v: f"{v/1e6:.1f}M"),
    ("Size", "largest mesh that fits in 32 GB",
     [83886080, 83886080, 16777216, 16777216, 16777216],
     lambda v: f"{v/1e6:.0f}M"),
]


def chart_hero(path):
    W, H = 1000, 524
    ptop, ph = 150, 196
    pw, gap, x0 = 268, 34, 62
    barw = 30
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{W/2:.0f}" y="42" font-family="{FONT}" font-size="24" font-weight="700" '
         f'fill="{INK}" text-anchor="middle">The fastest micromagnetic simulator on a Mac</text>',
         f'<text x="{W/2:.0f}" y="70" font-family="{FONT}" font-size="12.5" fill="{MUT}" '
         f'text-anchor="middle">Every simulator that runs on Apple Silicon, measured on one M4 MacBook Air — '
         f'same physics, same Heun integrator,</text>',
         f'<text x="{W/2:.0f}" y="88" font-family="{FONT}" font-size="12.5" fill="{MUT}" '
         f'text-anchor="middle">same fixed 1e-13 s step, same step count. Taller is better in all three panels.</text>']

    for pi, (ptitle, psub, vals, fmt) in enumerate(HERO_PANELS):
        px = x0 + pi * (pw + gap)
        vmax = max(vals) * 1.30
        s.append(f'<text x="{px}" y="{ptop-34}" font-family="{FONT}" font-size="14" '
                 f'font-weight="600" fill="{INK}">{ptitle}</text>')
        s.append(f'<text x="{px}" y="{ptop-18}" font-family="{FONT}" font-size="10.5" '
                 f'fill="{MUT}">{esc(psub)}</text>')
        s.append(f'<line x1="{px}" y1="{ptop+ph}" x2="{px+pw}" y2="{ptop+ph}" '
                 f'stroke="{AXIS}" stroke-width="1"/>')
        step = pw / len(vals)
        for i, v in enumerate(vals):
            name = HERO_ARMS[i]
            hero_bar = i == 0
            cx = px + step * (i + 0.5)
            bh = ph * v / vmax
            by = ptop + ph - bh
            s.append(f'<g><title>{esc(name)}: {fmt(v)}</title>'
                     f'<rect x="{cx-barw/2:.1f}" y="{by:.1f}" width="{barw}" height="{bh:.1f}" '
                     f'fill="{HERO_C if hero_bar else OTH_C}"/></g>')
            s.append(txt(cx, by - 20, fmt(v), size=11.5, fill=INK if hero_bar else MUT,
                         anchor="middle", bold=hero_bar))
            if not hero_bar:
                ratio = vals[0] / v
                lab = "tie" if abs(ratio - 1) < 0.01 else f"{ratio:.1f}×"
                s.append(txt(cx, by - 7, lab, size=10.5, fill=HERO_C, anchor="middle", bold=True))
            ly_ = ptop + ph + 8
            s.append(f'<text x="{cx:.1f}" y="{ly_}" font-family="{FONT}" font-size="10" '
                     f'fill="{INK if hero_bar else MUT}"{BOLD if hero_bar else ""} text-anchor="end" '
                     f'transform="rotate(-90 {cx:.1f} {ly_})">{esc(name)}</text>')

    note = ("MicroMagnetic.jl's Metal backend is absent because it cannot run at all — "
            "every path reaches a Float64 inside a Metal kernel, and Apple GPUs have no double precision.")
    s.append(txt(x0, H - 26, note, size=10.5, fill=FAIL_C))
    s.append(txt(x0, H - 11, "Size is a tie with OOMMF at 83.9M cells — mumax3-ultrafast is 4.5× faster at that size.",
                 size=10.5, fill=MUT))
    s.append("</svg>")
    open(path, "w").write("\n".join(s))


# ------------------------------------------------------------------ speed ---

def chart_speed(path):
    W, H = 860, 470
    L, R, T, B = 78, 232, 118, 54
    pw, ph = W - L - R, H - T - B
    ymin, ymax = 4e6, 2.6e8
    lx = lambda i: L + pw * i / (len(SIZES) - 1)
    ly = lambda v: T + ph * (1 - (math.log10(v) - math.log10(ymin)) /
                             (math.log10(ymax) - math.log10(ymin)))
    s = head(W, H, "Every micromagnetic simulator that runs on a Mac, on the same Mac", [
        "Throughput on one M4 MacBook Air. Identical physics in every arm — demag + exchange, Heun",
        "integration at a fixed 1e-13 s step, the same number of steps per mesh — so each arm performs",
        "exactly two effective-field evaluations per step. Higher is better.",
    ])

    for dec in (1e7, 1e8):
        for m in range(1, 10):
            v = dec * m
            if not (ymin <= v <= ymax):
                continue
            y, major = ly(v), m == 1
            fade = "" if major else ' opacity="0.5"'
            s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" stroke="{GRID}" '
                     f'stroke-width="1"{fade}/>')
            if major:
                s.append(txt(L - 8, y + 4, "10⁷" if dec == 1e7 else "10⁸", anchor="end"))
    frame(s, L, T, pw, ph)
    xticks(s, lx, T, ph)
    s.append(txt(L + pw / 2, H - 14, "mesh", size=11, fill=INK, anchor="middle"))
    s.append(f'<text x="18" y="{T+ph/2:.0f}" font-family="{FONT}" font-size="11" fill="{INK}" '
             f'text-anchor="middle" transform="rotate(-90 18 {T+ph/2:.0f})">cell-evaluations per second</text>')

    order = sorted(SPEED, key=lambda k: -SPEED[k][-1])
    for name in order:
        vals, c, hero = SPEED[name], colour(name), name == HERO
        pts = " ".join(f"{lx(i):.1f},{ly(v):.1f}" for i, v in enumerate(vals))
        s.append(f'<g><title>{esc(name)}: {vals[-1]/1e6:,.0f}M cell-evaluations/s at 2048²</title>')
        s.append(f'<polyline fill="none" points="{pts}" stroke="{c}" stroke-width="{3 if hero else 1.7}" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
        for i, v in enumerate(vals):
            s.append(f'<circle cx="{lx(i):.1f}" cy="{ly(v):.1f}" r="{3.4 if hero else 2.5}" fill="{c}"/>')
        s.append("</g>")

    lxl = L + pw + 12
    ys = spread([ly(SPEED[n][-1]) for n in order], 19.0, T + ph)
    for name, y in zip(order, ys):
        c, hero = colour(name), name == HERO
        yl = ly(SPEED[name][-1])
        if abs(y - yl) > 2:
            s.append(f'<path fill="none" stroke="{c}" stroke-width="1" opacity="0.5" '
                     f'd="M {L+pw+3:.1f} {yl:.1f} L {lxl-3:.1f} {y:.1f}"/>')
        s.append(f'<text x="{lxl:.1f}" y="{y+3.5:.1f}" font-family="{FONT}" font-size="10.5" '
                 f'fill="{INK if hero else MUT}"{BOLD if hero else ""}>{esc(name)} '
                 f'<tspan fill="{MUT}" font-size="9.5">{SPEED[name][-1]/1e6:,.0f}M/s</tspan></text>')

    yf = ys[-1] + 26
    s.append(txt(lxl, yf, "MicroMagnetic.jl (Metal)", size=10.5, fill=FAIL_C, bold=True))
    s.append(txt(lxl, yf + 13, "cannot run — every path reaches a", size=9.5, fill=FAIL_C))
    s.append(txt(lxl, yf + 24, "Float64 inside a Metal kernel", size=9.5, fill=FAIL_C))
    s.append("</svg>")
    open(path, "w").write("\n".join(s))


# --------------------------------------------------------------- bar charts --

def hbars(path, title, subtitle, rows, value_fmt, note, foot):
    """rows: (name, value, right_label, failed)"""
    W, L, R, T, rh = 860, 196, 214, 108, 34
    H = T + len(rows) * rh + 40
    pw = W - L - R
    vmax = max(r[1] for r in rows) * 1.02
    s = head(W, H, title, subtitle)
    for i, (name, val, right, failed) in enumerate(rows):
        y, hero = T + i * rh, name == HERO
        s.append(txt(L - 10, y + 17, esc(name), size=10.5, fill=INK if hero else MUT,
                     anchor="end", bold=hero))
        if failed:
            s.append(f'<rect x="{L}" y="{y+4}" width="2.5" height="20" fill="{FAIL_C}"/>')
            s.append(txt(L + 10, y + 18, esc(right), size=10.5, fill=FAIL_C, bold=True))
            continue
        bw = max(pw * val / vmax, 2)
        s.append(f'<g><title>{esc(name)}: {value_fmt(val)}</title>'
                 f'<rect x="{L}" y="{y+4}" width="{bw:.1f}" height="20" fill="{colour(name)}"/></g>')
        s.append(txt(L + bw + 8, y + 18, value_fmt(val), size=11,
                     fill=INK if hero else MUT, bold=hero))
        s.append(txt(W - 16, y + 18, esc(right), size=9.5, anchor="end"))
    s.append(txt(L, H - 14, note, size=10.5, fill=INK))
    if foot:
        s.append(txt(W - 16, H - 14, foot, size=9.5, anchor="end"))
    s.append("</svg>")
    open(path, "w").write("\n".join(s))


def chart_energy(path):
    rows = [(n, mj, f"{j:,.0f} J · {w:.1f} s · {p:.1f} W", False) for n, mj, j, w, p in ENERGY]
    hbars(path, "Energy for one identical simulation", [
        "512² mesh, 2000 Heun steps = 1.049×10⁹ cell-evaluations in every arm, so the physics performed is",
        "the same and only the joules differ. GPU, CPU and combined rails sampled with powermetrics at 2 Hz",
        "while the workload ran; the machine's idle floor was 0.195 W combined. Higher is better.",
    ], rows, lambda v: f"{v:.2f} M/J",
        "million cell-evaluations per joule", "measured on the rails — not derived from any TDP figure")


def chart_capacity(path):
    rows = [(n, c, mesh if c else "cannot run — no Apple GPU path works at all", c == 0)
            for n, c, mesh in CAPACITY]
    hbars(path, "Largest problem that fits on a 32 GB MacBook Air", [
        "The biggest mesh each simulator completes without paging out. A run that swaps has left the",
        "single-device regime the question is about, so it is killed and recorded as not fitting.",
    ], rows, lambda v: f"{v/1e6:.1f}M cells",
        "cells", "at the size where they tie, mumax3-ultrafast is 4.5× faster than OOMMF")


# ---------------------------------------------------------------- physics ---

def chart_physics(path):
    W, H = 860, 400
    L, R, T, B = 78, 232, 118, 54
    pw, ph = W - L - R, H - T - B
    lx = lambda i: L + pw * i / (len(SIZES) - 1)
    emin, emax = 1e-8, 1e-4
    ly = lambda v: T + ph * (1 - (math.log10(max(v, emin)) - math.log10(emin)) /
                             (math.log10(emax) - math.log10(emin)))
    ref = PHYSICS[HERO]
    s = head(W, H, "All four codes compute the same magnetization", [
        "Relative deviation of the mean in-plane magnetization from mumax3-ultrafast after the same number",
        "of steps. Four independently written simulators, three languages, three backends — the worst",
        "disagreement anywhere in the sweep is 7.1×10⁻⁵. Points resting on the axis floor are exact",
        "agreement to every digit printed. Lower is better.",
    ])
    for e, lab in ((1e-8, "10⁻⁸"), (1e-7, "10⁻⁷"), (1e-6, "10⁻⁶"),
                   (1e-5, "10⁻⁵"), (1e-4, "10⁻⁴")):
        y = ly(e)
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(txt(L - 8, y + 4, lab, anchor="end"))
    frame(s, L, T, pw, ph)
    xticks(s, lx, T, ph)
    s.append(txt(L + pw / 2, H - 14, "mesh", size=11, fill=INK, anchor="middle"))
    s.append(f'<text x="18" y="{T+ph/2:.0f}" font-family="{FONT}" font-size="11" fill="{INK}" '
             f'text-anchor="middle" transform="rotate(-90 18 {T+ph/2:.0f})">relative deviation</text>')

    ytol = ly(1e-4)
    s.append(f'<line x1="{L}" y1="{ytol:.1f}" x2="{L+pw}" y2="{ytol:.1f}" stroke="{AXIS}" '
             f'stroke-width="1" stroke-dasharray="5 4" opacity="0.7"/>')
    s.append(txt(L + pw + 12, ytol + 3.5, "0.01% — every code sits below", size=10, fill=INK))

    series = [(n, [abs(v - r) / abs(r) for v, r in zip(vals, ref)])
              for n, vals in PHYSICS.items() if n != HERO]
    series.sort(key=lambda kv: -kv[1][-1])
    for name, dev in series:
        c = colour(name)
        pts = " ".join(f"{lx(i):.1f},{ly(d):.1f}" for i, d in enumerate(dev))
        s.append(f'<g><title>{esc(name)}: {dev[-1]:.1e} relative deviation at 2048²</title>')
        s.append(f'<polyline fill="none" points="{pts}" stroke="{c}" stroke-width="1.7" '
                 f'stroke-linejoin="round"/>')
        for i, d in enumerate(dev):
            s.append(f'<circle cx="{lx(i):.1f}" cy="{ly(d):.1f}" r="2.5" fill="{c}"/>')
        s.append("</g>")
    ys = spread([ly(d[-1]) for _, d in series], 17.0, T + ph)
    for (name, _), y in zip(series, ys):
        s.append(txt(L + pw + 12, y + 3.5, esc(name), size=10.5, fill=colour(name)))
    s.append("</svg>")
    open(path, "w").write("\n".join(s))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    chart_hero(os.path.join(OUT, "hero.svg"))
    chart_speed(os.path.join(OUT, "speed.svg"))
    chart_energy(os.path.join(OUT, "energy.svg"))
    chart_capacity(os.path.join(OUT, "capacity.svg"))
    chart_physics(os.path.join(OUT, "physics.svg"))
    for f in ("hero", "speed", "energy", "capacity", "physics"):
        p = os.path.join(OUT, f + ".svg")
        print(f"  docs/bench/{f}.svg  {os.path.getsize(p):,} bytes")
