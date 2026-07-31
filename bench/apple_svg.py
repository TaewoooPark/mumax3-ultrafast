#!/usr/bin/env python3
"""Render the Apple Silicon benchmark charts as SVG.

bench/gpus.gplot needs gnuplot, which is not part of the macOS toolchain, so
this script draws the same kind of chart with nothing but the standard library.
Run it from the repository root or from bench/:

    python3 bench/apple_svg.py

It writes bench/apple.svg and bench/apple-vs-gpus.svg from bench/apple.txt and
bench/gpus.txt. Keep the visual conventions of gpus.gplot: plain bars, the y
axis in millions of cells per second, and the device names rotated upright.
"""

import os
import shlex

HERE = os.path.dirname(os.path.abspath(__file__))

FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
# Measured bars are saturated, projections are pale, so the distinction survives
# printing in greyscale as well as on screen.
COLOR_MEASURED = "#0b6bcb"
COLOR_PROJECTED = "#9dc4ea"
COLOR_OTHER = "#b9bcc0"
COLOR_AXIS = "#5f6368"
COLOR_TEXT = "#202124"


def escape(text):
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def read_apple(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            spec, achievable, limit, throughput, status, name = shlex.split(line)
            rows.append(
                {
                    "name": name,
                    "value": float(throughput) / 1e6,
                    "status": status,
                    "spec": float(spec),
                    "achievable": float(achievable),
                    "limit": limit,
                }
            )
    rows.sort(key=lambda row: row["value"])
    return rows


def read_gpus(path):
    """CUDA results from bench/gpus.txt. Apple rows are skipped here so they can
    come from apple.txt instead, which distinguishes measured from projected."""
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = shlex.split(line)
            if "Apple" in fields[3]:
                continue
            rows.append(
                {
                    "name": fields[3],
                    "value": float(fields[1]) / 1e6,
                    "status": "other",
                }
            )
    return rows


def nice_ticks(top, count=6):
    """Round tick values that cover [0, top]."""
    raw = top / count
    magnitude = 10 ** len(str(int(raw))) // 10 or 1
    for factor in (1, 2, 2.5, 5, 10):
        step = factor * magnitude
        if step >= raw:
            break
    ticks, value = [], 0.0
    while value <= top + step / 2:
        ticks.append(value)
        value += step
    return ticks


def wrap(text, width):
    """Greedy wrap so the subtitle cannot run past the right edge."""
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def render(rows, title, subtitle, path, legend):
    bar_slot = 26 if len(rows) <= 24 else 15
    left, right = 78, 26
    plot_w = bar_slot * len(rows)
    plot_h = max(260, int(plot_w * 0.45))
    width = left + plot_w + right

    # A rotated label needs about 6 px per character, plus room for the legend.
    longest = max(len(row["name"]) for row in rows)
    label_room = int(6.0 * longest) + 62
    # Wrap the subtitle to the available width, at roughly 2 px per character.
    subtitle_lines = wrap(subtitle, max(40, int((width - left - right) / 5.9)))
    top = 40 + 16 * len(subtitle_lines) + 18
    height = top + plot_h + label_room

    top_value = max(row["value"] for row in rows)
    ticks = nice_ticks(top_value)
    axis_max = ticks[-1]

    def y_of(value):
        return top + plot_h - (value / axis_max) * plot_h

    out = []
    add = out.append
    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
    )
    add(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    add(
        f'<text x="{left}" y="30" font-family="{FONT}" font-size="15" '
        f'font-weight="600" fill="{COLOR_TEXT}">{escape(title)}</text>'
    )
    for index, line in enumerate(subtitle_lines):
        add(
            f'<text x="{left}" y="{48 + 15 * index}" font-family="{FONT}" '
            f'font-size="11" fill="{COLOR_AXIS}">{escape(line)}</text>'
        )

    # gridlines and y axis
    for tick in ticks:
        y = y_of(tick)
        add(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" '
            f'y2="{y:.1f}" stroke="#e4e6e9" stroke-width="1"/>'
        )
        add(
            f'<text x="{left - 8}" y="{y + 4:.1f}" font-family="{FONT}" '
            f'font-size="10" fill="{COLOR_AXIS}" text-anchor="end">'
            f'{tick:g}</text>'
        )
    add(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        f'stroke="{COLOR_AXIS}" stroke-width="1"/>'
    )
    add(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="{COLOR_AXIS}" stroke-width="1"/>'
    )
    label_y = top + plot_h / 2
    add(
        f'<text x="18" y="{label_y:.1f}" font-family="{FONT}" font-size="11" '
        f'fill="{COLOR_TEXT}" text-anchor="middle" '
        f'transform="rotate(-90 18 {label_y:.1f})">'
        f'throughput (M cells/s)</text>'
    )

    palette = {
        "measured": COLOR_MEASURED,
        "projected": COLOR_PROJECTED,
        "other": COLOR_OTHER,
    }
    bar_w = bar_slot * 0.62
    for index, row in enumerate(rows):
        x = left + index * bar_slot + (bar_slot - bar_w) / 2
        y = y_of(row["value"])
        add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{top + plot_h - y:.1f}" fill="{palette[row["status"]]}"/>'
        )
        text_x = x + bar_w / 2
        text_y = top + plot_h + 7
        weight = ' font-weight="600"' if row["status"] == "measured" else ""
        fill = COLOR_TEXT if row["status"] == "measured" else COLOR_AXIS
        add(
            f'<text x="{text_x:.1f}" y="{text_y:.1f}" font-family="{FONT}" '
            f'font-size="10" fill="{fill}"{weight} text-anchor="end" '
            f'transform="rotate(-90 {text_x:.1f} {text_y:.1f})">'
            f'{escape(row["name"])}</text>'
        )

    # legend
    lx, ly = left, top + plot_h + label_room - 22
    for label, status in legend:
        add(
            f'<rect x="{lx}" y="{ly - 9}" width="11" height="11" '
            f'fill="{palette[status]}"/>'
        )
        add(
            f'<text x="{lx + 17}" y="{ly}" font-family="{FONT}" '
            f'font-size="11" fill="{COLOR_TEXT}">{escape(label)}</text>'
        )
        lx += 20 + 7.0 * len(label)
    add("</svg>")

    with open(path, "w") as handle:
        handle.write("\n".join(out) + "\n")
    print(f"wrote {path} ({len(rows)} bars)")


def main():
    apple = read_apple(os.path.join(HERE, "apple.txt"))
    render(
        apple,
        "MuMax3 on Apple Silicon, 4.19M cells (2048x2048)",
        "Apple M4 measured with the Metal backend. Every other bar is a "
        "projection, not a benchmark: see bench/apple_project.py, which takes "
        "the lesser of the memory-system and GPU-width limits.",
        os.path.join(HERE, "apple.svg"),
        [("Apple M4, measured", "measured"), ("projected", "projected")],
    )

    combined = read_gpus(os.path.join(HERE, "gpus.txt")) + apple
    combined.sort(key=lambda row: row["value"])
    render(
        combined,
        "MuMax3 throughput, 4.19M cells (2048x2048): Apple Silicon vs CUDA GPUs",
        "CUDA figures measured, from bench/gpus.txt. Apple M4 measured with the "
        "Metal backend; the other Apple bars are projections from "
        "bench/apple_project.py, not benchmarks.",
        os.path.join(HERE, "apple-vs-gpus.svg"),
        [
            ("Apple M4, measured", "measured"),
            ("Apple, projected", "projected"),
            ("NVIDIA, measured", "other"),
        ],
    )


if __name__ == "__main__":
    main()
