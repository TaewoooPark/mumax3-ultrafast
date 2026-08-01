#!/usr/bin/env python3
"""Render the Apple Silicon benchmark charts as SVG.

bench/gpus.gplot needs gnuplot, which is not part of the macOS toolchain, so
this script draws the same kind of chart with nothing but the standard library.
Run it from the repository root or from bench/:

    python3 bench/apple_svg.py

It writes all four bench/apple*.svg charts and refreshes doc/static/gpus.svg
from bench/apple.txt, bench/gpus.txt, bench/price_tiers.txt, and
bench/capacity.txt. Keep the visual conventions of gpus.gplot: plain bars, the
y axis in millions of cells per second, and the device names rotated upright.
"""

import math
import os
import shlex

HERE = os.path.dirname(os.path.abspath(__file__))

FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
# Measured bars are saturated, projections are pale, so the distinction survives
# printing in greyscale as well as on screen.
COLOR_MEASURED = "#0b4f9e"    # Apple, measured: darkest
COLOR_PROJECTED = "#4b9ae8"   # Apple, projected: mid blue, plus a hatch overlay
COLOR_OTHER = "#8d9299"       # NVIDIA, measured: neutral grey
HOST_USD = 800                # host machine added to every GPU card price
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


def read_gpus(path, include_apple=False):
    """Measured results from bench/gpus.txt.

    Apple rows are normally skipped so the Apple comparison can source its
    measured and projected rows consistently from apple.txt. The measured-only
    site chart includes the precise Apple row from gpus.txt.
    """
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = shlex.split(line)
            is_apple = "Apple" in fields[3]
            if is_apple and not include_apple:
                continue
            rows.append(
                {
                    "name": fields[3],
                    "value": float(fields[1]) / 1e6,
                    "status": "measured" if is_apple else "other",
                }
            )
    return rows


def read_oommf(path):
    """Read the CPU reference using the formula in bench/gpus.gplot."""
    with open(path) as handle:
        size, steps, wall = handle.read().split()
    value = 4 * int(size) ** 2 * int(steps) / float(wall) / 1e6
    return {"name": "OOMMF (CPU)", "value": value, "status": "other"}


def nice_ticks(top, count=7):
    """Round tick values that cover [0, top]."""
    raw = top / count
    magnitude = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 2.5, 5, 10):
        step = factor * magnitude
        if step >= raw:
            break
    axis_max = math.ceil(top / step) * step
    return [index * step for index in range(round(axis_max / step) + 1)]


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


def render(rows, title, subtitle, path, legend,
           y_label="throughput (M cell-evals/s)"):
    bar_slot = 26 if len(rows) <= 24 else 15
    left, right = 78, 26
    # Short charts still need enough room for their title and full legend. This
    # also fixes the old capacity chart, whose 260 px canvas clipped both.
    title_room = 8 * len(title)
    legend_room = sum(31 + 7 * len(label) for label, _ in legend)
    plot_w = max(bar_slot * len(rows), title_room, legend_room)
    bar_slot = plot_w / len(rows)
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
    add(
        '<defs><pattern id="hatch" width="5" height="5" '
        'patternTransform="rotate(45)" patternUnits="userSpaceOnUse">'
        f'<line x1="0" y1="0" x2="0" y2="5" stroke="#ffffff" '
        'stroke-width="2.1"/></pattern></defs>'
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
        f'{y_label}</text>'
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
        if row["status"] == "projected":
            add(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{top + plot_h - y:.1f}" fill="url(#hatch)"/>'
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
        if status == "projected":
            add(
                f'<rect x="{lx}" y="{ly - 9}" width="11" height="11" '
                f'fill="url(#hatch)"/>'
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


TIER_TITLES = {
    "entry": "entry",
    "mid": "mid",
    "high": "high",
    "top": "top",
}

COLOR_NVIDIA_PRICE = "#5f6f7a"


APPLE_TIER_CHIPS = {
    "Mac mini M4": "Apple M4",
    "Mac mini M4 Pro": "Apple M4 Pro",
    "Mac Studio M4 Max": "Apple M4 Max 40c",
    "Mac Studio M3 Ultra": "Apple M3 Ultra",
}


def read_tiers(path, apple_rows):
    apple_by_name = {row["name"]: row["value"] for row in apple_rows}
    tiers = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tier, vendor, price, throughput, label = shlex.split(line)
            price = int(price)
            total = price + (HOST_USD if vendor == "nvidia" else 0)
            value = float(throughput)
            if vendor == "apple":
                chip = APPLE_TIER_CHIPS[label]
                projected = apple_by_name[chip]
                if abs(value - projected) > 0.051:
                    raise ValueError(
                        f"stale Apple tier {label!r}: price_tiers.txt has "
                        f"{value:.1f}, apple.txt has {projected:.1f}"
                    )
                value = projected
            tiers.append(
                {
                    "tier": tier,
                    "vendor": vendor,
                    "total": total,
                    "value": value,
                    "label": label,
                }
            )
    order = []
    for row in tiers:
        if row["tier"] not in order:
            order.append(row["tier"])
    return tiers, order


def tier_speedup_range(tiers, order):
    ratios = []
    for tier in order:
        members = [row for row in tiers if row["tier"] == tier]
        apple = next(row for row in members if row["vendor"] == "apple")
        nvidia = next(row for row in members if row["vendor"] == "nvidia")
        ratios.append(nvidia["value"] / apple["value"])
    return min(ratios), max(ratios)


def render_price_tiers(path, tiers, order):
    """Grouped bars: one Apple machine against one NVIDIA machine per tier."""
    left, right, top = 78, 26, 96
    group_w, bar_w, gap = 236, 84, 12
    plot_w = group_w * len(order)
    plot_h = 300
    width = left + plot_w + right
    height = top + plot_h + 148

    axis_max = 2000.0
    ticks = [0, 500, 1000, 1500, 2000]

    def y_of(value):
        return top + plot_h - (value / axis_max) * plot_h

    out = []
    add = out.append
    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
    )
    add(
        '<defs><pattern id="hatch" width="5" height="5" '
        'patternTransform="rotate(45)" patternUnits="userSpaceOnUse">'
        '<line x1="0" y1="0" x2="0" y2="5" stroke="#ffffff" '
        'stroke-width="2.1"/></pattern></defs>'
    )
    add(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    add(
        f'<text x="{left}" y="30" font-family="{FONT}" font-size="15" '
        f'font-weight="600" fill="{COLOR_TEXT}">MuMax3 at 4.19M cells: '
        f'comparable-cost Mac vs PC</text>'
    )
    for index, line in enumerate(
        [
            "Whole machines, so every NVIDIA price adds "
            f"{HOST_USD} USD for a host. Approximate US launch prices; the "
            "RTX 50 series has",
            "traded well above MSRP, which flatters the PC column. Apple "
            "figures are projections except the measured Mac mini M4.",
        ]
    ):
        add(
            f'<text x="{left}" y="{50 + 15 * index}" font-family="{FONT}" '
            f'font-size="11" fill="{COLOR_AXIS}">{escape(line)}</text>'
        )

    for tick in ticks:
        y = y_of(tick)
        add(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'stroke="#e4e6e9" stroke-width="1"/>'
        )
        add(
            f'<text x="{left - 8}" y="{y + 4:.1f}" font-family="{FONT}" '
            f'font-size="10" fill="{COLOR_AXIS}" text-anchor="end">{tick}</text>'
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
        f'throughput (M cell-evals/s)</text>'
    )

    for group, tier in enumerate(order):
        members = [row for row in tiers if row["tier"] == tier]
        members.sort(key=lambda row: 0 if row["vendor"] == "apple" else 1)
        base = left + group * group_w
        for slot, row in enumerate(members):
            x = base + (group_w - (2 * bar_w + gap)) / 2 + slot * (bar_w + gap)
            y = y_of(row["value"])
            fill = COLOR_PROJECTED if row["vendor"] == "apple" else COLOR_NVIDIA_PRICE
            if row["label"] == "Mac mini M4":
                fill = COLOR_MEASURED
            add(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" '
                f'height="{top + plot_h - y:.1f}" fill="{fill}"/>'
            )
            if row["vendor"] == "apple" and row["label"] != "Mac mini M4":
                add(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" '
                    f'height="{top + plot_h - y:.1f}" fill="url(#hatch)"/>'
                )
            # throughput, then per-1000-USD efficiency
            add(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 17:.1f}" '
                f'font-family="{FONT}" font-size="10" font-weight="600" '
                f'fill="{COLOR_TEXT}" text-anchor="middle">'
                f'{row["value"]:.0f}</text>'
            )
            add(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" '
                f'font-family="{FONT}" font-size="9" fill="{COLOR_AXIS}" '
                f'text-anchor="middle">'
                f'{row["value"] / row["total"] * 1000:.0f}/$1k</text>'
            )
            caption = wrap(row["label"], 15) + [f'${row["total"]:,}']
            for offset, text in enumerate(caption):
                is_price = offset == len(caption) - 1
                add(
                    f'<text x="{x + bar_w / 2:.1f}" '
                    f'y="{top + plot_h + 16 + 12 * offset:.1f}" '
                    f'font-family="{FONT}" font-size="10" '
                    f'fill="{COLOR_AXIS if is_price else COLOR_TEXT}" '
                    f'text-anchor="middle">{escape(text)}</text>'
                )
        ratio = (
            [r for r in members if r["vendor"] == "nvidia"][0]["value"]
            / [r for r in members if r["vendor"] == "apple"][0]["value"]
        )
        add(
            f'<text x="{base + group_w / 2:.1f}" '
            f'y="{top + plot_h + 66:.1f}" font-family="{FONT}" font-size="10" '
            f'fill="{COLOR_AXIS}" text-anchor="middle">'
            f'PC is {ratio:.1f}x</text>'
        )

    lx, ly = left, height - 22
    for label, fill, hatched in [
        ("Apple, measured", COLOR_MEASURED, False),
        ("Apple, projected", COLOR_PROJECTED, True),
        ("NVIDIA, measured", COLOR_NVIDIA_PRICE, False),
    ]:
        add(f'<rect x="{lx}" y="{ly - 9}" width="11" height="11" fill="{fill}"/>')
        if hatched:
            add(
                f'<rect x="{lx}" y="{ly - 9}" width="11" height="11" '
                f'fill="url(#hatch)"/>'
            )
        add(
            f'<text x="{lx + 17}" y="{ly}" font-family="{FONT}" font-size="11" '
            f'fill="{COLOR_TEXT}">{escape(label)}</text>'
        )
        lx += 26 + 7.0 * len(label)
    add("</svg>")
    with open(path, "w") as handle:
        handle.write("\n".join(out) + "\n")
    print(f"wrote {path} ({len(order)} tiers)")


BYTES_PER_CELL = 133.1   # measured, see bench/capacity.txt


def read_capacity(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vendor, gb, frac, label = shlex.split(line)
            cells = int(gb) * (1024 ** 3) * float(frac) / BYTES_PER_CELL
            rows.append(
                {
                    "name": label,
                    "value": cells / 1e6,
                    "status": "measured" if vendor == "apple" else "other",
                    "mesh": int(cells ** 0.5),
                }
            )
    rows.sort(key=lambda row: row["value"])
    return rows


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

    tiers, order = read_tiers(os.path.join(HERE, "price_tiers.txt"), apple)
    render_price_tiers(os.path.join(HERE, "apple-price-tiers.svg"), tiers, order)
    min_ratio, max_ratio = tier_speedup_range(tiers, order)

    capacity = read_capacity(os.path.join(HERE, "capacity.txt"))
    for row in capacity:
        row["name"] = f'{row["name"]} ({row["mesh"]}^2)'
    render(
        capacity,
        "Largest MuMax3 simulation that fits in memory",
        f"At the {BYTES_PER_CELL:.1f} bytes per cell measured on this M4. Below "
        f"roughly 240M cells a same-cost PC is {min_ratio:.1f}-{max_ratio:.1f}x "
        "faster; above it the "
        "largest consumer NVIDIA card cannot run the problem at all.",
        os.path.join(HERE, "apple-capacity.svg"),
        [("Apple, unified memory", "measured"), ("NVIDIA, VRAM", "other")],
        y_label="largest simulation (M cells)",
    )

    measured = read_gpus(os.path.join(HERE, "gpus.txt"), include_apple=True)
    measured.append(read_oommf(os.path.join(HERE, "oommf4M.txt")))
    measured.sort(key=lambda row: row["value"])
    render(
        measured,
        "MuMax3 GPU benchmark, 4.19M cells (2048x2048)",
        "All GPU bars are measured results from bench/gpus.txt. Apple M4 uses "
        "the Metal backend; the other GPU bars use CUDA. OOMMF is the CPU "
        "reference used by the original gnuplot chart.",
        os.path.join(HERE, "..", "doc", "static", "gpus.svg"),
        [
            ("Apple M4, Metal", "measured"),
            ("CUDA GPU / OOMMF CPU", "other"),
        ],
    )


if __name__ == "__main__":
    main()
