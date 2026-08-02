#!/usr/bin/env python3
"""Render measured and explicitly modelled Apple benchmark charts as SVG.

The renderer uses only Python's standard library.  Run it after
``bench/apple_project.py``; ``doc/Makefile`` already invokes them in that order.
Model bars carry hatching and their low/high workload-model envelope is shown
as a whisker.  A whisker is never labelled as statistical uncertainty.
"""

import math
import os
import shlex


HERE = os.path.dirname(os.path.abspath(__file__))

FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
COLOR_MEASURED = "#0b4f9e"
COLOR_MODELED = "#4b9ae8"
COLOR_OTHER = "#8d9299"
COLOR_NVIDIA_PRICE = "#5f6f7a"
COLOR_RANGE = "#173a5e"
COLOR_AXIS = "#5f6368"
COLOR_TEXT = "#202124"
HOST_USD = 800


def escape(value):
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def read_apple(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            (
                bandwidth,
                cores,
                central,
                low,
                high,
                crossover,
                status,
                name,
            ) = shlex.split(line)
            if status not in {"measured", "modeled"}:
                raise ValueError(f"unknown Apple result status: {status}")
            rows.append(
                {
                    "name": name,
                    "value": float(central) / 1e6,
                    "low": float(low) / 1e6,
                    "high": float(high) / 1e6,
                    # Square mesh below which the per-evaluation overhead floor
                    # binds instead of bandwidth, so extra GPU width buys nothing.
                    "crossover_mesh": float(crossover),
                    "status": status,
                    "bandwidth": float(bandwidth),
                    "cores": int(cores),
                }
            )
    rows.sort(key=lambda row: row["value"])
    return rows


def read_gpus(path, include_apple=False):
    """Read measured results from bench/gpus.txt."""
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = shlex.split(line)
            # The measured row names the actual chassis rather than using an
            # abstract "Apple" prefix. Metal is unique in this CUDA dataset.
            is_apple = fields[3].endswith("(Metal)")
            if is_apple and not include_apple:
                continue
            value = float(fields[1]) / 1e6
            rows.append(
                {
                    "name": fields[3],
                    "value": value,
                    "low": value,
                    "high": value,
                    "status": "measured" if is_apple else "other",
                }
            )
    return rows


def read_oommf(path):
    """Read the CPU reference using the formula in bench/gpus.gplot."""
    with open(path) as handle:
        size, steps, wall = handle.read().split()
    value = 4 * int(size) ** 2 * int(steps) / float(wall) / 1e6
    return {
        "name": "OOMMF (CPU)",
        "value": value,
        "low": value,
        "high": value,
        "status": "other",
    }


def nice_ticks(top, count=7):
    """Return rounded ticks covering [0, top]."""
    raw = top / count
    magnitude = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 2.5, 5, 10):
        step = factor * magnitude
        if step >= raw:
            break
    axis_max = math.ceil(top / step) * step
    return [index * step for index in range(round(axis_max / step) + 1)]


def wrap(value, width):
    """Greedy text wrapping for SVG subtitles and captions."""
    lines = []
    current = ""
    for word in value.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def add_defs(add):
    add(
        '<defs><pattern id="hatch" width="5" height="5" '
        'patternTransform="rotate(45)" patternUnits="userSpaceOnUse">'
        '<line x1="0" y1="0" x2="0" y2="5" stroke="#ffffff" '
        'stroke-width="2.1"/></pattern></defs>'
    )


def add_whisker(add, center_x, low, high, y_of, cap):
    """Draw a non-statistical model-envelope whisker."""
    low_y = y_of(low)
    high_y = y_of(high)
    add(
        f'<line x1="{center_x:.1f}" y1="{high_y:.1f}" '
        f'x2="{center_x:.1f}" y2="{low_y:.1f}" '
        f'stroke="{COLOR_RANGE}" stroke-width="1.4"/>'
    )
    for y in (low_y, high_y):
        add(
            f'<line x1="{center_x - cap:.1f}" y1="{y:.1f}" '
            f'x2="{center_x + cap:.1f}" y2="{y:.1f}" '
            f'stroke="{COLOR_RANGE}" stroke-width="1.4"/>'
        )


def render(
    rows,
    title,
    subtitle,
    path,
    legend,
    y_label="throughput (M cell-evals/s)",
    value_unit="M cell-evals/s",
    show_values=False,
    value_format="{:.0f}",
):
    bar_slot = 26 if len(rows) <= 24 else 15
    left, right = 78, 26
    title_room = 8 * len(title)
    legend_room = sum(31 + 7 * len(label) for label, _ in legend)
    plot_w = max(bar_slot * len(rows), title_room, legend_room)
    bar_slot = plot_w / len(rows)
    plot_h = max(280, int(plot_w * 0.45))
    width = left + plot_w + right

    longest = max(len(row["name"]) for row in rows)
    label_room = int(6.0 * longest) + 62
    subtitle_lines = wrap(subtitle, max(40, int((width - left - right) / 5.9)))
    top = 40 + 16 * len(subtitle_lines) + 18
    height = top + plot_h + label_room

    top_value = max(row.get("high", row["value"]) for row in rows)
    ticks = nice_ticks(top_value)
    axis_max = ticks[-1]

    def y_of(value):
        return top + plot_h - (value / axis_max) * plot_h

    out = []
    add = out.append
    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">'
    )
    add_defs(add)
    add(f'<rect width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>')
    add(
        f'<text x="{left}" y="30" font-family="{FONT}" font-size="15" '
        f'font-weight="600" fill="{COLOR_TEXT}">{escape(title)}</text>'
    )
    for index, line in enumerate(subtitle_lines):
        add(
            f'<text x="{left}" y="{48 + 15 * index}" font-family="{FONT}" '
            f'font-size="11" fill="{COLOR_AXIS}">{escape(line)}</text>'
        )

    for tick in ticks:
        y = y_of(tick)
        add(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w:.1f}" '
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
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w:.1f}" '
        f'y2="{top + plot_h}" stroke="{COLOR_AXIS}" stroke-width="1"/>'
    )
    label_y = top + plot_h / 2
    add(
        f'<text x="18" y="{label_y:.1f}" font-family="{FONT}" font-size="11" '
        f'fill="{COLOR_TEXT}" text-anchor="middle" '
        f'transform="rotate(-90 18 {label_y:.1f})">{escape(y_label)}</text>'
    )

    palette = {
        "measured": COLOR_MEASURED,
        "modeled": COLOR_MODELED,
        "other": COLOR_OTHER,
    }
    bar_w = min(bar_slot * 0.62, 22)
    for index, row in enumerate(rows):
        x = left + index * bar_slot + (bar_slot - bar_w) / 2
        center_x = x + bar_w / 2
        y = y_of(row["value"])
        low = row.get("low", row["value"])
        high = row.get("high", row["value"])
        tooltip = f'{row["name"]}: {row["value"]:.1f} {value_unit}'
        if not math.isclose(low, high):
            tooltip += f"; workload/model range {low:.1f}–{high:.1f} (not CI)"
        add(f'<g><title>{escape(tooltip)}</title>')
        add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{top + plot_h - y:.1f}" fill="{palette[row["status"]]}"/>'
        )
        if row["status"] == "modeled":
            add(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{top + plot_h - y:.1f}" fill="url(#hatch)"/>'
            )
        if not math.isclose(low, high):
            add_whisker(add, center_x, low, high, y_of, min(7, bar_w * 0.42))
        if show_values:
            add(
                f'<text x="{center_x:.1f}" y="{y_of(high) - 5:.1f}" '
                f'font-family="{FONT}" font-size="9" fill="{COLOR_TEXT}" '
                f'text-anchor="middle">'
                f'{value_format.format(row["value"])}</text>'
            )
        add("</g>")

        text_y = top + plot_h + 7
        weight = ' font-weight="600"' if row["status"] == "measured" else ""
        fill = COLOR_TEXT if row["status"] == "measured" else COLOR_AXIS
        add(
            f'<text x="{center_x:.1f}" y="{text_y:.1f}" font-family="{FONT}" '
            f'font-size="10" fill="{fill}"{weight} text-anchor="end" '
            f'transform="rotate(-90 {center_x:.1f} {text_y:.1f})">'
            f'{escape(row["name"])}</text>'
        )

    lx, ly = left, top + plot_h + label_room - 22
    for label, status in legend:
        add(
            f'<rect x="{lx:.1f}" y="{ly - 9}" width="11" height="11" '
            f'fill="{palette[status]}"/>'
        )
        if status == "modeled":
            add(
                f'<rect x="{lx:.1f}" y="{ly - 9}" width="11" height="11" '
                f'fill="url(#hatch)"/>'
            )
            add_whisker(add, lx + 5.5, 0, 1, lambda value: ly + 3 - 13 * value, 4)
        add(
            f'<text x="{lx + 17:.1f}" y="{ly}" font-family="{FONT}" '
            f'font-size="11" fill="{COLOR_TEXT}">{escape(label)}</text>'
        )
        lx += 26 + 7.0 * len(label)
    add("</svg>")

    with open(path, "w") as handle:
        handle.write("\n".join(out) + "\n")
    print(f"wrote {path} ({len(rows)} bars)")


def read_latency(path):
    """Speedups on latency-bound workloads, from bench/latency.txt.

    Two bars per workload: the shipped default, and the same binary with the
    workload's opt-in flag enabled. Both are measured, so neither is hatched -
    hatching in these charts means "projected", which none of these are.
    """
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, mesh, baseline, default, optin, flag = shlex.split(line)[:6]
            if baseline == "-":
                # The sweep row has no single-run baseline: -j 1 is the baseline.
                baseline, default = default, None
            for value, label, status in (
                (default, "default", "measured"),
                (optin, flag, "other"),
            ):
                if value in (None, "-"):
                    continue
                rows.append(
                    {
                        "name": f"{name} {mesh}^2 [{label}]",
                        "value": float(baseline) / float(value),
                        "status": status,
                    }
                )
    return rows


def read_curve(path):
    """Measured throughput against problem size, from bench/curve.txt.

    One machine, one binary. This is what makes the projection model's regime
    boundary auditable: the per-evaluation overhead floor it uses is read off the
    small-mesh end of this curve.
    """
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            mesh, cells, value, spread, per_eval, window = line.split()
            rows.append(
                {
                    "name": f"{mesh}^2",
                    "value": float(value) / 1e6,
                    "status": "measured",
                    "cells": int(float(cells)),
                    "spread": float(spread),
                    "per_eval": float(per_eval),
                }
            )
    return rows


def read_tiers(path, apple_rows):
    """Read launch configurations; Apple references resolve into the model."""
    apple_by_name = {row["name"]: row for row in apple_rows}
    tiers = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tier, vendor, price, reference, label = shlex.split(line)
            price = int(price)
            total = price + (HOST_USD if vendor == "nvidia" else 0)
            if vendor == "apple":
                source = apple_by_name[reference]
                value, low, high = source["value"], source["low"], source["high"]
            elif vendor == "nvidia":
                value = float(reference)
                low = high = value
            else:
                raise ValueError(f"unknown tier vendor: {vendor}")
            tiers.append(
                {
                    "tier": tier,
                    "vendor": vendor,
                    "total": total,
                    "value": value,
                    "low": low,
                    "high": high,
                    "label": label,
                }
            )
    order = []
    for row in tiers:
        if row["tier"] not in order:
            order.append(row["tier"])
    for tier in order:
        members = [row for row in tiers if row["tier"] == tier]
        if sorted(row["vendor"] for row in members) != ["apple", "nvidia"]:
            raise ValueError(f"tier {tier!r} must have one Apple and one NVIDIA row")
    return tiers, order


def render_price_tiers(path, tiers, order):
    """Grouped launch-price context; groups are explicitly not equal-budget."""
    left, right = 78, 26
    group_w, bar_w, gap = 236, 84, 12
    plot_w = group_w * len(order)
    plot_h = 320
    subtitle = (
        f"US base launch configurations; groups are not equal-budget. NVIDIA adds an ${HOST_USD} host. "
        "NVIDIA bars are measured; Apple bars use the proxy model. The M4 anchor was measured in a "
        "MacBook Air and transferred to the same-chip Mac mini without a chassis adjustment."
    )
    subtitle_lines = wrap(subtitle, int(plot_w / 7.2))
    top = 42 + 15 * len(subtitle_lines) + 20
    width = left + plot_w + right
    height = top + plot_h + 150

    top_value = max(row["high"] for row in tiers)
    ticks = nice_ticks(top_value, count=5)
    axis_max = ticks[-1]

    def y_of(value):
        return top + plot_h - (value / axis_max) * plot_h

    out = []
    add = out.append
    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
    )
    add_defs(add)
    add(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    add(
        f'<text x="{left}" y="30" font-family="{FONT}" font-size="15" '
        f'font-weight="600" fill="{COLOR_TEXT}">MuMax3 at 4.19M cells: '
        "launch-price context (not equal-budget)</text>"
    )
    for index, line in enumerate(subtitle_lines):
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
            f'font-size="10" fill="{COLOR_AXIS}" text-anchor="end">{tick:g}</text>'
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
        "throughput (M cell-evals/s)</text>"
    )

    for group, tier in enumerate(order):
        members = [row for row in tiers if row["tier"] == tier]
        members.sort(key=lambda row: 0 if row["vendor"] == "apple" else 1)
        base = left + group * group_w
        for slot, row in enumerate(members):
            x = base + (group_w - (2 * bar_w + gap)) / 2 + slot * (bar_w + gap)
            center_x = x + bar_w / 2
            y = y_of(row["value"])
            is_apple = row["vendor"] == "apple"
            fill = COLOR_MODELED if is_apple else COLOR_NVIDIA_PRICE
            add(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" '
                f'height="{top + plot_h - y:.1f}" fill="{fill}"/>'
            )
            if is_apple:
                add(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" '
                    f'height="{top + plot_h - y:.1f}" fill="url(#hatch)"/>'
                )
                if not math.isclose(row["low"], row["high"]):
                    add_whisker(add, center_x, row["low"], row["high"], y_of, 12)

            if is_apple and not math.isclose(row["low"], row["high"]):
                value_label = (
                    f'{row["value"]:.0f} [{row["low"]:.0f}–{row["high"]:.0f}]'
                )
                value_y = y_of(row["high"]) - 7
            else:
                value_label = f'{row["value"]:.0f}'
                value_y = y - 7
            add(
                f'<text x="{center_x:.1f}" y="{value_y:.1f}" '
                f'font-family="{FONT}" font-size="9" font-weight="600" '
                f'fill="{COLOR_TEXT}" text-anchor="middle">{value_label}</text>'
            )
            add(
                f'<text x="{center_x:.1f}" y="{top + plot_h + 14:.1f}" '
                f'font-family="{FONT}" font-size="10" fill="{COLOR_TEXT}" '
                f'text-anchor="middle">{escape(row["label"])}</text>'
            )
            add(
                f'<text x="{center_x:.1f}" y="{top + plot_h + 29:.1f}" '
                f'font-family="{FONT}" font-size="10" fill="{COLOR_AXIS}" '
                f'text-anchor="middle">${row["total"]:,}; '
                f'{row["value"] / row["total"] * 1000:.0f} M/s/$1k</text>'
            )

        apple = next(row for row in members if row["vendor"] == "apple")
        nvidia = next(row for row in members if row["vendor"] == "nvidia")
        ratio = nvidia["value"] / apple["value"]
        ratio_low = nvidia["value"] / apple["high"]
        ratio_high = nvidia["value"] / apple["low"]
        if math.isclose(ratio_low, ratio_high):
            ratio_label = f"throughput ratio PC/Apple: {ratio:.1f}x"
        else:
            ratio_label = (
                f"PC/Apple: {ratio:.1f}x [{ratio_low:.1f}–{ratio_high:.1f}]"
            )
        add(
            f'<text x="{base + group_w / 2:.1f}" '
            f'y="{top + plot_h + 53:.1f}" font-family="{FONT}" font-size="10" '
            f'fill="{COLOR_AXIS}" text-anchor="middle">{ratio_label}</text>'
        )
        add(
            f'<text x="{base + group_w / 2:.1f}" '
            f'y="{top + plot_h + 70:.1f}" font-family="{FONT}" font-size="10" '
            f'font-weight="600" fill="{COLOR_TEXT}" text-anchor="middle">'
            f'{escape(tier)}</text>'
        )

    lx, ly = left, height - 22
    for label, fill, hatched in [
        ("Apple proxy model; whisker = model range, not CI", COLOR_MODELED, True),
        ("NVIDIA measured", COLOR_NVIDIA_PRICE, False),
    ]:
        add(f'<rect x="{lx}" y="{ly - 9}" width="11" height="11" fill="{fill}"/>')
        if hatched:
            add(
                f'<rect x="{lx}" y="{ly - 9}" width="11" height="11" '
                f'fill="url(#hatch)"/>'
            )
            add_whisker(add, lx + 5.5, 0, 1, lambda value: ly + 3 - 13 * value, 4)
        add(
            f'<text x="{lx + 17}" y="{ly}" font-family="{FONT}" font-size="11" '
            f'fill="{COLOR_TEXT}">{escape(label)}</text>'
        )
        lx += 26 + 7.0 * len(label)
    add("</svg>")
    with open(path, "w") as handle:
        handle.write("\n".join(out) + "\n")
    print(f"wrote {path} ({len(order)} price groups)")


BYTES_PER_CELL = 133.1


def read_capacity(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vendor, gb, fraction, label = shlex.split(line)
            cells = int(gb) * (1024 ** 3) * float(fraction) / BYTES_PER_CELL
            value = cells / 1e6
            rows.append(
                {
                    "name": label,
                    "value": value,
                    "low": value,
                    "high": value,
                    "status": "modeled" if vendor == "apple" else "other",
                    "mesh": int(cells ** 0.5),
                }
            )
    rows.sort(key=lambda row: row["value"])
    return rows


def main():
    apple = read_apple(os.path.join(HERE, "apple.txt"))
    render(
        apple,
        "MuMax3 Apple-Silicon model at 4.19M cells (2048x2048)",
        "M4 10c is the measured MacBook Air anchor. All other central values combine "
        "Metal STREAM, same-build llama.cpp TG/PP, bandwidth, GPU width, and (for M5) "
        "MLX. Whiskers are workload/model envelopes, not confidence intervals.",
        os.path.join(HERE, "apple.svg"),
        [
            ("M4 10c MacBook Air, measured", "measured"),
            ("proxy ensemble; whisker = workload/model range, not CI", "modeled"),
        ],
        show_values=True,
    )

    combined = read_gpus(os.path.join(HERE, "gpus.txt")) + apple
    combined.sort(key=lambda row: row["value"])
    render(
        combined,
        "MuMax3 at 4.19M cells: Apple proxy model vs measured CUDA GPUs",
        "CUDA bars come from bench/gpus.txt. Apple M4 10c was measured on the Metal "
        "backend; all other Apple central values and whiskers are proxy-model results, "
        "not MuMax3 benchmarks or confidence intervals.",
        os.path.join(HERE, "apple-vs-gpus.svg"),
        [
            ("M4 10c Air, measured", "measured"),
            ("Apple proxy ensemble + model range", "modeled"),
            ("CUDA, measured", "other"),
        ],
    )

    tiers, order = read_tiers(os.path.join(HERE, "price_tiers.txt"), apple)
    render_price_tiers(os.path.join(HERE, "apple-price-tiers.svg"), tiers, order)

    capacity = read_capacity(os.path.join(HERE, "capacity.txt"))
    for row in capacity:
        row["name"] = f'{row["name"]} ({row["mesh"]}^2)'
    render(
        capacity,
        "Estimated largest MuMax3 simulation that fits in memory",
        f"Uses {BYTES_PER_CELL:.1f} B/cell measured on the M4 Air and explicit usable-memory "
        "assumptions: 68% for Apple unified memory, 92% for NVIDIA VRAM. These are capacity "
        "estimates, not throughput results.",
        os.path.join(HERE, "apple-capacity.svg"),
        [("Apple capacity estimate", "modeled"), ("NVIDIA capacity estimate", "other")],
        y_label="largest simulation (M cells)",
        value_unit="M cells",
        show_values=True,
    )

    latency = read_latency(os.path.join(HERE, "latency.txt"))
    render(
        latency,
        "Speedup on latency-bound MuMax3 workloads (Apple M4 10c)",
        "Same binary against commit a02cfd9b, medians of three interleaved runs in one "
        "session. These are the workloads whose step rate is set by host round trips "
        "rather than by arithmetic. The published 4.19M-cell point is bandwidth-bound "
        "and does not move: 1.0498e8 before against 1.0551e8 after, inside the 0.91% CV "
        "of the pooled anchor. The sweep bar compares -j 3 against -j 1, so it is aggregate "
        "throughput, not single-run speed.",
        os.path.join(HERE, "apple-latency.svg"),
        [
            ("shipped default", "measured"),
            ("with the named opt-in enabled", "other"),
        ],
        y_label="speedup over a02cfd9b",
        value_unit="x",
        show_values=True,
        value_format="{:.2f}x",
    )

    curve = read_curve(os.path.join(HERE, "curve.txt"))
    render(
        curve,
        "Measured MuMax3 throughput against problem size (Apple M4 10c)",
        "One machine, one binary, one size per fresh process, median of three, with the "
        "timed window sized per mesh so every point measures seconds of steady state. "
        "Three regimes, all three set by code paths rather than by the hardware: an "
        "overhead floor of 187 us per evaluation at and below 128^2, a peak at 256^2 "
        "where padded 512^2 is inside the default VkFFT gate, a dip at 512^2 where "
        "padded 1024^2 is not, the MPSGraph plateau that the published 4.19M-cell point "
        "sits on, and a decline past 4096^2. The published point is not this machine's "
        "fastest: 256^2 is 1.39x better per cell.",
        os.path.join(HERE, "apple-size-scaling.svg"),
        [("measured, median of three fresh processes", "measured")],
        show_values=True,
        value_format="{:.1f}",
    )

    crossover = [
        {
            "name": row["name"],
            "value": row["crossover_mesh"],
            "status": row["status"],
        }
        for row in apple
    ]
    render(
        crossover,
        "Smallest square mesh where extra Apple GPU width starts to pay",
        "Below its own bar a chip is limited by the 187 us per-evaluation overhead floor "
        "measured on the M4, not by bandwidth, so every chip here converges to the same "
        "ceiling and a wider GPU buys nothing. Above it the projected bandwidth-limited "
        "rate applies. Derived from each chip's projected 4.19M-cell rate and one measured "
        "floor, so only the M4 bar rests on a measurement of that chip; the floor itself "
        "is treated as chip-independent, which a faster CPU or cheaper dispatch would "
        "lower. Read it as an order of magnitude, not a threshold.",
        os.path.join(HERE, "apple-crossover.svg"),
        [
            ("M4 10c, floor measured on this chip", "measured"),
            ("projected rate x measured floor", "modeled"),
        ],
        y_label="crossover mesh (cells per side)",
        value_unit="cells per side",
        show_values=True,
    )

    measured = read_gpus(os.path.join(HERE, "gpus.txt"), include_apple=True)
    measured.append(read_oommf(os.path.join(HERE, "oommf4M.txt")))
    measured.sort(key=lambda row: row["value"])
    render(
        measured,
        "MuMax3 GPU benchmark, 4.19M cells (2048x2048)",
        "All GPU bars are measured results from bench/gpus.txt. The MacBook Air M4 "
        "10c uses the Metal backend; other GPU bars use CUDA. OOMMF is the CPU reference.",
        os.path.join(HERE, "..", "doc", "static", "gpus.svg"),
        [
            ("MacBook Air M4 10c, Metal", "measured"),
            ("CUDA GPU / OOMMF CPU", "other"),
        ],
    )


if __name__ == "__main__":
    main()
