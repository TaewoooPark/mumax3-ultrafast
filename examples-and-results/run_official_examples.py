#!/usr/bin/env python3
"""Run and validate every script embedded in mumax.github.io/examples.html."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SOURCE_URL = "https://mumax.github.io/examples.html"
SOURCE_TEMPLATE_URL = (
    "https://github.com/mumax/3/blob/master/doc/templates/examples-template.html"
)
LICENSE_EXPRESSION = "GPL-3.0-or-later"
LICENSE_URL = "https://github.com/mumax/3/blob/master/LICENSE"
EXAMPLE_NAMES = {
    1: "standard-problem-4",
    2: "standard-problem-2",
    3: "hysteresis",
    4: "geometry",
    5: "initial-magnetization",
    6: "rotating-cheese",
    7: "regions",
    8: "slicing-output",
    9: "mfm",
    10: "pma-racetrack",
    11: "py-racetrack",
    12: "voronoi",
    13: "rkky",
    14: "slonczewski-stt",
    15: "spinning-hard-disk",
}
EXPECTED_OVF_COUNTS = {
    1: 7,
    2: 1,
    3: 0,
    4: 16,
    5: 13,
    6: 4,
    7: 6,
    8: 4,
    9: 4,
    10: 6,
    11: 11,
    12: 6,
    13: 0,
    14: 11,
    15: 7,
}
EXPECTED_TABLE_LINES = {
    1: 102,
    3: 501,
    11: 52,
    13: 361,
    14: 102,
}
EXAMPLE_RE = re.compile(
    rb"<a\s+id=example(\d+)></a><pre>(.*?)</pre>", re.IGNORECASE | re.DOTALL
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repository_root()
    default_binary = Path.home() / "go" / "bin" / "mumax3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=default_binary)
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root
        / "examples-and-results"
        / "runs"
        / datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600,
        help="per-example timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--example",
        type=int,
        action="append",
        choices=range(1, 16),
        help="run only this example; repeat to select more than one",
    )
    return parser.parse_args()


def fetch_source(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "mumax3-for-mac-official-example-runner/1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def extract_examples(source: bytes) -> dict[int, str]:
    examples: dict[int, str] = {}
    for raw_number, raw_code in EXAMPLE_RE.findall(source):
        number = int(raw_number)
        code = html.unescape(raw_code.decode("utf-8"))
        examples[number] = code.rstrip() + "\n"

    expected = set(EXAMPLE_NAMES)
    found = set(examples)
    if found != expected:
        raise RuntimeError(
            f"official example set changed: expected {sorted(expected)}, "
            f"found {sorted(found)}"
        )
    return examples


def add_fixture(root: Path, case_dir: Path, number: int) -> str | None:
    if number == 4:
        source = root / "test" / "testdata" / "mask.png"
        target = case_dir / "mask.png"
    elif number == 5:
        source = root / "test" / "testdata" / "randommag4x4x1.ovf"
        target = case_dir / "myfile.ovf"
    else:
        return None

    shutil.copy2(source, target)
    return f"{source.relative_to(root)} -> {target.name}"


def validate_artifacts(case_dir: Path, number: int) -> tuple[list[str], dict[str, int]]:
    output_dir = case_dir / "output.out"
    errors: list[str] = []
    ovf_count = len(list(output_dir.glob("*.ovf")))
    expected_ovf_count = EXPECTED_OVF_COUNTS[number]
    if ovf_count != expected_ovf_count:
        errors.append(f"expected {expected_ovf_count} OVF files, found {ovf_count}")

    table_path = output_dir / "table.txt"
    table_lines = 0
    if number in EXPECTED_TABLE_LINES:
        if not table_path.is_file():
            errors.append("expected table.txt, but it was not created")
        else:
            lines = table_path.read_text(encoding="utf-8").splitlines()
            table_lines = len(lines)
            expected_lines = EXPECTED_TABLE_LINES[number]
            if table_lines != expected_lines:
                errors.append(
                    f"expected {expected_lines} table lines, found {table_lines}"
                )
            for line_number, line in enumerate(lines, start=1):
                if not line or line.startswith("#"):
                    continue
                try:
                    values = [float(value) for value in line.split()]
                except ValueError:
                    errors.append(f"table line {line_number} is not numeric")
                    break
                if not all(value == value and abs(value) != float("inf") for value in values):
                    errors.append(f"table line {line_number} contains NaN or infinity")
                    break

    for artifact in output_dir.iterdir() if output_dir.is_dir() else []:
        if artifact.is_file() and artifact.stat().st_size == 0:
            errors.append(f"empty output artifact: {artifact.name}")

    return errors, {"ovf_files": ovf_count, "table_lines": table_lines}


def run_example(
    root: Path,
    binary: Path,
    run_dir: Path,
    number: int,
    code: str,
    timeout: float,
) -> dict[str, object]:
    slug = EXAMPLE_NAMES[number]
    case_dir = run_dir / f"example{number:02d}-{slug}"
    case_dir.mkdir(parents=True)
    input_path = case_dir / f"example{number}.mx3"
    input_path.write_text(code, encoding="utf-8")
    fixture = add_fixture(root, case_dir, number)

    command = [
        str(binary),
        "-http=",
        "-f",
        "-o",
        "output.out",
        input_path.name,
    ]
    started = time.monotonic()
    timed_out = False
    return_code: int | None
    with (case_dir / "run.log").open("wb") as log_file:
        log_file.write(("$ " + " ".join(command) + "\n").encode())
        log_file.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=case_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = None

    elapsed = time.monotonic() - started
    validation_errors: list[str] = []
    artifact_counts = {"ovf_files": 0, "table_lines": 0}
    if return_code == 0 and not timed_out:
        validation_errors, artifact_counts = validate_artifacts(case_dir, number)
    passed = return_code == 0 and not timed_out and not validation_errors
    marker = case_dir / ("PASS" if passed else "FAIL")
    marker.write_text(
        f"elapsed_seconds={elapsed:.3f}\n"
        f"return_code={return_code}\n"
        f"timed_out={str(timed_out).lower()}\n",
        encoding="utf-8",
    )
    return {
        "number": number,
        "name": slug,
        "status": "PASS" if passed else "FAIL",
        "return_code": return_code,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed, 3),
        "fixture": fixture,
        "artifact_counts": artifact_counts,
        "validation_errors": validation_errors,
        "directory": str(case_dir.relative_to(run_dir)),
    }


def main() -> int:
    args = parse_args()
    root = repository_root()
    binary = args.binary.expanduser().resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"mumax3 binary not found: {binary}")

    run_dir = args.results_dir.expanduser().resolve()
    if run_dir.exists():
        raise FileExistsError(f"results directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    source = fetch_source(args.source_url)
    source_path = run_dir / "official-examples.html"
    source_path.write_bytes(source)
    examples = extract_examples(source)
    selected = sorted(set(args.example or EXAMPLE_NAMES))

    source_sha256 = hashlib.sha256(source).hexdigest()
    metadata = {
        "source_url": args.source_url,
        "source_template_url": SOURCE_TEMPLATE_URL,
        "source_sha256": source_sha256,
        "license_expression": LICENSE_EXPRESSION,
        "license_url": LICENSE_URL,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "selected_examples": selected,
        "timeout_seconds": args.timeout,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    results: list[dict[str, object]] = []
    for position, number in enumerate(selected, start=1):
        name = EXAMPLE_NAMES[number]
        print(f"[{position}/{len(selected)}] example{number}: {name}", flush=True)
        result = run_example(
            root, binary, run_dir, number, examples[number], args.timeout
        )
        results.append(result)
        print(
            f"  {result['status']} ({result['elapsed_seconds']:.3f}s)",
            flush=True,
        )

    passed = sum(result["status"] == "PASS" for result in results)
    summary = {
        **metadata,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "failed": len(results) - passed,
        "total": len(results),
        "results": results,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Results: {run_dir}")
    print(f"Summary: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"Official example runner error: {error}", file=sys.stderr)
        sys.exit(2)
