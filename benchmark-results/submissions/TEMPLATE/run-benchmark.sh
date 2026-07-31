#!/bin/bash

set -euo pipefail

submission_directory="$(cd "$(dirname "$0")" && pwd)"
repository_root="$(git -C "$submission_directory" rev-parse --show-toplevel)"
cd "$submission_directory"

mumax3_binary="${MUMAX3_BIN:-}"
if [[ -z "$mumax3_binary" ]]; then
  mumax3_binary="$(command -v mumax3 || true)"
fi
if [[ -z "$mumax3_binary" || ! -x "$mumax3_binary" ]]; then
  echo "mumax3 was not found. Install it or set MUMAX3_BIN to an executable." >&2
  exit 1
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Comparable submissions require a native Apple Silicon macOS session." >&2
  exit 1
fi

generated_paths=(
  hardware-profile.txt
  macos-version.txt
  power-settings.txt
  binary-sha256.txt
  repository-commit.txt
  statistics.json
)
for run_number in 01 02 03 04 05; do
  generated_paths+=("run-${run_number}.log" "run-${run_number}.out")
done
for generated_path in "${generated_paths[@]}"; do
  if [[ -e "$generated_path" ]]; then
    echo "Refusing to overwrite $generated_path. Use a fresh submission directory." >&2
    exit 1
  fi
done

LC_ALL=C LANG=C system_profiler SPHardwareDataType SPDisplaysDataType |
  sed -E '/Serial Number|Hardware UUID|Provisioning UDID|Activation Lock Status/d' \
  > hardware-profile.txt
sw_vers > macos-version.txt
pmset -g custom > power-settings.txt
shasum -a 256 "$mumax3_binary" | awk '{print $1}' > binary-sha256.txt
git -C "$repository_root" rev-parse HEAD > repository-commit.txt

for run_number in 01 02 03 04 05; do
  "$mumax3_binary" -http="" -f -o "run-${run_number}.out" benchmark-4m.mx3 \
    2>&1 | tee "run-${run_number}.log"
done

(
  cd "$repository_root"
  go run ./benchmark-results/tools/summarize \
    -directory "$submission_directory" \
    -output "$submission_directory/statistics.json"
)

echo
echo "Benchmark complete. Review the generated files before committing them."
