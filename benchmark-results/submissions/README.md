# Submit an Apple Silicon benchmark

Measured results from other Apple Silicon Macs are welcome. Each accepted
submission replaces the matching orange projection in the comparison chart
with a blue measurement, while preserving the raw evidence needed to review or
repeat it.

## Submission procedure

1. Install this repository's current `master` branch and confirm that
   `mumax3 -test` passes.
2. Copy [`TEMPLATE/`](TEMPLATE/) to
   `benchmark-results/submissions/<chip>-<mac-model>-YYYYMMDD/`.
3. Complete every applicable field in `metadata.json`. Do not include a serial
   number, hardware UUID, host name, account name, or other personal data.
4. Connect the Mac to AC power, disable Low Power Mode, close substantial
   background workloads, and run `./run-benchmark.sh`. The script performs five
   headless runs of the official 4-million-cell workload, records sanitized
   hardware and power information, and creates `statistics.json`.
5. Add the measurement to [`benchmark-results/registry.json`](../registry.json).
   Set `replaces_estimate_label` to the exact matching label in
   [`apple-silicon-estimates.json`](../apple-silicon-estimates.json), when one
   exists.

   ```json
   {
     "label": "Apple M3 Max 40-core GPU — MacBook Pro (measured)",
     "backend": "Metal",
     "result_path": "benchmark-results/submissions/m3-max-macbook-pro-YYYYMMDD/statistics.json",
     "evidence_path": "benchmark-results/submissions/m3-max-macbook-pro-YYYYMMDD/README.md",
     "replaces_estimate_label": "Apple M3 Max 40-core GPU"
   }
   ```

   Insert the object into the existing `measurements` array; use the real date
   and paths from the submitted directory.
6. Regenerate the chart from the repository root:

   ```bash
   go run ./benchmark-results/tools/chart
   ```

7. Open a pull request using the
   [**Benchmark submission** template](../../.github/PULL_REQUEST_TEMPLATE/benchmark_submission.md).
   Include the whole submission directory, the registry change, and the
   regenerated SVG.

The benchmark must run on physical Apple hardware. Virtual machines, remote
machines whose exact configuration cannot be documented, altered benchmark
physics, `-sync`, Web UI overhead, and cherry-picked single runs are not
accepted as comparable measurements. Results remain useful when they are lower
than expected; please submit all five consecutive runs.

## What reviewers check

- The workload matches `TEMPLATE/benchmark-4m.mx3` byte for byte.
- Five raw `benchmark.txt` files and five complete console logs are present.
- The median in `statistics.json` is reproducible from those five files.
- The backend is Metal, the machine is physical Apple Silicon, and the relevant
  build and environment details are recorded.
- No personal or device-unique identifiers are committed.

The first verified measurement is available as a complete example in
[`apple-m4-macbook-air-20260731/`](../apple-m4-macbook-air-20260731/).
