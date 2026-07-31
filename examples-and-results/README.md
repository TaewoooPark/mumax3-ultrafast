# Working examples and results

This directory pairs the 15 input scripts published on the
[official mumax³ examples page](https://mumax.github.io/examples.html) with
the outputs produced by this repository's macOS/Metal build.

- [`official-examples/`](official-examples/) contains the exact `.mx3` input,
  any required fixture, the complete `output.out/`, console log, and `PASS`
  marker for every example.
- [`RESULTS.md`](RESULTS.md) is the human-readable execution report and links
  each problem directly to its result.
- [`SOURCE_AND_LICENSE.md`](SOURCE_AND_LICENSE.md) records upstream source,
  attribution, and the GPL-3.0-or-later license covering the example inputs.
- [`run_official_examples.py`](run_official_examples.py) downloads the live
  official page, extracts all 15 scripts without editing them, runs each in
  isolation, and validates its artifacts.

The checked-in scripts are byte-for-byte identical to the 15 example blocks in
the GPL-licensed upstream
[`doc/templates/examples-template.html`](https://github.com/mumax/3/blob/master/doc/templates/examples-template.html).
The archived rendered page has SHA-256
`c5ec8c43ad74f11724d9223052675fbcc00e707650a46a262a2f1e7280b2ef8a`.

Examples 4 and 5 refer to files that the published page does not provide.
Their folders therefore include the compatible regression fixtures identified
in the report:

- `test/testdata/mask.png` copied as `mask.png`
- `test/testdata/randommag4x4x1.ovf` copied as `myfile.ovf`

## Re-run all examples

```bash
make
python3 examples-and-results/run_official_examples.py \
  --binary "$(go env GOPATH)/bin/mumax3"
```

New runs are written under the ignored
`examples-and-results/runs/<timestamp>/` directory. A run is successful only
when every process exits normally and all expected OVF/table artifacts are
present, non-empty, and numerically finite.
