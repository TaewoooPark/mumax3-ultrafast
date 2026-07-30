# Official examples QA

This directory contains the reproducible macOS/Metal QA harness for the 15
scripts published at <https://mumax.github.io/examples.html>.

Run all examples:

```bash
python3 qa/examples/run_official_examples.py \
  --binary "$(go env GOPATH)/bin/mumax3"
```

The harness downloads and archives the current official HTML, extracts
`example1` through `example15` without changing their code, and runs every
script in an isolated directory. Results, inputs, logs, timing, and a JSON
summary are stored under `qa/examples/results/<timestamp>/`.

The official page references two files which it does not publish:
`mask.png` in example 4 and `myfile.ovf` in example 5. The harness supplies
compatible fixtures already used by this repository's regression suite:
`test/testdata/mask.png` and `test/testdata/randommag4x4x1.ovf`.
