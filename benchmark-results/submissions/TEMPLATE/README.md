# Apple Silicon benchmark submission

Rename this directory to `<chip>-<mac-model>-YYYYMMDD`, fill in
`metadata.json`, and run:

```bash
./run-benchmark.sh
```

The script refuses to overwrite an earlier run. It produces five console logs,
five output directories, `statistics.json`, a sanitized hardware profile, the
macOS version, power settings, and the tested binary's SHA-256 digest.

Before opening a pull request, inspect every generated text file and remove any
personal information that macOS may have added. Do not remove benchmark output
or ordinary model identifiers. Then add this measurement to
`../../registry.json`, regenerate `../../apple-silicon-vs-cuda.svg`, and follow
the [submission instructions](../README.md).
