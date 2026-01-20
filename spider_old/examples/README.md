## Examples (tracked in the code-only repo)

Because this repo is **code-only**, examples that should stay in sync with the
runtime are kept under `spider/examples/`.

- `params_template.json`: minimal nested-config template with placeholder paths.

Notes:
- Replace `PATH/TO/...` placeholders before running.
- For single-device commands (`locate-map`, `sample`, `locate-full`) either set
  `inference.compute.devices` to a single entry or pass `--device`.

