## Examples

`spider/examples/params_template.json` is a minimal nested‑config template.

### How to use

1) Copy the template:

```bash
cp spider/examples/params_template.json my_params.json
```

2) Edit required paths:

- `io.dtime_file`
- `io.station_file`
- `io.catalog_infile`
- `io.catalog_outfile`
- `io.samples_outfile`
- `io.checkpoint_dir`
- `model.model_file`

3) Set spatial bounds under `model.domain`.

4) Run:

```bash
python -m spider locate-map my_params.json --device 0
python -m spider sample my_params.json --device 0
```

### Notes

- For single‑device commands (`locate-map`, `sample`, `locate-full`) either set
  `inference.compute.devices` to one entry or pass `--device`.
- For multi‑GPU independent chains, use `sample-multi` with `--devices`.

