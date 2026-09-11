# Input Data Formats

SPIDER reads CSV inputs via Polars. Column order does not matter; extra columns are handled as
noted below.

## Event catalog (`io.catalog_infile`)

Required columns:

- `evid` (integer-castable; must match `evid1`/`evid2` in the differential times)
- `longitude`, `latitude` (degrees)
- `depth` (km)
- `time` (ISO-8601 string, e.g. `2000-12-13T15:01:01.000000`; parsed with Polars `str.strptime`)

Additional columns (e.g. `mag`) are preserved and written back out in `<catalog_outfile>_MAP.csv`.
If `model.filters.events.lat_bounds` / `lon_bounds` are set, events outside those bounds are
dropped and differential times referencing them are removed; the counts are logged as
`Initial origins n=... / Initial dtimes n=... / After spatial filtering dtimes n=...`.

## Station table (`io.station_file`)

Required columns: `network`, `station`, `longitude`, `latitude`, `depth`.

`depth` (km, negative above sea level) must be present as a column; null/NaN values are filled
with `0.0`. Duplicate `(network, station)` rows are de-duplicated (one kept). Differential times
whose `(network, station)` is absent from this table are dropped (inner join, no warning).

## Differential times (`io.dtime_file`)

Required columns: `dt`, `network`, `station`, `evid1`, `evid2`, `phase`, `cc`.

All seven are required — `cc` is not optional; the loader selects exactly this set and drops any
other columns.

- `dt`: observed differential time in seconds for the pair (`evid1`, `evid2`)
- `phase`: `P`/`S` (case-insensitive; any string other than `S` maps to P) or numeric `0`/`1`
- `cc`: correlation/quality value used by `model.filters.dtimes.cc_min`
- `evid1`, `evid2`: integer-castable and present in the event catalog

The same pair may appear on multiple rows (e.g. different stations or phases). Rows that
duplicate a (pair, network, station, phase) combination are collapsed when
`model.filters.dtimes.remove_duplicates` is `true`.

`spider synth` writes a compatible file (`<dtime_file basename>_synthetic.csv`) with this header.

See {doc}`configuration-reference` for the full filter list and execution order.
