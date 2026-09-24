# Input Data Formats

SPIDER reads CSV inputs via Polars. Column order does not matter; extra columns are handled as
noted below. All three files are named in the `io` block (`catalog_infile`, `station_file`,
`dtime_file`); relative paths are resolved against the current working directory.

## Event catalog (`io.catalog_infile`)

Required columns:

- `evid` (integer-castable; must match `evid1`/`evid2` in the differential times)
- `longitude`, `latitude` (degrees)
- `depth` (km, positive down)
- `time` (ISO-8601 string, e.g. `2000-12-13T15:01:01.000000`; parsed with Polars `str.strptime`)

Additional columns (e.g. `mag`) are preserved and written back out in `<catalog_outfile>_MAP.csv`.
Duplicate `evid` rows are reduced to the first occurrence (logged as
`Deduplicated origins by evid`).

If `model.filters.events.lat_bounds` / `lon_bounds` (`[min, max]` degrees) are set, events outside
those bounds are dropped (`Catalog bounds filter (lat=..., lon=...): kept n/n0 events`) and
differential times referencing them are removed. The counts are logged as
`Initial origins n=...`, `Initial dtimes n=...`, `After spatial filtering dtimes n=...`.

## Station table (`io.station_file`)

Required columns: `network`, `station`, `longitude`, `latitude`, `depth`.

`depth` (km, negative above sea level) must be present as a column; null/NaN values are filled
with `0.0`. Duplicate `(network, station)` rows are de-duplicated (one kept). Differential times
whose `(network, station)` is absent from this table are dropped (inner join, no warning), so a
sharp drop between `Initial dtimes n=` and later counts often means station names do not match.

## Differential times (`io.dtime_file`)

Required columns: `dt`, `network`, `station`, `evid1`, `evid2`, `phase`, `cc`.

All seven are required — `cc` is not optional; the loader selects exactly this set and drops any
other columns.

- `dt`: observed differential time in seconds for the pair (`evid1`, `evid2`).
  **Sign convention: `dt = t(evid2) − t(evid1)`** (SPIDER predicts `T(evid2) − T(evid1)`).
  HypoDD-style `dt.cc` files carry `t1 − t2`; set `model.filters.dtimes.flip_dt_sign: true` for
  those. A quick check: with the starting catalog, straight-ray `(d2 − d1)/v` (station distances
  `d1`, `d2`) should correlate positively with `dt`. With the wrong sign the MAP drives events to
  the `inference.safety.max_abs_dX` clamp (`dr_max` stuck at the clamp value).
- `phase`: `P`/`S` (case-insensitive; any string other than `S` maps to P) or numeric `0`/`1`
- `cc`: correlation/quality value used by `model.filters.dtimes.cc_min` (rows with
  `cc < cc_min` are dropped; the filter is off when `cc_min` is 0). Fill with `1.0` if you have no
  quality measure.
- `evid1`, `evid2`: integer-castable and present in the event catalog

The same pair may appear on multiple rows (e.g. different stations or phases). Rows that
duplicate an unordered (pair, network, station, phase) combination are collapsed to one when
`model.filters.dtimes.remove_duplicates` is `true`.

`spider synth` writes a compatible file with this header next to `io.dtime_file`, replacing the
extension with `_synthetic.csv` (`dtimes.csv` -> `dtimes_synthetic.csv`).

## Data-prep filter order

`prepare_input_dfs` applies the `model.filters` settings in this order (each step logs its effect
under `[FILTER]`; the defaults in parentheses disable the step):

1. `dtimes.flip_dt_sign` (default `false`): applied to the freshly loaded table, before any filter
   that looks at the sign of `dt` (logged as `Flipping sign of dt (flip_dt_sign=true)`).
2. Catalog bounds (`events.lat_bounds` / `lon_bounds`, default none); dtimes restricted to events
   present in the catalog.
3. Inner join with the station table.
4. `events.linearization_error` with `phase: "before"` (default phase `after_phase1`).
5. `residual` with `phase: "before"` (default phase `after_phase1`).
6. `dtimes.remove_duplicates` (default `false`).
7. `dtimes.max_abs_input_dt` (default 99999 s): keep `|dt| <` threshold.
8. `dtimes.dtime_thin_frac` (default 1.0): random subsample of rows, fixed seed.
9. `dtimes.cc_min` (default 0.0).
10. `events.min_unique_phase_per_event` (default 1): drop events observed at fewer distinct
    (station, phase) pairs.
11. `events.min_dtimes_per_pair` (default 1): drop event pairs with fewer rows.
12. `events.max_pair_station_ratio` with `ratio_filter_phase: "before"` (default phase `before`,
    default ratio 1.0): drop rows whose inter-event separation divided by the distance from the
    pair centroid to the station exceeds the threshold.
13. Rows with null or non-finite `dt` or station coordinates are dropped.
14. `events.min_dtimes` (default 1): second per-pair minimum, applied after the row filters above.
15. `events.min_event_degree` (default 0): iteratively drop events connected to fewer other events.
16. `events.min_events_per_cluster` (default 0): drop connected components with fewer events.
17. Catalog de-duplicated by `evid` and restricted to events that still have differential times;
    `Final dtimes n=...` / `Final origins n=...` are logged.

Two consequences worth knowing:

- `flip_dt_sign` is step 1, so the `phase: "before"` residual and linearization filters (steps
  4 and 5) always see the corrected sign. (Earlier versions applied the flip after those filters,
  which made them evaluate residuals of the wrong sign when `flip_dt_sign: true`.)
- The `after_phase1` filters (residual, linearization error, and `max_pair_station_ratio` with
  `ratio_filter_phase: "after_phase1"`) run at the MAP locations at the end of Phase 1; events
  left with no rows, or in components smaller than `min_events_per_cluster`, are then removed
  before the second MAP pass and the Phase-2 bundle (see {doc}`outputs-and-formats`).

See {doc}`configuration-reference` for the full key list.
