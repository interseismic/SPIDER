# Input Data Formats

SPIDER reads CSV inputs via Polars.

## Event catalog (`io.catalog_infile`)

Required columns:

- `evid`
- `longitude`
- `latitude`
- `depth`
- `time` (parseable datetime)

## Station table (`io.station_file`)

Required columns:

- `network`
- `station`
- `longitude`
- `latitude`

Optional:

- `depth` (missing values are treated as `0.0`)

## Differential times (`io.dtime_file`)

Required columns:

- `network`
- `station`
- `evid1`
- `evid2`
- `dt`
- `phase` (`P`/`S` or `0`/`1`; normalized internally)

Optional:

- `cc` (cross-correlation score)
