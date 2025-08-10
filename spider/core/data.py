from pyproj import Proj
import polars as pl
import numpy as np

import torch


def prepare_stations(stations, dtimes, lat_min, lon_min):
    """
    Prepare station data by projecting coordinates and joining with differential times.

    Args:
        stations: Station DataFrame with network, station, longitude, latitude, depth
        dtimes: Differential times DataFrame
        lat_min, lon_min: Projection center coordinates

    Returns:
        DataFrame with projected station coordinates joined to differential times
    """
    projector = Proj(proj='laea', lat_0=lat_min, lon_0=lon_min,
                    datum='WGS84', units='km')
    XX, YY = projector(stations["longitude"].to_numpy(), stations["latitude"].to_numpy())

    stations_proj = stations.with_columns([
        pl.lit(XX).alias("X"),
        pl.lit(YY).alias("Y"),
        pl.col("depth").alias("Z")
    ])

    return dtimes.join(stations_proj, on=["network", "station"], how="inner")


class DTData(torch.utils.data.Dataset):
    """Dataset class for differential time data."""

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def spatial_cat_subset(params, origins):
    """
    Filter origins by spatial bounds if specified in parameters.

    Args:
        params: Parameter dictionary
        origins: Origins DataFrame

    Returns:
        Filtered origins DataFrame
    """
    if "event_lat_bounds" in params:
        origins = origins.filter(
            (pl.col("latitude") >= params["event_lat_bounds"][0]) &
            (pl.col("latitude") <= params["event_lat_bounds"][1])
        )
    if "event_lon_bounds" in params:
        origins = origins.filter(
            (pl.col("longitude") >= params["event_lon_bounds"][0]) &
            (pl.col("longitude") <= params["event_lon_bounds"][1])
        )
    return origins


def break_edges_longer_than(dtimes, origins, R_max, lat_min, lon_min):
    """
    Remove differential time pairs with events separated by more than R_max.

    Args:
        dtimes: Differential times DataFrame
        origins: Origins DataFrame
        R_max: Maximum allowed distance between events
        lat_min, lon_min: Projection center coordinates

    Returns:
        Filtered differential times DataFrame
    """
    projector = Proj(proj='laea', lat_0=lat_min, lon_0=lon_min,
                    datum='WGS84', units='km')
    XX, YY = projector(origins["longitude"].to_numpy(), origins["latitude"].to_numpy())

    # Create evid_to_row mapping
    evid_to_row = {}
    for i, row in enumerate(origins.iter_rows(named=True)):
        evid_to_row[row['evid']] = i

    filter_list = set()

    # Group by evid1 and evid2
    for name, group in dtimes.group_by(["evid1", "evid2"]):
        e1 = evid_to_row[name[0]]
        e2 = evid_to_row[name[1]]
        R = np.sqrt((XX[e1] - XX[e2])**2 + (YY[e1] - YY[e2])**2 +
                   (origins["depth"][e1] - origins["depth"][e2])**2)
        if R >= R_max:
            filter_list.add((name[0], name[1]))

    # Filter out the long edges
    return dtimes.group_by(["evid1", "evid2"]).filter(lambda x: x.name not in filter_list)


def filter_min_dtimes_per_pair(params, dtimes, origins):
    """
    Filter to keep only event pairs with minimum number of differential times.

    Args:
        params: Parameter dictionary
        dtimes: Differential times DataFrame
        origins: Origins DataFrame

    Returns:
        Filtered dtimes and origins DataFrames
    """
    # Count number of entries per (evid1, evid2) pair
    pair_counts = dtimes.group_by(["evid1", "evid2"]).len()
    valid_pairs = pair_counts.filter(pl.col("len") >= params["min_dtimes_per_pair"])

    # Convert to DataFrame for fast filtering
    valid_pairs_df = valid_pairs.select(["evid1", "evid2"])

    # Join to keep only valid pairs
    dtimes = dtimes.join(valid_pairs_df, on=["evid1", "evid2"], how="inner")

    # Build set of involved evids from filtered dtimes
    evids = np.union1d(dtimes["evid1"].to_numpy(), dtimes["evid2"].to_numpy())

    # Filter origins and dtimes based on these evids
    origins = origins.filter(pl.col("evid").is_in(evids))
    dtimes = dtimes.filter(
        pl.col("evid1").is_in(evids) & pl.col("evid2").is_in(evids)
    )

    return origins, dtimes


def filter_min_rows_per_unordered_pair(df: pl.DataFrame, N: int) -> pl.DataFrame:
    """
    Filter to keep only unordered event pairs with minimum number of observations.

    Args:
        df: DataFrame with evid1 and evid2 columns
        N: Minimum number of observations per pair

    Returns:
        Filtered DataFrame
    """
    df = df.with_columns([
        pl.min_horizontal("evid1", "evid2").alias("evid_min"),
        pl.max_horizontal("evid1", "evid2").alias("evid_max"),
    ])

    while True:
        pair_counts = (
            df.group_by(["evid_min", "evid_max"])
              .len()
              .filter(pl.col("len") >= N)
              .select(["evid_min", "evid_max"])
        )

        df_new = df.join(pair_counts, on=["evid_min", "evid_max"], how="inner")

        if df_new.shape[0] == df.shape[0]:
            break
        df = df_new

    return df.drop(["evid_min", "evid_max"])


def prepare_input_dfs(params):
    """
    Prepare input dataframes for SPIDER processing.

    Args:
        params: Parameter dictionary

    Returns:
        Tuple of (stations, dtimes, origins) DataFrames
    """
    # Read station data and drop duplicates immediately
    stations = pl.read_csv(params["station_file"])
    stations = stations.unique(subset=["network", "station"])

    # Read and process origins
    origins = pl.read_csv(params["catalog_infile"])
    print(f"Initial origin set ({origins.shape[0]})")
    origins = origins.with_columns(pl.col("time").str.strptime(pl.Datetime))
    origins = spatial_cat_subset(params, origins)

    # Read differential times efficiently
    dtimes = pl.read_csv(params["dtime_file"])
    print(f"Initial dtime set ({dtimes.shape[0]})")
    dtimes = dtimes.with_columns(
        (pl.col("phase").str.to_uppercase() == "S").cast(pl.Int8).alias("phase")
    )
    dtimes = dtimes.select(["network", "station", "dt", "evid1", "evid2", "phase", "cc"])

    # Filter to events present in origins
    event_ids = set(origins["evid"].to_numpy())
    dtimes = dtimes.filter(
        pl.col("evid1").is_in(event_ids) & pl.col("evid2").is_in(event_ids)
    )

    print(f"{dtimes.shape[0]} differential times before after spatial filtering")

    # Remove duplicates if requested
    if params["remove_duplicates"]:
        print("Removing duplicates")
        dtimes = dtimes.with_columns([
            pl.min_horizontal("evid1", "evid2").alias("evid_min"),
            pl.max_horizontal("evid1", "evid2").alias("evid_max")
        ])
        dtimes = dtimes.unique(subset=["evid_min", "evid_max", "network", "station", "phase"])
        dtimes = dtimes.drop(["evid_min", "evid_max"])

    print(f"{dtimes.shape[0]} dtimes before filtering on max_abs_input_dt")

    # Filter by maximum absolute differential time
    if params["max_abs_input_dt"] > 0.0:
        dtimes = dtimes.filter(pl.col("dt").abs() < params["max_abs_input_dt"])

    # Sample if thinning is needed
    thin_frac = params["dtime_thin_frac"]
    if thin_frac < 1.0:
        print(f"Thinning dtimes with fraction {thin_frac}")
        dtimes = dtimes.sample(fraction=thin_frac, seed=42)

    # Optional: Flip sign
    if params["flip_dt_sign"]:
        dtimes = dtimes.with_columns(pl.col("dt") * -1)

    # Filter by cross-correlation threshold
    if params["cc_min"] > 0.0:
        print(f"Filtering dtimes with cc_min={params['cc_min']}")
        dtimes = dtimes.filter(pl.col("cc") >= params["cc_min"])

    # Filter by minimum differential times per pair
    if params["min_dtimes_per_pair"] > 0:
        print(f"Filtering dtimes with min_dtimes_per_pair={params['min_dtimes_per_pair']}")
        origins, dtimes = filter_min_dtimes_per_pair(params, dtimes, origins)

    # Prepare station coordinates
    dtimes = prepare_stations(stations, dtimes, params["lat_min"], params["lon_min"])

    # Filter by minimum observations per unordered pair
    if params["min_dtimes"] > 0:
        print(f"Filtering dtimes with min_dtimes={params['min_dtimes']}")
        dtimes = filter_min_rows_per_unordered_pair(dtimes, params["min_dtimes"])

    # Update origins to include only events with differential times
    event_ids = np.concatenate([dtimes["evid1"].to_numpy(), dtimes["evid2"].to_numpy()])
    unique_events = set(np.unique(event_ids))
    origins = origins.filter(pl.col("evid").is_in(unique_events))

    print(f"Final dtime set ({dtimes.shape[0]})")
    print(f"Final origin set ({origins.shape[0]})")

    return stations, dtimes, origins
