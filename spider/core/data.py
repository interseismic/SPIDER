from pyproj import Proj
import polars as pl
import numpy as np

import torch

from spider.utils.console import info, warn

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

    # Ensure numeric station coordinates and depths; fill missing depth with 0.0
    stations_proj = stations.with_columns([
        pl.Series(name="X", values=XX).alias("X"),
        pl.Series(name="Y", values=YY).alias("Y"),
        pl.col("depth").fill_null(0.0).fill_nan(0.0).alias("Z"),
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


def filter_min_unique_phase_per_event(params, dtimes, origins):
    """
    Filter to keep only events that have at least a minimum number of unique
    (station, phase) observations (across all pairs the event participates in).

    Args:
        params: Parameter dictionary
        dtimes: Differential times DataFrame
        origins: Origins DataFrame

    Returns:
        Filtered dtimes and origins DataFrames
    """
    min_unique = int(params.get("min_unique_phase_per_event", 0))
    if min_unique <= 0:
        return origins, dtimes
    # Build (evid, network, station, phase) occurrences from both evid1 and evid2
    df1 = dtimes.select(["network", "station", "phase", "evid1"]).rename({"evid1": "evid"})
    df2 = dtimes.select(["network", "station", "phase", "evid2"]).rename({"evid2": "evid"})
    sp = pl.concat([df1, df2])
    # Unique station-phase per event
    spu = sp.unique(subset=["evid", "network", "station", "phase"])
    counts = spu.group_by("evid").len().rename({"len": "n_unique_sp"})
    valid = counts.filter(pl.col("n_unique_sp") >= min_unique).select(["evid"])
    valid_evids = set(valid["evid"].to_numpy())
    # Filter origins to valid events
    origins = origins.filter(pl.col("evid").is_in(valid_evids))
    # Filter dtimes to rows where both events are valid
    dtimes = dtimes.filter(
        pl.col("evid1").is_in(valid_evids) & pl.col("evid2").is_in(valid_evids)
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


def filter_min_event_degree(params, dtimes, origins):
    """
    Filter to keep only events that are connected to at least K unique *other* events.
    Iteratively prunes the graph until all remaining events satisfy the degree constraint.

    Args:
        params: Parameter dictionary containing 'min_event_degree'
        dtimes: Differential times DataFrame
        origins: Origins DataFrame

    Returns:
        Filtered origins and dtimes DataFrames
    """
    k_min = int(params.get("min_event_degree", 0))
    if k_min <= 0:
        return origins, dtimes

    info(f"Filtering events min_event_degree={k_min} (iterative)", section="FILTER")
    
    # Work with a simplified graph dataframe: (evid1, evid2) pairs
    # We only care about unique edges between events
    pairs = dtimes.select(["evid1", "evid2"]).unique()
    
    initial_n_events = origins.shape[0]
    
    while True:
        # 1. Compute degree for each event
        # Concat both columns to get all mentions of each event
        all_mentions = pl.concat([
            pairs.select(pl.col("evid1").alias("evid")),
            pairs.select(pl.col("evid2").alias("evid"))
        ])
        
        # Count unique partners (degree)
        # Since 'pairs' is already unique(evid1, evid2), the count of an evid in 'all_mentions'
        # IS its degree (number of edges connected to it).
        degrees = all_mentions.group_by("evid").len().rename({"len": "degree"})
        
        # 2. Identify events to keep
        keep_evids = degrees.filter(pl.col("degree") >= k_min).select("evid")
        
        # Check convergence
        current_n_pairs = pairs.shape[0]
        
        # 3. Filter pairs to only those where BOTH events are in keep_evids
        pairs = pairs.join(keep_evids, left_on="evid1", right_on="evid", how="inner") \
                     .join(keep_evids, left_on="evid2", right_on="evid", how="inner")
        
        if pairs.shape[0] == current_n_pairs:
            # No pairs removed -> stable
            break
            
    # Final set of valid events
    final_evids = set(pl.concat([
        pairs.select(pl.col("evid1").alias("evid")),
        pairs.select(pl.col("evid2").alias("evid"))
    ]).unique()["evid"].to_list())
    
    final_n_events = len(final_evids)
    info(f"Event degree filter events_in={initial_n_events} events_out={final_n_events}", section="FILTER")
    
    # Apply to dtimes and origins
    # Filter origins
    origins = origins.filter(pl.col("evid").is_in(final_evids))
    # Filter dtimes
    dtimes = dtimes.filter(
        pl.col("evid1").is_in(final_evids) & pl.col("evid2").is_in(final_evids)
    )
    
    return origins, dtimes


def filter_by_pair_station_ratio(
    params: dict,
    dtimes: pl.DataFrame,
    origins: pl.DataFrame,
    *,
    lat_min: float,
    lon_min: float,
) -> pl.DataFrame:
    """
    Filter rows by the ratio R_ev / D_es, where:
      - R_ev is the separation between the two events in km (3D: X,Y,Z)
      - D_es is the distance from the event-pair centroid to the receiver in km (3D: X,Y,Z)
    The filter keeps rows with ratio <= params['max_pair_station_ratio'] (if provided).
    """
    ratio_thr = float(params.get("max_pair_station_ratio", -1.0))
    if not (ratio_thr > 0.0):
        return dtimes

    # Project event lon/lat to local XY (km)
    projector = Proj(proj="laea", lat_0=lat_min, lon_0=lon_min, datum="WGS84", units="km")
    XX, YY = projector(origins["longitude"].to_numpy(), origins["latitude"].to_numpy())
    ZZ = origins["depth"].to_numpy()
    ev_xy = pl.DataFrame(
        {
            "evid": origins["evid"],
            "X_ev": pl.Series(values=XX, dtype=pl.Float64),
            "Y_ev": pl.Series(values=YY, dtype=pl.Float64),
            "Z_ev": pl.Series(values=ZZ, dtype=pl.Float64),
        }
    )

    # Join event coordinates for evid1 and evid2
    ev1 = ev_xy.rename({"evid": "evid1", "X_ev": "X1", "Y_ev": "Y1", "Z_ev": "Z1"})
    ev2 = ev_xy.rename({"evid": "evid2", "X_ev": "X2", "Y_ev": "Y2", "Z_ev": "Z2"})
    dt = (
        dtimes
        .join(ev1, on="evid1", how="left")
        .join(ev2, on="evid2", how="left")
    )

    # Compute pair separation R_ev and centroid-station distance D_es (3D)
    dt = dt.with_columns([
        ((pl.col("X1") - pl.col("X2")) ** 2 + (pl.col("Y1") - pl.col("Y2")) ** 2 + (pl.col("Z1") - pl.col("Z2")) ** 2).sqrt().alias("R_ev"),
        (((pl.col("X1") + pl.col("X2")) * 0.5 - pl.col("X")) ** 2 +
         ((pl.col("Y1") + pl.col("Y2")) * 0.5 - pl.col("Y")) ** 2 +
         ((pl.col("Z1") + pl.col("Z2")) * 0.5 - pl.col("Z")) ** 2).sqrt().alias("D_es"),
    ])
    # Safe ratio (avoid divide-by-zero)
    dt = dt.with_columns([
        pl.when(pl.col("D_es") < 1e-6).then(1e-6).otherwise(pl.col("D_es")).alias("D_es_safe")
    ])
    dt = dt.with_columns(
        (pl.col("R_ev") / pl.col("D_es_safe")).alias("pair_station_ratio")
    )

    # Filter and drop temporaries
    before = dt.shape[0]
    dt = dt.filter(pl.col("pair_station_ratio") <= ratio_thr)
    after = dt.shape[0]
    info(f"Filtering dtimes max_pair_station_ratio={ratio_thr} kept={after}/{before}", section="FILTER")

    dt = dt.drop(["X1", "Y1", "Z1", "X2", "Y2", "Z2", "R_ev", "D_es", "D_es_safe", "pair_station_ratio"])
    return dt


def prepare_input_dfs(params, *, model=None, device=None):
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
    # Fill missing station depths with 0.0 to avoid NaNs in receiver Z
    if "depth" in stations.columns:
        stations = stations.with_columns(
            pl.col("depth").fill_null(0.0).fill_nan(0.0)
        )

    # Read and process origins
    origins = pl.read_csv(params["catalog_infile"])
    info(f"Initial origins n={origins.shape[0]}", section="DATA")
    origins = origins.with_columns(pl.col("time").str.strptime(pl.Datetime))
    origins = spatial_cat_subset(params, origins)

    # Read differential times efficiently
    dtimes = pl.read_csv(params["dtime_file"])
    info(f"Initial dtimes n={dtimes.shape[0]}", section="DATA")
    # Normalize phase column to Int8: 0 for P, 1 for S
    try:
        ph_dtype = dtimes.schema.get("phase", None)
    except Exception:
        ph_dtype = None
    if ph_dtype is None:
        raise ValueError("Input dtimes must contain a 'phase' column")
    # If phase is string/categorical, map "S"->1, others->0 (after uppercase)
    if ph_dtype in (pl.Utf8, pl.Categorical, pl.Enum):
        dtimes = dtimes.with_columns(
            (pl.col("phase").cast(pl.Utf8).str.to_uppercase() == "S").cast(pl.Int8).alias("phase")
        )
    else:
        # Assume already numeric 0/1; just cast to Int8
        dtimes = dtimes.with_columns(pl.col("phase").cast(pl.Int8).alias("phase"))
    dtimes = dtimes.select(["network", "station", "dt", "evid1", "evid2", "phase", "cc"])

    # Filter to events present in origins
    event_ids = set(origins["evid"].to_numpy())
    dtimes = dtimes.filter(
        pl.col("evid1").is_in(event_ids) & pl.col("evid2").is_in(event_ids)
    )

    info(f"After spatial filtering dtimes n={dtimes.shape[0]}", section="DATA")

    # Attach station coordinates early so the linearization error filter (if requested) can run
    # before the rest of the dtimes/event filters.
    dtimes = prepare_stations(stations, dtimes, params["lat_min"], params["lon_min"])

    # Optional: residual-based outlier removal (Observed - Predicted) at initial locations (ΔX=0).
    # This should run as early as possible (right after station coords are attached), and
    # specifically *after* the linearization filter when that filter is enabled for phase='before',
    # so both filters reuse the same model/device and the same early tensorization logic.
    def _maybe_residual_outlier_filter_dtimes() -> None:
        if not bool(params.get("residual_filter_enable", False)):
            return
        # Only run once per pipeline invocation (helps if prepare_input_dfs is called multiple times
        # with the same params dict in interactive workflows).
        if bool(params.get("_residual_filter_applied_in_prepare_input_dfs", False)):
            return
        if model is None or device is None:
            raise ValueError(
                "filters.events.residual_filter.enabled=true requires passing model and device "
                "to prepare_input_dfs(params, model=..., device=...)."
            )
        try:
            import torch
            from spider.core.modeling import compute_residuals_full, med_abs_dev_torch
        except Exception as e:
            raise RuntimeError(f"Could not import torch/modeling needed for residual filter: {e}") from e

        method = str(params.get("residual_filter_method", "mad")).lower()
        sigma = float(params.get("residual_filter_mad_sigma", 6.0))
        abs_max = float(params.get("residual_filter_abs_max", 1.0))
        bs = max(int(params.get("batch_size_warmup", 1)), 1)

        # Build the same tensorization artifacts as the linearization filter uses (but from the
        # *current* dtimes/origins, which may already have been filtered).
        nonlocal dtimes
        evid_to_row = {int(row["evid"]): idx for idx, row in enumerate(origins.iter_rows(named=True))}
        from pyproj import Proj
        projector = Proj(
            proj="laea",
            lat_0=params["lat_min"],
            lon_0=params["lon_min"],
            datum="WGS84",
            units="km",
        )
        XX, YY0 = projector(origins["longitude"].to_numpy(), origins["latitude"].to_numpy())
        X_src = torch.zeros((origins.shape[0], 4), dtype=torch.float32, device=device)
        X_src[:, 0] = torch.tensor(XX, dtype=torch.float32, device=device)
        X_src[:, 1] = torch.tensor(YY0, dtype=torch.float32, device=device)
        X_src[:, 2] = torch.tensor(origins["depth"].to_numpy(), dtype=torch.float32, device=device)
        dX_src = torch.zeros_like(X_src, device=device)
        e1_idx = np.array([evid_to_row[int(x)] for x in dtimes["evid1"]], dtype=np.int64)
        e2_idx = np.array([evid_to_row[int(x)] for x in dtimes["evid2"]], dtype=np.int64)
        YY_np = dtimes[["dt", "X", "Y", "Z", "phase"]].to_numpy()

        N = int(dtimes.shape[0])
        if N <= 0:
            params["_residual_filter_applied_in_prepare_input_dfs"] = True
            return

        # Build tensors and compute residuals in batches.
        II = torch.tensor(np.column_stack([e1_idx, e2_idx]), dtype=torch.int64, device=device)
        YY = torch.tensor(YY_np, dtype=torch.float32, device=device)
        zero_dX = torch.zeros_like(dX_src, device=device)
        residuals = compute_residuals_full(II, YY, X_src, zero_dX, model, bs, N)

        abs_thr = abs_max if (abs_max is not None and float(abs_max) > 0.0) else float("inf")
        if method == "abs":
            thr = abs_thr
            mask_keep = torch.abs(residuals) <= thr
        else:
            # Robust MAD threshold about the residual median, but also respect abs_max:
            # use thr = min(abs_max, mad_sigma * MAD) when abs_max > 0.
            r_med = torch.median(residuals)
            r_mad = med_abs_dev_torch(residuals)
            if not torch.isfinite(r_mad) or float(r_mad.item()) <= 0.0:
                thr = abs_thr
                info(
                    f"Residual filter: MAD not useful (mad={float(r_mad.item()):.3e}); using abs_max={thr:.3f}s",
                    section="FILTER",
                )
                mask_keep = torch.abs(residuals - r_med) <= thr
            else:
                mad_thr = sigma * float(r_mad.item())
                thr = min(abs_thr, mad_thr)
                mask_keep = torch.abs(residuals - r_med) <= thr

        keep_count = int(mask_keep.sum().item())
        drop_count = int(N - keep_count)
        if drop_count > 0:
            mask_cpu = mask_keep.detach().cpu().numpy()
            if method == "abs":
                thr_msg = f"thr={thr:.3f}s (abs_max)"
            else:
                thr_msg = f"thr={thr:.3f}s (min(abs_max={abs_thr:.3f}s, mad_sigma*mad={mad_thr:.3f}s))"
            info(
                f"Residual filter: dropping {drop_count}/{N} dtimes ({thr_msg}, method={method}).",
                section="FILTER",
            )
            dtimes = dtimes.filter(pl.Series(mask_cpu))
            info(f"Residual filter: remaining dtimes n={dtimes.shape[0]}", section="FILTER")
        else:
            info("Residual filter: no outliers detected; keeping all rows.", section="FILTER")

        params["_residual_filter_applied_in_prepare_input_dfs"] = True

    # Optional: apply linearization error filter at the *beginning* of the dtimes filter pipeline (when enabled).
    # This requires the model and device (torch) because it evaluates the forward model.
    if bool(params.get("linearization_error_enable", False)) and str(params.get("linearization_error_phase", "")).strip().lower() == "before":
        if model is None or device is None:
            raise ValueError(
                "filters.events.linearization_error.enabled=true with phase='before' requires passing model and device "
                "to prepare_input_dfs(params, model=..., device=...)."
            )
        try:
            import torch
            from spider.core.modeling import compute_linearization_error_ratio
        except Exception as e:
            raise RuntimeError(f"Could not import torch/modeling needed for linearization filter: {e}") from e

        lin_bs = int(params.get("linearization_error_batch_size", 50000))
        lin_bs = max(1, lin_bs)
        lin_log_every = int(params.get("linearization_error_log_every_batches", 25))
        lin_max_ratio = params.get("linearization_error_max_ratio", None)
        lin_max_ratio_f = float(lin_max_ratio) if lin_max_ratio is not None else None
        if not (lin_max_ratio_f is not None and lin_max_ratio_f > 0.0):
            raise ValueError(
                "linearization_error.enabled=true with phase='before' but linearization_error.max_ratio is missing or <= 0"
            )

        # Build event tensor X_src (km) and row index tensors for dtimes
        evid_to_row = {int(row["evid"]): idx for idx, row in enumerate(origins.iter_rows(named=True))}
        # Project events into local XY (km)
        from pyproj import Proj
        projector = Proj(proj="laea", lat_0=params["lat_min"], lon_0=params["lon_min"], datum="WGS84", units="km")
        XX, YY0 = projector(origins["longitude"].to_numpy(), origins["latitude"].to_numpy())
        X_src = torch.zeros((origins.shape[0], 4), dtype=torch.float32, device=device)
        X_src[:, 0] = torch.tensor(XX, dtype=torch.float32, device=device)
        X_src[:, 1] = torch.tensor(YY0, dtype=torch.float32, device=device)
        X_src[:, 2] = torch.tensor(origins["depth"].to_numpy(), dtype=torch.float32, device=device)
        dX_src = torch.zeros_like(X_src, device=device)

        # Map dtimes evid -> indices (CPU numpy)
        e1_idx = np.array([evid_to_row[int(x)] for x in dtimes["evid1"]], dtype=np.int64)
        e2_idx = np.array([evid_to_row[int(x)] for x in dtimes["evid2"]], dtype=np.int64)
        YY_np = dtimes[["dt", "X", "Y", "Z", "phase"]].to_numpy()

        N = int(dtimes.shape[0])
        keep_mask = np.zeros((N,), dtype=np.bool_)
        n_batches = (N + lin_bs - 1) // lin_bs
        import time as _time
        t0 = _time.time()
        for b in range(n_batches):
            i0 = b * lin_bs
            i1 = min(i0 + lin_bs, N)
            II_b = torch.tensor(np.column_stack([e1_idx[i0:i1], e2_idx[i0:i1]]), dtype=torch.int64, device=device)
            YY_b = torch.tensor(YY_np[i0:i1, :], dtype=torch.float32, device=device)
            _, ratio_b, _, _ = compute_linearization_error_ratio(
                idx=II_b,
                y=YY_b,
                X_src=X_src,
                ΔX_src=dX_src,
                model=model,
            )
            ratio_cpu = ratio_b.detach().float().cpu().numpy()
            keep_mask[i0:i1] = (ratio_cpu <= lin_max_ratio_f)
            if lin_log_every > 0 and ((b + 1) % lin_log_every == 0 or (b + 1) == n_batches):
                dt_s = _time.time() - t0
                rate = float((i1)) / max(dt_s, 1e-6)
                info(
                    f"Linearization filter (before): {b+1}/{n_batches} rows={i1}/{N} rows/s={rate:.3g}",
                    section="FILTER",
                )

        before_n = int(dtimes.shape[0])
        keep_idx = np.nonzero(keep_mask)[0].astype(np.int64, copy=False)
        after_n = int(keep_idx.size)
        info(
            f"Applied linearization_error (before) max_ratio={lin_max_ratio_f}; kept {after_n}/{before_n} dtimes.",
            section="FILTER",
        )
        if hasattr(dtimes, "take"):
            try:
                dtimes = dtimes.take(keep_idx.tolist())  # type: ignore[attr-defined]
            except Exception:
                dtimes = dtimes.take(pl.Series(keep_idx))  # type: ignore[attr-defined]
        elif hasattr(dtimes, "gather"):
            dtimes = dtimes.gather(keep_idx.tolist())  # type: ignore[attr-defined]
        else:
            tmp = dtimes.with_row_index("__row") if hasattr(dtimes, "with_row_index") else dtimes.with_row_count("__row")
            dtimes = tmp.filter(pl.col("__row").is_in(pl.Series(keep_idx))).drop("__row")

        # Immediately after the linearization filter (before), apply residual-based outlier filtering if enabled.
        try:
            _maybe_residual_outlier_filter_dtimes()
        except Exception as e:
            warn(f"Residual filter (after linearization 'before') failed: {e}", section="FILTER")

    else:
        # If no linearization filter ran here, still run residual outlier filtering as early as possible.
        try:
            _maybe_residual_outlier_filter_dtimes()
        except Exception as e:
            warn(f"Residual filter (early) failed: {e}", section="FILTER")

    # Remove duplicates if requested
    if bool(params.get("remove_duplicates", False)):
        info("Removing duplicate (pair,station,phase) rows", section="FILTER")
        dtimes = dtimes.with_columns([
            pl.min_horizontal("evid1", "evid2").alias("evid_min"),
            pl.max_horizontal("evid1", "evid2").alias("evid_max")
        ])
        dtimes = dtimes.unique(subset=["evid_min", "evid_max", "network", "station", "phase"])
        dtimes = dtimes.drop(["evid_min", "evid_max"])

    info(f"Before max_abs_input_dt filter dtimes n={dtimes.shape[0]}", section="FILTER")

    # Filter by maximum absolute differential time
    max_abs_input_dt = float(params.get("max_abs_input_dt", 0.0))
    if max_abs_input_dt > 0.0:
        dtimes = dtimes.filter(pl.col("dt").abs() < max_abs_input_dt)

    # Sample if thinning is needed
    thin_frac = float(params.get("dtime_thin_frac", 1.0))
    if thin_frac < 1.0:
        info(f"Thinning dtimes fraction={thin_frac}", section="FILTER")
        dtimes = dtimes.sample(fraction=thin_frac, seed=42)

    # Optional: Flip sign
    if bool(params.get("flip_dt_sign", False)):
        dtimes = dtimes.with_columns(pl.col("dt") * -1)

    # Filter by cross-correlation threshold
    cc_min = float(params.get("cc_min", 0.0))
    if cc_min > 0.0:
        info(f"Filtering dtimes cc_min={cc_min}", section="FILTER")
        dtimes = dtimes.filter(pl.col("cc") >= cc_min)

    # Filter by minimum number of unique (station, phase) per event
    _min_unique = int(params.get("min_unique_phase_per_event", 0))
    if _min_unique > 0:
        info(f"Filtering events min_unique_phase_per_event={_min_unique}", section="FILTER")
        origins, dtimes = filter_min_unique_phase_per_event(params, dtimes, origins)

    # Filter by minimum observations per unordered pair (true pair filtering)
    _min_pair = int(params.get("min_dtimes_per_pair", 0))
    if _min_pair > 0:
        info(f"Filtering event pairs min_dtimes_per_pair={_min_pair}", section="FILTER")
        dtimes = filter_min_rows_per_unordered_pair(dtimes, _min_pair)

    # Filter by minimum degree (number of connected events)
    # Applied AFTER pair filtering to ensure degree counts only valid edges
    # (Moved to end of function to handle subsequent filters)
    # _min_degree = int(params.get("min_event_degree", 0))
    # if _min_degree > 0:
    #    origins, dtimes = filter_min_event_degree(params, dtimes, origins)

    # Station coordinates already attached above.

    # Optional: filter by pair/centroid-station ratio (R_ev / D_es)
    # Apply here only if configured to run before Phase 1 (default).
    _ratio_phase = str(params.get("ratio_filter_phase", "before")).strip().lower()
    if (
        "max_pair_station_ratio" in params
        and float(params["max_pair_station_ratio"]) > 0.0
        and _ratio_phase == "before"
    ):
        dtimes = filter_by_pair_station_ratio(
            params, dtimes, origins, lat_min=params["lat_min"], lon_min=params["lon_min"]
        )

    # Drop rows with any non-finite receiver coordinates or dt values
    required_cols = ["dt", "X", "Y", "Z", "phase"]
    existing_required = [c for c in required_cols if c in dtimes.columns]
    if existing_required:
        dtimes = (
            dtimes
            .drop_nulls(subset=existing_required)
            .filter(
                pl.col("dt").is_finite() &
                pl.col("X").is_finite() &
                pl.col("Y").is_finite() &
                pl.col("Z").is_finite()
            )
        )

    # Filter by minimum observations per unordered pair
    min_dtimes = int(params.get("min_dtimes", 0))
    if min_dtimes > 0:
        info(f"Filtering dtimes min_dtimes={min_dtimes}", section="FILTER")
        dtimes = filter_min_rows_per_unordered_pair(dtimes, min_dtimes)

    # Filter by minimum degree (number of connected events)
    # Applied LAST to ensure degree counts reflect the final graph structure
    # after all edge filtering steps (ratio, finite checks, min_dtimes, etc).
    _min_degree = int(params.get("min_event_degree", 0))
    if _min_degree > 0:
        origins, dtimes = filter_min_event_degree(params, dtimes, origins)

    # Update origins to include only events with differential times
    event_ids = np.concatenate([dtimes["evid1"].to_numpy(), dtimes["evid2"].to_numpy()])
    unique_events = set(np.unique(event_ids))
    origins = origins.filter(pl.col("evid").is_in(unique_events))

    info(f"Final dtimes n={dtimes.shape[0]}", section="DATA")
    info(f"Final origins n={origins.shape[0]}", section="DATA")

    return stations, dtimes, origins
