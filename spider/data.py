from __future__ import annotations

import math
from typing import Tuple, Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from pyproj import Proj

from .modeling import compute_residuals, compute_linearization_error_ratio
from .utils.console import info, warn


def prepare_stations(stations: pl.DataFrame, dtimes: pl.DataFrame, lat_min: float, lon_min: float) -> pl.DataFrame:
    projector = Proj(proj="laea", lat_0=lat_min, lon_0=lon_min, datum="WGS84", units="km")
    XX, YY = projector(stations["longitude"].to_numpy(), stations["latitude"].to_numpy())
    stations_proj = stations.with_columns(
        [
            pl.Series(name="X", values=XX).alias("X"),
            pl.Series(name="Y", values=YY).alias("Y"),
            pl.col("depth").fill_null(0.0).fill_nan(0.0).alias("Z"),
        ]
    )
    return dtimes.join(stations_proj, on=["network", "station"], how="inner")


def _normalize_phase(dtimes: pl.DataFrame) -> pl.DataFrame:
    try:
        ph_dtype = dtimes.schema.get("phase", None)
    except Exception:
        ph_dtype = None
    if ph_dtype in (pl.Utf8, pl.Categorical, pl.Enum):
        return dtimes.with_columns((pl.col("phase").cast(pl.Utf8).str.to_uppercase() == "S").cast(pl.Int8).alias("phase"))
    return dtimes.with_columns(pl.col("phase").cast(pl.Int8).alias("phase"))


def _filter_min_rows_per_unordered_pair(df: pl.DataFrame, N: int) -> pl.DataFrame:
    if N <= 1:
        return df
    df = df.with_columns(
        [
            pl.min_horizontal("evid1", "evid2").alias("evid_min"),
            pl.max_horizontal("evid1", "evid2").alias("evid_max"),
        ]
    )
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


def _filter_min_unique_phase_per_event(params: dict, dtimes: pl.DataFrame, origins: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    min_unique = int(params.get("min_unique_phase_per_event", 0))
    if min_unique <= 0:
        return origins, dtimes
    df1 = dtimes.select(["network", "station", "phase", "evid1"]).rename({"evid1": "evid"})
    df2 = dtimes.select(["network", "station", "phase", "evid2"]).rename({"evid2": "evid"})
    sp = pl.concat([df1, df2])
    spu = sp.unique(subset=["evid", "network", "station", "phase"])
    counts = spu.group_by("evid").len().rename({"len": "n_unique_sp"})
    valid = counts.filter(pl.col("n_unique_sp") >= min_unique).select(["evid"])
    valid_evids = set(valid["evid"].to_numpy())
    origins = origins.filter(pl.col("evid").is_in(valid_evids))
    dtimes = dtimes.filter(pl.col("evid1").is_in(valid_evids) & pl.col("evid2").is_in(valid_evids))
    return origins, dtimes


def _filter_min_event_degree(params: dict, dtimes: pl.DataFrame, origins: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    k_min = int(params.get("min_event_degree", 0))
    if k_min <= 0:
        return origins, dtimes
    info(f"Filtering events min_event_degree={k_min} (iterative)", section="FILTER")
    pairs = dtimes.select(["evid1", "evid2"]).unique()
    initial_n_events = origins.shape[0]
    while True:
        all_mentions = pl.concat([pairs.select(pl.col("evid1").alias("evid")), pairs.select(pl.col("evid2").alias("evid"))])
        degrees = all_mentions.group_by("evid").len().rename({"len": "degree"})
        keep_evids = degrees.filter(pl.col("degree") >= k_min).select("evid")
        current_n_pairs = pairs.shape[0]
        pairs = (
            pairs.join(keep_evids, left_on="evid1", right_on="evid", how="inner")
            .join(keep_evids, left_on="evid2", right_on="evid", how="inner")
            .select(["evid1", "evid2"])
        )
        if pairs.shape[0] == current_n_pairs:
            break
    final_evids = set(pl.concat([pairs.select(pl.col("evid1").alias("evid")), pairs.select(pl.col("evid2").alias("evid"))]).unique()["evid"].to_list())
    info(f"Event degree filter events_in={initial_n_events} events_out={len(final_evids)}", section="FILTER")
    origins = origins.filter(pl.col("evid").is_in(final_evids))
    dtimes = dtimes.filter(pl.col("evid1").is_in(final_evids) & pl.col("evid2").is_in(final_evids))
    return origins, dtimes


def _filter_by_pair_station_ratio(params: dict, dtimes: pl.DataFrame, origins: pl.DataFrame, *, lat_min: float, lon_min: float) -> pl.DataFrame:
    ratio_thr = float(params.get("max_pair_station_ratio", -1.0))
    if not (ratio_thr > 0.0):
        return dtimes
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
    ev1 = ev_xy.rename({"evid": "evid1", "X_ev": "X1", "Y_ev": "Y1", "Z_ev": "Z1"})
    ev2 = ev_xy.rename({"evid": "evid2", "X_ev": "X2", "Y_ev": "Y2", "Z_ev": "Z2"})
    dt = dtimes.join(ev1, on="evid1", how="left").join(ev2, on="evid2", how="left")
    dt = dt.with_columns(
        [
            ((pl.col("X1") - pl.col("X2")) ** 2 + (pl.col("Y1") - pl.col("Y2")) ** 2 + (pl.col("Z1") - pl.col("Z2")) ** 2).sqrt().alias("R_ev"),
            (((pl.col("X1") + pl.col("X2")) * 0.5 - pl.col("X")) ** 2
             + ((pl.col("Y1") + pl.col("Y2")) * 0.5 - pl.col("Y")) ** 2
             + ((pl.col("Z1") + pl.col("Z2")) * 0.5 - pl.col("Z")) ** 2).sqrt().alias("D_es"),
        ]
    )
    dt = dt.with_columns([pl.when(pl.col("D_es") < 1e-6).then(1e-6).otherwise(pl.col("D_es")).alias("D_es_safe")])
    dt = dt.with_columns((pl.col("R_ev") / pl.col("D_es_safe")).alias("pair_station_ratio"))
    before = dt.shape[0]
    dt = dt.filter(pl.col("pair_station_ratio") <= ratio_thr)
    after = dt.shape[0]
    info(f"Filtering dtimes max_pair_station_ratio={ratio_thr} kept={after}/{before}", section="FILTER")
    return dt.drop(["X1", "Y1", "Z1", "X2", "Y2", "Z2", "R_ev", "D_es", "D_es_safe", "pair_station_ratio"])


def _residual_filter(
    *,
    params: dict,
    dtimes: pl.DataFrame,
    origins: pl.DataFrame,
    model: nn.Module,
    device: torch.device,
    lat_min: float,
    lon_min: float,
) -> pl.DataFrame:
    if not bool(params.get("residual_filter_enable", False)):
        return dtimes
    method = str(params.get("residual_filter_method", "mad")).lower()
    sigma = float(params.get("residual_filter_mad_sigma", 6.0))
    abs_max = float(params.get("residual_filter_abs_max", 1.0))
    bs = max(int(params.get("batch_size_warmup", 1)), 1)

    # Build tensors
    evid_to_row = {row["evid"]: idx for idx, row in enumerate(origins.iter_rows(named=True))}
    e1_idx = [evid_to_row[x] for x in dtimes["evid1"]]
    e2_idx = [evid_to_row[x] for x in dtimes["evid2"]]
    II = torch.tensor(np.column_stack([e1_idx, e2_idx]), dtype=torch.int64, device=device)
    YY = torch.tensor(dtimes[["dt", "X", "Y", "Z", "phase"]].to_numpy(), dtype=torch.float32, device=device)
    projector = Proj(proj="laea", lat_0=lat_min, lon_0=lon_min, datum="WGS84", units="km")
    XX, YY_ev = projector(origins["longitude"].to_numpy(), origins["latitude"].to_numpy())
    X_src = torch.zeros((origins.shape[0], 4), dtype=torch.float32, device=device)
    X_src[:, 0] = torch.tensor(XX, dtype=torch.float32, device=device)
    X_src[:, 1] = torch.tensor(YY_ev, dtype=torch.float32, device=device)
    X_src[:, 2] = torch.tensor(origins["depth"].to_numpy(), dtype=torch.float32, device=device)
    dX_src = torch.zeros_like(X_src)

    residuals = torch.empty((YY.shape[0],), device=device, dtype=torch.float32)
    with torch.no_grad():
        for i0 in range(0, YY.shape[0], bs):
            i1 = min(i0 + bs, YY.shape[0])
            r = compute_residuals(II[i0:i1], YY[i0:i1], X_src, dX_src, model)
            residuals[i0:i1] = r

    res = residuals.detach().cpu().numpy().astype(np.float64)
    mask = np.isfinite(res)
    res = res[mask]
    if res.size == 0:
        warn("Residual filter skipped (no finite residuals)", section="FILTER")
        return dtimes
    med = float(np.median(res))
    mad = float(np.median(np.abs(res - med)))
    if mad <= 0.0:
        mad = max(float(np.std(res)), 1e-6)
    thr = sigma * mad
    keep = np.abs(residuals.detach().cpu().numpy() - med) <= thr
    if method == "mad":
        keep = keep & (np.abs(residuals.detach().cpu().numpy()) <= abs_max)
    before = dtimes.shape[0]
    dtimes = dtimes.filter(pl.Series(keep))
    after = dtimes.shape[0]
    info(f"Residual filter kept={after}/{before} (mad_sigma={sigma}, abs_max={abs_max})", section="FILTER")
    return dtimes


def _linearization_error_filter(
    *,
    params: dict,
    dtimes: pl.DataFrame,
    origins: pl.DataFrame,
    model: nn.Module,
    device: torch.device,
    lat_min: float,
    lon_min: float,
) -> pl.DataFrame:
    if not bool(params.get("linearization_error_enable", False)):
        return dtimes
    if str(params.get("linearization_error_phase", "before")).lower() != "before":
        return dtimes
    max_ratio = float(params.get("linearization_error_max_ratio", 0.0))
    if not (max_ratio > 0.0):
        return dtimes
    bs = int(params.get("linearization_error_batch_size", 50000))
    bs = max(1, bs)
    info(f"Computing linearization ratio (before Phase 1) max_ratio={max_ratio}", section="FILTER")

    evid_to_row = {row["evid"]: idx for idx, row in enumerate(origins.iter_rows(named=True))}
    e1_idx = [evid_to_row[x] for x in dtimes["evid1"]]
    e2_idx = [evid_to_row[x] for x in dtimes["evid2"]]
    II = torch.tensor(np.column_stack([e1_idx, e2_idx]), dtype=torch.int64, device=device)
    YY = torch.tensor(dtimes[["dt", "X", "Y", "Z", "phase"]].to_numpy(), dtype=torch.float32, device=device)
    projector = Proj(proj="laea", lat_0=lat_min, lon_0=lon_min, datum="WGS84", units="km")
    XX, YY_ev = projector(origins["longitude"].to_numpy(), origins["latitude"].to_numpy())
    X_src = torch.zeros((origins.shape[0], 4), dtype=torch.float32, device=device)
    X_src[:, 0] = torch.tensor(XX, dtype=torch.float32, device=device)
    X_src[:, 1] = torch.tensor(YY_ev, dtype=torch.float32, device=device)
    X_src[:, 2] = torch.tensor(origins["depth"].to_numpy(), dtype=torch.float32, device=device)
    dX_src = torch.zeros_like(X_src)

    keep_mask = np.zeros((YY.shape[0],), dtype=np.bool_)
    for i0 in range(0, YY.shape[0], bs):
        i1 = min(i0 + bs, YY.shape[0])
        _, ratio, _, _ = compute_linearization_error_ratio(II[i0:i1], YY[i0:i1], X_src, dX_src, model)
        ratio_cpu = ratio.detach().cpu().numpy()
        keep_mask[i0:i1] = ratio_cpu <= max_ratio
    before = dtimes.shape[0]
    dtimes = dtimes.filter(pl.Series(keep_mask))
    after = dtimes.shape[0]
    info(f"Linearization filter kept={after}/{before}", section="FILTER")
    return dtimes


def prepare_input_dfs(
    params: dict,
    *,
    model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    stats: dict[str, int] = {}
    stations = pl.read_csv(params["station_file"]).unique(subset=["network", "station"])
    if "depth" in stations.columns:
        stations = stations.with_columns(pl.col("depth").fill_null(0.0).fill_nan(0.0))

    origins = pl.read_csv(params["catalog_infile"])
    info(f"Initial origins n={origins.shape[0]}", section="DATA")
    stats["filter/events_initial"] = int(origins.shape[0])
    if "time" in origins.columns:
        origins = origins.with_columns(pl.col("time").cast(pl.Utf8).str.strptime(pl.Datetime("ns"), strict=False))

    dtimes = pl.read_csv(params["dtime_file"])
    info(f"Initial dtimes n={dtimes.shape[0]}", section="DATA")
    stats["filter/dtimes_initial"] = int(dtimes.shape[0])
    dtimes = _normalize_phase(dtimes)
    dtimes = dtimes.select(["network", "station", "dt", "evid1", "evid2", "phase", "cc"])

    if bool(params.get("flip_dt_sign", False)):
        dtimes = dtimes.with_columns((-pl.col("dt")).alias("dt"))

    if bool(params.get("remove_duplicates", False)):
        dtimes = dtimes.unique()
    max_abs_dt = float(params.get("max_abs_input_dt", 0.0))
    if max_abs_dt > 0.0 and math.isfinite(max_abs_dt):
        dtimes = dtimes.filter(pl.col("dt").abs() <= max_abs_dt)
    cc_min = float(params.get("cc_min", -1.0))
    if cc_min > 0.0:
        dtimes = dtimes.filter(pl.col("cc") >= cc_min)

    # event filter: only those present in origins
    event_ids = set(origins["evid"].to_numpy())
    dtimes = dtimes.filter(pl.col("evid1").is_in(event_ids) & pl.col("evid2").is_in(event_ids))

    # Dtime thinning
    thin_frac = float(params.get("dtime_thin_frac", 1.0))
    if thin_frac < 1.0:
        rng = np.random.default_rng(int(params.get("runtime_seed", 0)))
        keep = rng.random(dtimes.shape[0]) < thin_frac
        dtimes = dtimes.filter(pl.Series(keep))

    # Attach station coords early
    dtimes = prepare_stations(stations, dtimes, params["lat_min"], params["lon_min"])

    # Pair-station ratio filter (before)
    if str(params.get("ratio_filter_phase", "before")).lower() == "before":
        dtimes = _filter_by_pair_station_ratio(params, dtimes, origins, lat_min=float(params["lat_min"]), lon_min=float(params["lon_min"]))

    # Filter min_dtimes per event
    min_dtimes = int(params.get("min_dtimes", 0))
    if min_dtimes > 1:
        counts = (
            pl.concat([dtimes.select(pl.col("evid1").alias("evid")), dtimes.select(pl.col("evid2").alias("evid"))])
            .group_by("evid")
            .len()
            .rename({"len": "n"})
        )
        keep_evids = set(counts.filter(pl.col("n") >= min_dtimes)["evid"].to_numpy())
        origins = origins.filter(pl.col("evid").is_in(keep_evids))
        dtimes = dtimes.filter(pl.col("evid1").is_in(keep_evids) & pl.col("evid2").is_in(keep_evids))

    origins, dtimes = _filter_min_unique_phase_per_event(params, dtimes, origins)

    min_pair = int(params.get("min_dtimes_per_pair", 0))
    if min_pair > 1:
        dtimes = _filter_min_rows_per_unordered_pair(dtimes, min_pair)

    origins, dtimes = _filter_min_event_degree(params, dtimes, origins)

    if bool(params.get("linearization_error_enable", False)):
        if model is None or device is None:
            raise ValueError("linearization_error filter requires model and device")
        lin_before = int(dtimes.shape[0])
        dtimes = _linearization_error_filter(
            params=params,
            dtimes=dtimes,
            origins=origins,
            model=model,
            device=device,
            lat_min=float(params["lat_min"]),
            lon_min=float(params["lon_min"]),
        )
        stats["filter/linearization_kept"] = int(dtimes.shape[0])
        stats["filter/linearization_removed"] = int(lin_before - dtimes.shape[0])

    if bool(params.get("residual_filter_enable", False)):
        if model is None or device is None:
            raise ValueError("residual filter requires model and device")
        resid_before = int(dtimes.shape[0])
        dtimes = _residual_filter(
            params=params,
            dtimes=dtimes,
            origins=origins,
            model=model,
            device=device,
            lat_min=float(params["lat_min"]),
            lon_min=float(params["lon_min"]),
        )
        stats["filter/residual_kept"] = int(dtimes.shape[0])
        stats["filter/residual_removed"] = int(resid_before - dtimes.shape[0])

    info(f"Final dtimes n={dtimes.shape[0]}", section="DATA")
    stats["filter/events_final"] = int(origins.shape[0])
    stats["filter/dtimes_final"] = int(dtimes.shape[0])
    params["_filter_stats"] = stats
    return stations, dtimes, origins
