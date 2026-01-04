from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import polars as pl
from pyproj import Proj


@dataclass(frozen=True)
class SynthCatalogResult:
    truth: pl.DataFrame
    init: pl.DataFrame
    dX: np.ndarray  # (n_events, 4) in [km,km,km,sec]


def _require_cols(df: pl.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _project_to_xyz_km(origins: pl.DataFrame, *, lat0: float, lon0: float) -> Tuple[np.ndarray, Proj]:
    projector = Proj(proj="laea", lat_0=lat0, lon_0=lon0, datum="WGS84", units="km")
    xx, yy = projector(origins["longitude"].to_numpy(), origins["latitude"].to_numpy())
    zz = origins["depth"].to_numpy()
    xyz = np.column_stack([np.asarray(xx, dtype=np.float64), np.asarray(yy, dtype=np.float64), np.asarray(zz, dtype=np.float64)])
    return xyz, projector


def _inverse_project_xy_km(projector: Proj, xx_km: np.ndarray, yy_km: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lon, lat = projector(xx_km, yy_km, inverse=True)
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def synth_initial_catalog_from_truth(
    truth_catalog: pl.DataFrame,
    *,
    lat0: float,
    lon0: float,
    event_prior_std: Sequence[float],
    seed: int = 42,
    spatial_only: bool = False,
    clip_domain: bool = True,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
) -> SynthCatalogResult:
    """
    Given a 'truth' catalog, generate a synthetic *initial* catalog by perturbing
    each event with noise drawn from the event prior (independent Normal per dim).

    Units:
      - X,Y,Z are in km (LAEA projection; depth is assumed km in catalog)
      - delta_t is in seconds (applied to 'time')
      - event_prior_std = [σx_km, σy_km, σz_km, σt_sec]

    The returned DataFrames preserve all input columns, but overwrite:
      - longitude, latitude, depth, time
    """
    _require_cols(truth_catalog, ["evid", "longitude", "latitude", "depth", "time"], "truth_catalog")

    std = np.asarray([float(x) for x in event_prior_std], dtype=np.float64)
    if std.shape != (4,):
        raise ValueError(f"event_prior_std must have length 4 [σx,σy,σz,σt]; got shape {std.shape}")

    # Ensure datetime parsing (force ns so our time shifting math is unambiguous)
    truth = truth_catalog.with_columns(
        pl.col("time").cast(pl.Utf8).str.strptime(pl.Datetime("ns"), strict=False).alias("time")
    )
    if truth["time"].null_count() > 0:
        raise ValueError("truth_catalog.time could not be parsed as datetime for some rows")

    xyz_km, projector = _project_to_xyz_km(truth, lat0=lat0, lon0=lon0)  # (N,3)
    n = int(xyz_km.shape[0])

    rng = np.random.default_rng(int(seed))
    dX = rng.normal(loc=0.0, scale=std[None, :], size=(n, 4)).astype(np.float64)
    if spatial_only:
        dX[:, 3] = 0.0

    xyz_init = xyz_km + dX[:, :3]
    if clip_domain and (z_min is not None) and (z_max is not None):
        xyz_init[:, 2] = np.clip(xyz_init[:, 2], float(z_min), float(z_max))

    lon_init, lat_init = _inverse_project_xy_km(projector, xyz_init[:, 0], xyz_init[:, 1])
    depth_init = xyz_init[:, 2].astype(np.float64)

    # Shift event origin time by dT seconds
    time_ns = truth["time"].cast(pl.Int64).to_numpy()  # ns since epoch
    dt_ns = np.round(dX[:, 3] * 1e9).astype(np.int64)
    time_init_ns = time_ns + dt_ns

    init = truth.with_columns(
        [
            pl.Series("longitude", lon_init.astype(np.float64)),
            pl.Series("latitude", lat_init.astype(np.float64)),
            pl.Series("depth", depth_init.astype(np.float64)),
            pl.Series("time", time_init_ns).cast(pl.Datetime("ns")),
        ]
    )

    return SynthCatalogResult(truth=truth, init=init, dX=dX.astype(np.float64))


