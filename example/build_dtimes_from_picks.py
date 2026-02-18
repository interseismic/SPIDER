#!/usr/bin/env python3
"""
Build SPIDER-style differential times (dtimes) from absolute picks.

Given:
- a catalog CSV with per-event hypocenters + origin times (use your best estimates or truth)
- a picks CSV with absolute pick UTC times for (event, station, phase)

We create dt for every station+phase for all event pairs within a 3D hypocentral distance threshold:
  dt = (t_pick2 - t0_2) - (t_pick1 - t0_1)

Optionally, we can add Laplace noise *after differencing*:
  dt_noisy = dt + eps,   eps ~ Laplace(0, b),   where MAE(|eps|) = b

Optionally, we can add Gaussian noise *after differencing*:
  dt_noisy = dt + eps,   eps ~ Normal(0, sigma)

Output columns match SPIDER's dtimes CSV expectations:
  network,station,dt,evid1,evid2,phase,cc
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

try:
    from pyproj import CRS, Transformer
except Exception as e:
    raise RuntimeError("This script requires pyproj (SPIDER already uses it). Install: pip install pyproj") from e

try:
    from scipy.spatial import cKDTree  # optional but recommended
except Exception:
    cKDTree = None


@dataclass(frozen=True)
class EventInfo:
    evid: int
    lat: float
    lon: float
    depth_km: float
    t0_utc: np.datetime64
    xyz_m: np.ndarray  # shape (3,)


def _build_local_transformers(lats: np.ndarray, lons: np.ndarray) -> Tuple[Transformer, Transformer]:
    # Local azimuthal equidistant projection centered on dataset mean
    lat0 = float(np.mean(lats))
    lon0 = float(np.mean(lons))
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs")
    wgs84 = CRS.from_epsg(4326)
    fwd = Transformer.from_crs(wgs84, aeqd, always_xy=True)
    inv = Transformer.from_crs(aeqd, wgs84, always_xy=True)
    return fwd, inv


def _build_local_transformer(lats: np.ndarray, lons: np.ndarray) -> Transformer:
    return _build_local_transformers(lats, lons)[0]


def read_catalog_df(catalog_csv: str) -> pl.DataFrame:
    cat = pl.read_csv(catalog_csv)
    required = {"evid", "latitude", "longitude", "depth", "time"}
    missing = required - set(cat.columns)
    if missing:
        raise ValueError(f"Catalog missing columns: {sorted(missing)}")

    cat = cat.with_columns(
        [
            pl.col("evid").cast(pl.Int64),
            pl.col("latitude").cast(pl.Float64),
            pl.col("longitude").cast(pl.Float64),
            pl.col("depth").cast(pl.Float64),
            # Treat as UTC; your inputs are naive but represent UTC timestamps.
            pl.col("time").cast(pl.Utf8).str.strptime(pl.Datetime, strict=False),
        ]
    )

    return cat


def catalog_df_to_events(cat: pl.DataFrame) -> Dict[int, EventInfo]:
    transformer = _build_local_transformer(cat["latitude"].to_numpy(), cat["longitude"].to_numpy())

    events: Dict[int, EventInfo] = {}
    for r in cat.iter_rows(named=True):
        lon = float(r["longitude"])
        lat = float(r["latitude"])
        dep_km = float(r["depth"])
        evid = int(r["evid"])
        # Store as datetime64[ns]
        t0 = np.datetime64(r["time"]).astype("datetime64[ns]")
        x_m, y_m = transformer.transform(lon, lat)
        z_m = dep_km * 1000.0
        events[evid] = EventInfo(
            evid=evid,
            lat=lat,
            lon=lon,
            depth_km=dep_km,
            t0_utc=t0,
            xyz_m=np.array([x_m, y_m, z_m], dtype=np.float64),
        )
    return events


def apply_event_noise(
    cat: pl.DataFrame,
    *,
    sigma_x_km: float = 0.0,
    sigma_y_km: float = 0.0,
    sigma_z_km: float = 0.0,
    sigma_t_s: float = 0.0,
    seed: int = 0,
) -> pl.DataFrame:
    sigma_x_km = float(sigma_x_km)
    sigma_y_km = float(sigma_y_km)
    sigma_z_km = float(sigma_z_km)
    sigma_t_s = float(sigma_t_s)
    if any(s < 0.0 for s in (sigma_x_km, sigma_y_km, sigma_z_km, sigma_t_s)):
        raise ValueError("Event noise sigmas must be >= 0.")
    if sigma_x_km == 0.0 and sigma_y_km == 0.0 and sigma_z_km == 0.0 and sigma_t_s == 0.0:
        return cat

    lats = cat["latitude"].to_numpy()
    lons = cat["longitude"].to_numpy()
    dep_km = cat["depth"].to_numpy()
    t0 = cat["time"].to_numpy().astype("datetime64[ns]")

    fwd, inv = _build_local_transformers(lats, lons)
    x_m, y_m = fwd.transform(lons, lats)
    z_m = dep_km * 1000.0

    rng = np.random.default_rng(int(seed))
    if sigma_x_km > 0.0:
        x_m = x_m + rng.normal(scale=sigma_x_km * 1000.0, size=x_m.shape)
    if sigma_y_km > 0.0:
        y_m = y_m + rng.normal(scale=sigma_y_km * 1000.0, size=y_m.shape)
    if sigma_z_km > 0.0:
        z_m = z_m + rng.normal(scale=sigma_z_km * 1000.0, size=z_m.shape)
    if sigma_t_s > 0.0:
        t_noise_ns = rng.normal(scale=sigma_t_s, size=t0.shape) * 1e9
        t_noise_ns = np.rint(t_noise_ns).astype(np.int64)
        t0 = t0 + t_noise_ns.astype("timedelta64[ns]")

    lon_noisy, lat_noisy = inv.transform(x_m, y_m)
    dep_noisy = z_m / 1000.0

    return cat.with_columns(
        [
            pl.Series("latitude", lat_noisy),
            pl.Series("longitude", lon_noisy),
            pl.Series("depth", dep_noisy),
            pl.Series("time", t0),
        ]
    )


def read_catalog(catalog_csv: str) -> Dict[int, EventInfo]:
    cat = read_catalog_df(catalog_csv)
    return catalog_df_to_events(cat)


def read_picks(picks_csv: str) -> pl.DataFrame:
    pk = pl.read_csv(picks_csv)
    required = {"time", "evid", "phase", "network", "station"}
    missing = required - set(pk.columns)
    if missing:
        raise ValueError(f"Picks missing columns: {sorted(missing)}")

    pk = pk.with_columns(
        [
            pl.col("evid").cast(pl.Int64),
            pl.col("time").cast(pl.Utf8).str.strptime(pl.Datetime, strict=False),
            pl.col("phase").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
            pl.col("network").cast(pl.Utf8).str.strip_chars(),
            pl.col("station").cast(pl.Utf8).str.strip_chars(),
        ]
    )
    pk = pk.filter(pl.col("phase").is_in(["P", "S"]))
    return pk


def build_dtimes(
    events: Dict[int, EventInfo],
    picks: pl.DataFrame,
    max_hypo_dist_km: float,
    *,
    max_neighbors: Optional[int] = None,
    seed: int = 0,
    laplace_mae_s: float = 0.0,
    gaussian_std_s: float = 0.0,
) -> pl.DataFrame:
    max_dist_m = float(max_hypo_dist_km) * 1000.0
    # Keep neighbor subsampling stable regardless of whether noise is enabled.
    rng_pairs = np.random.default_rng(int(seed))
    rng_noise = np.random.default_rng(int(seed) + 1)
    laplace_mae_s = float(laplace_mae_s)
    gaussian_std_s = float(gaussian_std_s)
    if laplace_mae_s > 0.0 and gaussian_std_s > 0.0:
        raise ValueError("Choose only one noise model: set either laplace_mae_s>0 or gaussian_std_s>0 (not both).")
    event_keys = pl.Series("evid", list(events.keys()), dtype=pl.Int64)

    networks: List[str] = []
    stations: List[str] = []
    dts: List[float] = []
    evid1s: List[int] = []
    evid2s: List[int] = []
    phases: List[str] = []
    ccs: List[float] = []

    # group by station-phase so each dt uses same receiver+phase
    for (net, sta, ph), g in picks.group_by(["network", "station", "phase"], maintain_order=True):
        # keep only events that exist in catalog
        g = g.filter(pl.col("evid").is_in(event_keys))
        if g.height < 2:
            continue

        # For each event, keep just ONE pick time (earliest) per station-phase
        # (If you have multiple arid picks per event-station-phase, this avoids duplicates.)
        g = g.sort("time").unique(subset=["evid"], keep="first")

        eids = g["evid"].to_numpy().astype(np.int64, copy=False)
        tpick = g["time"].to_numpy().astype("datetime64[ns]", copy=False)
        # vectorize origin times and xyz
        xyz = np.stack([events[int(e)].xyz_m for e in eids], axis=0)  # (n,3)
        t0 = np.array([events[int(e)].t0_utc for e in eids], dtype="datetime64[ns]")  # datetime64[ns]

        # travel times in seconds relative to origin
        tt_s = (tpick - t0).astype("timedelta64[ns]").astype(np.int64) * 1e-9

        n = int(eids.size)
        if n < 2:
            continue

        if cKDTree is not None:
            tree = cKDTree(xyz)
            for i in range(n):
                nbr = tree.query_ball_point(xyz[i], r=max_dist_m)
                jj = [j for j in nbr if j > i]
                if max_neighbors is not None and max_neighbors > 0 and len(jj) > max_neighbors:
                    jj = rng_pairs.choice(np.asarray(jj, dtype=np.int64), size=int(max_neighbors), replace=False).tolist()
                for j in jj:
                    dt = float(tt_s[j] - tt_s[i])
                    if laplace_mae_s > 0.0:
                        dt += float(rng_noise.laplace(loc=0.0, scale=laplace_mae_s))
                    elif gaussian_std_s > 0.0:
                        dt += float(rng_noise.normal(loc=0.0, scale=gaussian_std_s))
                    networks.append(str(net))
                    stations.append(str(sta))
                    dts.append(dt)
                    evid1s.append(int(eids[i]))
                    evid2s.append(int(eids[j]))
                    phases.append(str(ph))
                    ccs.append(1.0)
        else:
            # Brute force (O(n^2) per station-phase). OK for small n.
            for i in range(n):
                d = xyz[i + 1 :] - xyz[i]
                dist = np.sqrt(np.sum(d * d, axis=1))
                js = np.nonzero(dist <= max_dist_m)[0] + (i + 1)
                jj = js.tolist()
                if max_neighbors is not None and max_neighbors > 0 and len(jj) > max_neighbors:
                    jj = rng_pairs.choice(np.asarray(jj, dtype=np.int64), size=int(max_neighbors), replace=False).tolist()
                for j in jj:
                    dt = float(tt_s[j] - tt_s[i])
                    if laplace_mae_s > 0.0:
                        dt += float(rng_noise.laplace(loc=0.0, scale=laplace_mae_s))
                    elif gaussian_std_s > 0.0:
                        dt += float(rng_noise.normal(loc=0.0, scale=gaussian_std_s))
                    networks.append(str(net))
                    stations.append(str(sta))
                    dts.append(dt)
                    evid1s.append(int(eids[i]))
                    evid2s.append(int(eids[j]))
                    phases.append(str(ph))
                    ccs.append(1.0)

    out = pl.DataFrame(
        {
            "network": networks,
            "station": stations,
            "dt": dts,
            "evid1": evid1s,
            "evid2": evid2s,
            "phase": phases,
            "cc": ccs,
        }
    )
    return out


def write_growclust_dtcc(dtimes: pl.DataFrame, out_path: str) -> None:
    if dtimes.height == 0:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("")
        return

    dtimes = dtimes.sort(["evid1", "evid2", "station", "phase"])
    with open(out_path, "w", encoding="utf-8") as f:
        for (evid1, evid2), g in dtimes.group_by(["evid1", "evid2"], maintain_order=True):
            f.write(f"# {int(evid1)} {int(evid2)} 0\n")
            for r in g.iter_rows(named=True):
                sta = str(r["station"])
                dt = float(r["dt"])
                cc = float(r["cc"])
                phase = str(r["phase"])
                f.write(f"{sta:>5s} {dt:.6f} {cc:.3f} {phase}\n")


def write_growclust_event_list(cat: pl.DataFrame, out_path: str) -> None:
    mag_col = None
    for name in ("mag", "magnitude", "MAG", "Mag"):
        if name in cat.columns:
            mag_col = name
            break

    with open(out_path, "w", encoding="utf-8") as f:
        for r in cat.iter_rows(named=True):
            t0 = r["time"]
            if t0 is None:
                raise ValueError("Catalog time column contains null values.")
            year = int(t0.year)
            month = int(t0.month)
            day = int(t0.day)
            hour = int(t0.hour)
            minute = int(t0.minute)
            sec = float(t0.second) + float(t0.microsecond) / 1_000_000.0

            evid = int(r["evid"])
            lat = float(r["latitude"])
            lon = float(r["longitude"])
            dep_km = float(r["depth"])
            mag = float(r[mag_col]) if mag_col is not None and r[mag_col] is not None else 0.0

            # Fill relocation fields with catalog values; other fields default to 0.
            line = (
                f"{year:04d} {month:02d} {day:02d} {hour:02d} {minute:02d} {sec:09.6f} "
                f"{lat:.6f} {lon:.6f} {dep_km:.3f} {mag:.3f} "
                "0 0 0 "
                f"{evid:d}\n"
            )
            f.write(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True, help="CSV with evid,latitude,longitude,depth,time (UTC)")
    ap.add_argument("--picks", required=True, help="CSV with time,evid,phase,network,station (UTC)")
    ap.add_argument("--max_dist_km", type=float, default=3.0, help="Max 3D hypocentral distance (km) for pairing")
    ap.add_argument(
        "--max_neighbors",
        type=int,
        default=0,
        help="Optional cap: max neighbors per event within radius (0 disables cap).",
    )
    ap.add_argument("--seed", type=int, default=0, help="Random seed (used only if --max_neighbors > 0).")
    ap.add_argument(
        "--laplace_mae_s",
        type=float,
        default=0.0,
        help="If >0, add Laplace(0, b) noise to dt (after differencing), where MAE(|noise|)=b seconds.",
    )
    ap.add_argument(
        "--gaussian_std_s",
        type=float,
        default=0.0,
        help="If >0, add Normal(0, sigma) noise to dt (after differencing), where STD(noise)=sigma seconds.",
    )
    ap.add_argument("--sigma_x_km", type=float, default=0.0, help="Gaussian noise sigma for event X (km).")
    ap.add_argument("--sigma_y_km", type=float, default=0.0, help="Gaussian noise sigma for event Y (km).")
    ap.add_argument("--sigma_z_km", type=float, default=0.0, help="Gaussian noise sigma for event Z/depth (km).")
    ap.add_argument("--sigma_t_s", type=float, default=0.0, help="Gaussian noise sigma for event origin time (s).")
    ap.add_argument("--out", help="Output dtimes CSV path (SPIDER format)")
    ap.add_argument("--out_dtcc", help="Output GrowClust dt.cc path")
    ap.add_argument("--out_evlist", help="Output GrowClust event list path")
    ap.add_argument("--out_catalog", help="Output perturbed catalog CSV path (SPIDER format)")
    args = ap.parse_args()

    if not args.out and not args.out_dtcc and not args.out_evlist and not args.out_catalog:
        ap.error("Must provide at least one output: --out, --out_dtcc, --out_evlist, or --out_catalog.")

    cat_df = read_catalog_df(args.catalog)
    cat_df = apply_event_noise(
        cat_df,
        sigma_x_km=float(args.sigma_x_km),
        sigma_y_km=float(args.sigma_y_km),
        sigma_z_km=float(args.sigma_z_km),
        sigma_t_s=float(args.sigma_t_s),
        seed=int(args.seed) + 2,
    )
    events = catalog_df_to_events(cat_df)
    picks = read_picks(args.picks)
    max_neighbors = int(args.max_neighbors)
    dtimes = build_dtimes(
        events,
        picks,
        max_hypo_dist_km=float(args.max_dist_km),
        max_neighbors=(max_neighbors if max_neighbors > 0 else None),
        seed=int(args.seed),
        laplace_mae_s=float(args.laplace_mae_s),
        gaussian_std_s=float(args.gaussian_std_s),
    )

    extra = ""
    if float(args.laplace_mae_s) > 0.0:
        extra = f" (with Laplace noise MAE={float(args.laplace_mae_s):g}s)"
    elif float(args.gaussian_std_s) > 0.0:
        extra = f" (with Gaussian noise STD={float(args.gaussian_std_s):g}s)"

    if args.out:
        dtimes.write_csv(args.out)
        print(f"Wrote {dtimes.height:,} dtimes rows to {args.out}{extra}")
    if args.out_dtcc:
        write_growclust_dtcc(dtimes, args.out_dtcc)
        print(f"Wrote {dtimes.height:,} dtimes rows to {args.out_dtcc} (GrowClust dt.cc){extra}")
    if args.out_evlist:
        write_growclust_event_list(cat_df, args.out_evlist)
        print(f"Wrote {cat_df.height:,} events to {args.out_evlist} (GrowClust event list)")
    if args.out_catalog:
        cat_df.write_csv(args.out_catalog)
        print(f"Wrote {cat_df.height:,} events to {args.out_catalog} (SPIDER catalog)")


if __name__ == "__main__":
    main()