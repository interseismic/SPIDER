import argparse
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import polars as pl
from pyproj import Proj


def _project_events_to_xyz(origins: pl.DataFrame, lat0: float, lon0: float) -> np.ndarray:
    """
    Project event lon/lat to local equal-area XY (km) and append depth (km).
    Returns an (n_events, 3) array [X, Y, Z].
    """
    projector = Proj(proj="laea", lat_0=lat0, lon_0=lon0, datum="WGS84", units="km")
    XX, YY = projector(origins["longitude"].to_numpy(), origins["latitude"].to_numpy())
    ZZ = origins["depth"].to_numpy()
    return np.column_stack([np.asarray(XX, dtype=float), np.asarray(YY, dtype=float), np.asarray(ZZ, dtype=float)])


def _build_event_neighbor_pairs(
    event_xyz: np.ndarray,
    k_neighbors: int = 64,
    max_dist_km: float = 50.0,
) -> List[Tuple[int, int]]:
    """
    Build unique unordered neighbor event index pairs using a simple k-NN within a radius.
    Note: O(N * (N + k)) worst-case; suitable for small-to-medium catalogs.
    """
    n = int(event_xyz.shape[0])
    pairs: set[Tuple[int, int]] = set()
    for i in range(n):
        # squared distances to all events
        d2 = np.sum((event_xyz - event_xyz[i]) ** 2, axis=1)
        # get top k+1 (self included), then filter radius and drop self
        k = min(k_neighbors + 1, n)
        idx_k = np.argpartition(d2, k - 1)[:k]
        idx_k = idx_k[idx_k != i]
        if max_dist_km is not None and max_dist_km > 0.0:
            idx_k = idx_k[np.sqrt(d2[idx_k]) <= float(max_dist_km)]
        for j in idx_k:
            a, b = (i, int(j)) if i < j else (int(j), i)
            if a != b:
                pairs.add((a, b))
    return sorted(pairs)


def _compute_relative_pick_seconds(picks: pl.DataFrame, origins: pl.DataFrame) -> pl.DataFrame:
    """
    Given picks with absolute times and an origins catalog with origin time,
    compute per-pick travel time in seconds relative to the event origin time.
    Returns the picks DataFrame with an added 't_rel' Float64 seconds column.
    """
    # Map evid -> origin time
    evid_to_t0: Dict[str, np.datetime64] = {row["evid"]: row["time"] for row in origins.iter_rows(named=True)}
    # Build aligned origin times column for picks
    t0_series = pl.Series("t0", [evid_to_t0.get(e, None) for e in picks["evid"].to_list()])
    picks = picks.with_columns(t0_series)
    # Compute seconds: (time - t0) in seconds using int ns conversion
    picks = picks.with_columns(
        ((pl.col("time").cast(pl.Int64) - pl.col("t0").cast(pl.Int64)) / pl.lit(1_000_000_000)).alias("t_rel")
    )
    return picks


def _build_evid_pick_map(picks: pl.DataFrame) -> Dict[str, Dict[Tuple[str, str, str], float]]:
    """
    Build mapping:
      evid -> {(network, station, phase): t_rel_seconds}
    If duplicates exist for an (evid, net, sta, phase), keep the earliest t_rel.
    """
    # Ensure phase is upper-case string
    picks = picks.with_columns(pl.col("phase").cast(pl.Utf8).str.to_uppercase())
    # Group and take minimal t_rel per key
    g = (
        picks
        .group_by(["evid", "network", "station", "phase"])
        .agg(pl.col("t_rel").min().alias("t_rel"))
    )
    evid_map: Dict[str, Dict[Tuple[str, str, str], float]] = {}
    for row in g.iter_rows(named=True):
        eid = row["evid"]
        key = (row["network"], row["station"], row["phase"])
        evid_map.setdefault(eid, {})[key] = float(row["t_rel"])
    return evid_map


def generate_dtimes_from_picks(
    params_path: str,
    picks_path: Optional[str] = None,
    out_path: Optional[str] = None,
    max_dist_km: float = 50.0,
    k_neighbors: int = 64,
    require_pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> pl.DataFrame:
    """
    Generate differential times (dt) CSV for SPIDER from a catalog and absolute pick times.
    The dt for a station/phase and event pair (e1,e2) is (t_rel_e2 - t_rel_e1), where
    t_rel = pick_time - origin_time for that event.

    Inputs (via params JSON and/or explicit args):
      - catalog_infile: CSV with columns ['evid','time','longitude','latitude','depth', ...]
      - picks_file/phase_file: CSV with columns ['time','evid','phase','network','station', ...]
      - station_file: only needed by SPIDER later; not required here

    Output schema (pl.DataFrame and written CSV):
      ['dt','network','station','evid1','evid2','phase','cc']
    """
    # Load params
    params = pl.read_json(params_path)
    params_dict = {k: params[k][0] for k in params.columns}

    catalog_infile = str(params_dict.get("catalog_infile", ""))
    if not catalog_infile:
        raise ValueError("params must include 'catalog_infile'")
    picks_path = picks_path or str(params_dict.get("picks_file", params_dict.get("phase_file", "")))
    if not picks_path:
        raise ValueError("Provide 'picks_path' or include 'picks_file'/'phase_file' in params")
    out_path = out_path or str(params_dict.get("dtime_file", "dtimes.csv"))

    lat0 = float(params_dict.get("lat_min"))
    lon0 = float(params_dict.get("lon_min"))

    # Read catalog and parse time
    origins = pl.read_csv(catalog_infile).with_columns(pl.col("time").str.strptime(pl.Datetime))
    # Project to local XY(Z) for neighbor selection
    event_xyz = _project_events_to_xyz(origins, lat0=lat0, lon0=lon0)
    # Build neighbor pairs by index, then map to evid strings
    pairs_idx = _build_event_neighbor_pairs(event_xyz, k_neighbors=k_neighbors, max_dist_km=max_dist_km)
    evids = origins["evid"].to_list()
    idx_to_evid = {i: str(e) for i, e in enumerate(evids)}
    pairs_evid = [(idx_to_evid[i], idx_to_evid[j]) for (i, j) in pairs_idx]
    # Optional: restrict to explicit pairs
    if require_pairs:
        allowed = set((str(a), str(b)) if str(a) < str(b) else (str(b), str(a)) for a, b in require_pairs)
        pairs_evid = [(a, b) for (a, b) in pairs_evid if ((a, b) in allowed)]

    # Read picks and compute relative seconds to origin
    picks = pl.read_csv(picks_path).with_columns(pl.col("time").str.strptime(pl.Datetime))
    picks = _compute_relative_pick_seconds(picks, origins)
    evid_pick_map = _build_evid_pick_map(picks)

    # Build differential times rows
    out_rows: List[Tuple[float, str, str, str, str, str, float]] = []
    for e1, e2 in pairs_evid:
        m1 = evid_pick_map.get(str(e1))
        m2 = evid_pick_map.get(str(e2))
        if not m1 or not m2:
            continue
        # Intersection of station/phase keys
        common_keys = set(m1.keys()).intersection(m2.keys())
        if not common_keys:
            continue
        for (net, sta, ph) in common_keys:
            dt = float(m2[(net, sta, ph)] - m1[(net, sta, ph)])
            out_rows.append((dt, net, sta, str(e1), str(e2), ph, 1.0))

    if not out_rows:
        print("Warning: no differential times generated.")
        df = pl.DataFrame(
            schema={"dt": pl.Float64, "network": pl.Utf8, "station": pl.Utf8, "evid1": pl.Utf8, "evid2": pl.Utf8, "phase": pl.Utf8, "cc": pl.Float64}
        )
        return df

    dtimes = pl.DataFrame(
        out_rows,
        schema=["dt", "network", "station", "evid1", "evid2", "phase", "cc"],
    )
    # Write CSV
    dtimes.write_csv(out_path)
    print(f"Wrote {dtimes.shape[0]} differential times to {out_path}")
    return dtimes


def _parse_args(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(description="Generate SPIDER differential times from absolute picks.")
    ap.add_argument("--params", required=True, help="Path to params.json")
    ap.add_argument("--picks", default=None, help="Path to picks CSV (overrides params picks_file/phase_file)")
    ap.add_argument("--out", default=None, help="Output dtimes CSV path (defaults to params['dtime_file'])")
    ap.add_argument("--max-dist", type=float, default=50.0, help="Max neighbor distance in km")
    ap.add_argument("--k-neighbors", type=int, default=64, help="K for approximate neighbor selection per event")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    generate_dtimes_from_picks(
        params_path=args.params,
        picks_path=args.picks,
        out_path=args.out,
        max_dist_km=float(args.max_dist),
        k_neighbors=int(args.k_neighbors),
    )


if __name__ == "__main__":
    main()


