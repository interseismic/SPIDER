from spider.utils.console import info, warn


# Standardized stdout helper
def _log(*parts, section: str = "ANALYSIS", **_kwargs) -> None:
    msg = " ".join(str(p) for p in parts)
    low = msg.strip().lower()
    if low.startswith("warning") or low.startswith("error"):
        warn(msg, section=section)
    else:
        info(msg, section=section)

"""
Diagnose "sticky" events: suspiciously high ESS and unrealistically low uncertainty.

This script computes:
- Per-event degree in the (filtered) dtimes graph used by SPIDER (via prepare_input_dfs)
- Per-event posterior standard deviations from the samples store
- Per-event ESS (min over x/y/z/t) using existing SPIDER utilities

It then joins these into a single table (no pandas required) and prints summaries.

Example:
  python -m spider.analysis.diagnose_sticky_events \\
    --params yifan_redo/SPIDER_yifan.json \\
    --burn-in 0 --thin 1 --top 50
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Any, Optional

import numpy as np
import polars as pl


def _materialize_params(params_path: str) -> Dict[str, Any]:
    from spider.core.config_schema import (
        validate_and_materialize_block1,
        validate_and_materialize_block2,
        validate_and_materialize_block3,
        validate_and_materialize_block4,
        validate_and_materialize_block5,
    )
    from spider.core.priors_config import validate_and_materialize_priors

    with open(params_path, "r") as f:
        p = json.load(f)
    for fn in (
        validate_and_materialize_block1,
        validate_and_materialize_block2,
        validate_and_materialize_block3,
        validate_and_materialize_block4,
        validate_and_materialize_block5,
        validate_and_materialize_priors,
    ):
        p = fn(p)
    return p


def _degree_table_from_dtimes(dtimes: pl.DataFrame) -> pl.DataFrame:
    # event degree in the undirected multigraph: count incident edges
    a = dtimes.select(pl.col("evid1").cast(pl.Utf8).alias("evid"))
    b = dtimes.select(pl.col("evid2").cast(pl.Utf8).alias("evid"))
    deg = (
        pl.concat([a, b])
        .group_by("evid")
        .len()
        .rename({"len": "degree"})
        .sort("degree", descending=True)
    )
    return deg


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    try:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) < 3:
            return float("nan")
        return float(np.corrcoef(x[m], y[m])[0, 1])
    except Exception:
        return float("nan")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="Path to SPIDER params JSON (the same used for the run).")
    ap.add_argument("--device", default="cpu", help="Device for model eval used in data prep (cpu/cuda[:idx]).")
    ap.add_argument("--burn-in", type=int, default=0, help="Burn-in samples to drop per event.")
    ap.add_argument("--thin", type=int, default=1, help="Thin factor when reading the samples store.")
    ap.add_argument("--max-lag", type=int, default=256, help="Max lag for ESS computation.")
    ap.add_argument("--ess-device", default=None, help="Device for ESS computation (default: auto).")
    ap.add_argument("--events-chunk-size", type=int, default=2048, help="Chunk size for summary computations.")
    ap.add_argument("--top", type=int, default=50, help="How many suspicious events to print.")
    ap.add_argument("--out-csv", default=None, help="Optional path to write the joined diagnostics table as CSV.")
    args = ap.parse_args(argv)

    params = _materialize_params(str(args.params))

    # --- Load filtered dtimes graph (this reproduces the filtering pipeline) ---
    from spider.cli import _load_model  # reuse the robust loader
    from spider.core.data import prepare_input_dfs

    model = _load_model(params, args.device)
    stations, dtimes, origins = prepare_input_dfs(params, model=model, device=args.device)
    deg = _degree_table_from_dtimes(dtimes)

    # --- Load samples + compute summary stats / ESS ---
    from spider.io import read_all_samples
    from spider.analysis.results import EventSamplesSummary, compute_ess_summary

    event_samples = read_all_samples(params, backend="numpy", thin=int(args.thin))
    if not event_samples or "event_ids" not in event_samples:
        raise RuntimeError("No samples found (or missing event_ids). Check params['samples_outfile'].")

    # Build centered samples (no pandas needed because we don't request cat_dd)
    summary = EventSamplesSummary.compute(
        event_samples,
        burn_in=int(args.burn_in),
        include=["X", "Y", "Z", "T"],
        events_chunk_size=int(args.events_chunk_size),
        show_progress=True,
    )
    ess = compute_ess_summary(
        summary,
        max_lag=int(args.max_lag),
        device=args.ess_device,
        show_progress=True,
    )

    X = summary.X
    Y = summary.Y
    Z = summary.Z
    T = summary.T
    assert X is not None and Y is not None and Z is not None and T is not None

    std_x = np.std(X, axis=1, ddof=1)
    std_y = np.std(Y, axis=1, ddof=1)
    std_z = np.std(Z, axis=1, ddof=1)
    std_t = np.std(T, axis=1, ddof=1)
    std_xyz = np.sqrt(std_x * std_x + std_y * std_y + std_z * std_z)

    event_ids = np.asarray(event_samples["event_ids"], dtype=str)

    df = pl.DataFrame(
        {
            "evid": event_ids,
            "std_x_km": std_x.astype(np.float32, copy=False),
            "std_y_km": std_y.astype(np.float32, copy=False),
            "std_z_km": std_z.astype(np.float32, copy=False),
            "std_t_s": std_t.astype(np.float32, copy=False),
            "std_xyz_km": std_xyz.astype(np.float32, copy=False),
            "ess_xyzt_min": np.asarray(ess.get("ess_per_event_xyzt_min", ess.get("ess_per_event", [])), dtype=np.float32),
            "ess_x": np.asarray(ess.get("ess_per_event_x", []), dtype=np.float32),
            "ess_y": np.asarray(ess.get("ess_per_event_y", []), dtype=np.float32),
            "ess_z": np.asarray(ess.get("ess_per_event_z", []), dtype=np.float32),
            "ess_t": np.asarray(ess.get("ess_per_event_t", []), dtype=np.float32),
            "n_samples": int(ess.get("n_samples", int(X.shape[1]))),
        }
    )

    df = df.join(deg, on="evid", how="left").with_columns(pl.col("degree").fill_null(0).cast(pl.Int32))

    # Heuristic "suspicious" score: high ESS + tiny std_xyz
    # (rank-based to be scale-free across runs)
    df = df.with_columns(
        [
            (pl.col("ess_xyzt_min").rank("dense", descending=True) / pl.count()).alias("ess_rank_hi"),
            (pl.col("std_xyz_km").rank("dense", descending=False) / pl.count()).alias("std_rank_lo"),
        ]
    ).with_columns(
        (pl.col("ess_rank_hi") * pl.col("std_rank_lo")).alias("suspicious_score")
    )

    # Print correlations (degree vs stickiness proxies)
    try:
        deg_np = df["degree"].to_numpy()
        corr_deg_std = _safe_corr(deg_np, df["std_xyz_km"].to_numpy())
        corr_deg_ess = _safe_corr(deg_np, df["ess_xyzt_min"].to_numpy())
        _log(f"[sticky] corr(degree, std_xyz_km) = {corr_deg_std:.3f}")
        _log(f"[sticky] corr(degree, ess_xyzt_min) = {corr_deg_ess:.3f}")
    except Exception:
        pass

    # Show the most suspicious events
    top = int(max(1, args.top))
    cols = [
        "evid",
        "degree",
        "ess_xyzt_min",
        "std_xyz_km",
        "std_x_km",
        "std_y_km",
        "std_z_km",
        "std_t_s",
    ]
    _log("\n[sticky] Top suspicious events (high ESS + low std):")
    _log(
        df.sort("suspicious_score", descending=True)
        .select(cols)
        .head(top)
    )

    # Also show degree quantiles for the "worst" 10% by suspicious_score
    try:
        q = df.select(pl.col("suspicious_score").quantile(0.9)).item()
        worst = df.filter(pl.col("suspicious_score") >= float(q))
        deg_q = worst.select(
            [
                pl.count().alias("n"),
                pl.col("degree").min().alias("deg_min"),
                pl.col("degree").median().alias("deg_med"),
                pl.col("degree").quantile(0.9).alias("deg_p90"),
                pl.col("degree").max().alias("deg_max"),
            ]
        )
        _log("\n[sticky] Degree stats for worst 10% suspicious_score:")
        _log(deg_q)
    except Exception:
        pass

    if args.out_csv:
        out_path = str(args.out_csv)
        df.sort("suspicious_score", descending=True).write_csv(out_path)
        _log(f"\n[sticky] Wrote diagnostics CSV: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


