from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from pyproj import Proj

from .utils.console import info, warn


@dataclass
class LocateState:
    params: dict
    device: torch.device
    projector: Proj
    origins0: pl.DataFrame
    dtimes: pl.DataFrame
    X_src: torch.Tensor
    dX_src: torch.nn.Parameter
    II: torch.Tensor
    YY: torch.Tensor
    row_station_index: Optional[torch.Tensor]
    n_stations: int
    model: nn.Module
    optimizer: Optional[torch.optim.Optimizer] = None
    sampler: Optional[torch.optim.Optimizer] = None
    N: int = 0
    batch_size_warmup: int = 0
    batch_size_sgld: int = 0
    cluster_ids: Optional[torch.Tensor] = None
    cluster_counts: Optional[torch.Tensor] = None
    clamp_abs_dX: Optional[torch.Tensor] = None


def _parse_clamp_tensor(params: dict, device: torch.device) -> Optional[torch.Tensor]:
    v = params.get("max_abs_dX", None)
    if v is None:
        return None
    try:
        vals = [float(x) for x in v]
        if len(vals) != 4:
            return None
        t = torch.tensor(vals, device=device, dtype=torch.float32)
        return torch.abs(t)
    except Exception:
        return None


def build_state(
    *,
    params: dict,
    origins0: pl.DataFrame,
    dtimes: pl.DataFrame,
    model: nn.Module,
    device: torch.device,
    dX_src_init: Optional[torch.Tensor] = None,
) -> LocateState:
    evid_to_row = {row["evid"]: idx for idx, row in enumerate(origins0.iter_rows(named=True))}
    evid1_idx = [evid_to_row[x] for x in dtimes["evid1"]]
    evid2_idx = [evid_to_row[x] for x in dtimes["evid2"]]
    dtimes = dtimes.with_columns(
        [pl.Series(evid1_idx).alias("evid1_idx"), pl.Series(evid2_idx).alias("evid2_idx")]
    )

    # Station index per row (for shared_event_re grouping)
    row_station_index = None
    n_stations = 0
    try:
        sta_keys = (
            dtimes.select([pl.col("network"), pl.col("station")])
            .unique(maintain_order=True)
            .with_row_index("sta_idx")
        )
        dtimes = dtimes.join(sta_keys, on=["network", "station"], how="left")
        if dtimes["sta_idx"].null_count() > 0:
            dtimes = dtimes.with_columns(pl.col("sta_idx").fill_null(-1))
        sta_idx_np = dtimes["sta_idx"].to_numpy().astype(np.int64, copy=False)
        row_station_index = torch.tensor(sta_idx_np, dtype=torch.int64, device=device).contiguous()
        n_stations = int(sta_keys.shape[0])
    except Exception:
        row_station_index = None
        n_stations = 0

    projector = Proj(proj="laea", lat_0=params["lat_min"], lon_0=params["lon_min"], datum="WGS84", units="km")
    XX, YY = projector(origins0["longitude"].to_numpy(), origins0["latitude"].to_numpy())
    X_src = torch.zeros((origins0.shape[0], 4), dtype=torch.float32, device=device)
    X_src[:, 0] = torch.tensor(XX, dtype=torch.float32, device=device)
    X_src[:, 1] = torch.tensor(YY, dtype=torch.float32, device=device)
    X_src[:, 2] = torch.tensor(origins0["depth"].to_numpy(), dtype=torch.float32, device=device)

    if dX_src_init is None:
        dX_src = torch.zeros_like(X_src)
    else:
        dX_src = dX_src_init.to(device=device, dtype=torch.float32)
    dX_src.requires_grad_(True)
    dX_src = torch.nn.Parameter(dX_src)

    II = torch.tensor(dtimes[["evid1_idx", "evid2_idx"]].to_numpy(), dtype=torch.int64, device=device)
    YY_t = torch.tensor(dtimes[["dt", "X", "Y", "Z", "phase"]].to_numpy(), dtype=torch.float32, device=device)
    N = dtimes.shape[0]
    info(f"Differential times rows N={int(N):,}", section="DATA")

    # Cluster ids for gauge projection (optional)
    cluster_ids = None
    cluster_counts = None
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        n_events = origins0.shape[0]
        row = np.array(evid1_idx)
        col = np.array(evid2_idx)
        data = np.ones(len(row), dtype=int)
        adj = coo_matrix((data, (row, col)), shape=(n_events, n_events))
        n_components, labels = connected_components(csgraph=adj, directed=False, return_labels=True)
        cluster_ids = torch.tensor(labels, dtype=torch.int64, device=device)
        counts = torch.bincount(cluster_ids, minlength=n_components).float().unsqueeze(1)
        cluster_counts = counts
        info(f"Cluster analysis components={n_components}", section="GRAPH")
    except Exception as e:
        warn(f"Cluster analysis unavailable: {e}", section="GRAPH")
        cluster_ids = None
        cluster_counts = None

    clamp_abs_dX = _parse_clamp_tensor(params, device)

    return LocateState(
        params=params,
        device=device,
        projector=projector,
        origins0=origins0,
        dtimes=dtimes,
        X_src=X_src,
        dX_src=dX_src,
        II=II,
        YY=YY_t,
        row_station_index=row_station_index,
        n_stations=n_stations,
        model=model.to(device),
        N=int(N),
        batch_size_warmup=int(params.get("batch_size_warmup", 10000)),
        batch_size_sgld=int(params.get("batch_size_sgld", 10000)),
        cluster_ids=cluster_ids,
        cluster_counts=cluster_counts,
        clamp_abs_dX=clamp_abs_dX,
    )
