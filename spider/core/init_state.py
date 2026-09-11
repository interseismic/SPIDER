from __future__ import annotations

from typing import List, Optional
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from pyproj import Proj
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

from spider.core.state import LocateState, _parse_clamp_tensor
from spider.utils.console import info, warn


def _build_initial_state(
    params: dict,
    origins0: pl.DataFrame,
    dtimes: pl.DataFrame,
    model: nn.Module,
    device: torch.device,
) -> LocateState:
    """Prepare tensors, priors, and optimizer."""

    # Vectorized evid -> origin-row mapping via polars joins. (The previous
    # Python dict + per-element list comprehensions took minutes and gigabytes
    # of transient objects at 1e8 dtime rows.) keep="last" matches the old
    # dict-overwrite semantics if origins0 ever contained duplicate evids.
    idx_map = (
        origins0.select(pl.col("evid"))
        .with_row_index("__origin_row")
        .unique(subset="evid", keep="last", maintain_order=True)
    )
    n_rows0 = int(dtimes.shape[0])
    joined = (
        dtimes.with_row_index("__dt_row")
        .join(idx_map.rename({"evid": "evid1", "__origin_row": "evid1_idx"}), on="evid1", how="left")
        .join(idx_map.rename({"evid": "evid2", "__origin_row": "evid2_idx"}), on="evid2", how="left")
        .sort("__dt_row")
        .drop("__dt_row")
    )
    missing_e1_count = int(joined["evid1_idx"].null_count())
    missing_e2_count = int(joined["evid2_idx"].null_count())
    if missing_e1_count > 0 or missing_e2_count > 0 or int(joined.shape[0]) != n_rows0:
        # Provide a clear consistency error instead of a raw KeyError.
        # This usually means dtimes references events not present in origins0,
        # or evid dtypes differ (e.g., int vs string ids).
        try:
            missing_e1_examples = joined.filter(pl.col("evid1_idx").is_null())["evid1"].head(5).to_list()
            missing_e2_examples = joined.filter(pl.col("evid2_idx").is_null())["evid2"].head(5).to_list()
        except Exception:
            missing_e1_examples = []
            missing_e2_examples = []

        origin_evid_type = "unknown"
        try:
            if origins0.shape[0] > 0:
                origin_evid_type = type(origins0["evid"][0]).__name__
        except Exception:
            origin_evid_type = "unknown"

        dt_e1_type = "unknown"
        dt_e2_type = "unknown"
        try:
            if dtimes.shape[0] > 0:
                dt_e1_type = type(dtimes["evid1"][0]).__name__
                dt_e2_type = type(dtimes["evid2"][0]).__name__
        except Exception:
            pass

        missing_value = (missing_e1_examples + missing_e2_examples)[:1] or ["<unknown>"]
        raise ValueError(
            "Inconsistent event ids between origins0 and dtimes while building initial state. "
            f"Missing evid example={missing_value[0]!r}; "
            f"missing counts: evid1={missing_e1_count}, evid2={missing_e2_count}; "
            f"example missing evid1={missing_e1_examples}, evid2={missing_e2_examples}. "
            f"Observed types: origins0.evid={origin_evid_type}, "
            f"dtimes.evid1={dt_e1_type}, dtimes.evid2={dt_e2_type}. "
            "Likely causes: (1) dtimes references events absent from origins0, "
            "(2) origins0/dtimes came from different filtering steps or bundle files, "
            "(3) evid dtype mismatch (e.g., int vs string)."
        )
    dtimes = joined.with_columns(
        [
            pl.col("evid1_idx").cast(pl.Int64),
            pl.col("evid2_idx").cast(pl.Int64),
            pl.Series(np.arange(n_rows0)).alias("arid"),
        ]
    )

    # Stable integer station index per row for station-dependent latent models.
    # (Uses (network,station) rather than receiver XYZ to avoid float-key grouping.)
    try:
        sta_keys = (
            dtimes.select([pl.col("network"), pl.col("station")])
            .unique(maintain_order=True)
            .with_row_index("sta_idx")
        )
        dtimes = dtimes.join(sta_keys, on=["network", "station"], how="left")
        # Guard against nulls in join keys (nulls do not match in Polars join).
        if dtimes["sta_idx"].null_count() > 0:
            dtimes = dtimes.with_columns(pl.col("sta_idx").fill_null(-1))
        sta_idx_np = dtimes["sta_idx"].to_numpy().astype(np.int64, copy=False)
        if sta_idx_np.size > 0:
            mn = int(sta_idx_np.min())
            mx = int(sta_idx_np.max())
            n_st = int(sta_keys.shape[0])
            if mn < 0 or mx >= n_st:
                raise ValueError(f"Invalid sta_idx after join: min={mn} max={mx} n_stations={n_st}")
        row_station_index = torch.tensor(sta_idx_np, dtype=torch.int64, device=device).contiguous()
        n_stations = int(sta_keys.shape[0])
        if n_stations < 0:
            n_stations = 0
    except Exception:
        row_station_index = None
        n_stations = 0

    projector = Proj(
        proj="laea",
        lat_0=params["lat_min"],
        lon_0=params["lon_min"],
        datum="WGS84",
        units="km",
    )
    XX, YY = projector(
        origins0["longitude"].to_numpy(), origins0["latitude"].to_numpy()
    )

    X_src = torch.zeros(origins0.shape[0], 4, dtype=torch.float32, device=device)
    X_src[:, 0] = torch.tensor(XX, dtype=torch.float32, device=device)
    X_src[:, 1] = torch.tensor(YY, dtype=torch.float32, device=device)
    X_src[:, 2] = torch.tensor(
        origins0["depth"].to_numpy(), dtype=torch.float32, device=device
    )

    dX_src = torch.zeros_like(X_src)
    dX_src.requires_grad_()
    dX_src = torch.nn.Parameter(dX_src)

    # Materialize the index block once and reuse it for both the device tensor
    # and the CPU mirror (this was previously two full to_numpy() passes).
    II_cpu_np = dtimes[["evid1_idx", "evid2_idx"]].to_numpy().astype(np.int64, copy=False)
    II = torch.from_numpy(np.ascontiguousarray(II_cpu_np)).to(device=device)
    YY = torch.tensor(
        dtimes[["dt", "X", "Y", "Z", "phase"]].to_numpy(),
        dtype=torch.float32,
        device=device,
    )
    # Single-device execution: multi-GPU should be achieved via multiple independent processes
    # (see `python -m spider sample-multi` / `python -m spider locate-multi`), not torch.nn.DataParallel.
    model = model.to(device)

    batch_size_warmup = params["batch_size_warmup"]
    batch_size_sgld = params["batch_size_sgld"]
    N = dtimes.shape[0]
    info(f"Differential times rows N={int(N):,}", section="DATA")

    # Priors: strict enable keys are materialized by `validate_and_materialize_priors`.
    # If disabled, we create placeholder distributions purely to satisfy existing plumbing; they
    # are never used in the loss because compute_prior_loss gates by enable flags.
    event_prior_enable = bool(params["prior_event_enable"])

    prior_event_std = params.get("prior_event_std", None)
    if event_prior_enable and prior_event_std is None:
        raise KeyError("prior_event_std is required when prior_event_enable=true")
    if prior_event_std is None:
        prior_event_std = [9999.0, 9999.0, 9999.0, 9999.0]

    prior_event = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.zeros(len(prior_event_std), device=device, dtype=torch.float32),
        covariance_matrix=torch.diag(torch.tensor(prior_event_std, device=device, dtype=torch.float32) ** 2),
    )

    centroid_prior_enable = bool(params.get("prior_centroid_enable", False))
    prior_centroid_std = params.get("prior_centroid_std", None)
    if centroid_prior_enable and prior_centroid_std is None:
        raise KeyError("prior_centroid_std is required when prior_centroid_enable=true")
    if prior_centroid_std is None:
        prior_centroid_std = [9999.0, 9999.0, 9999.0, 9999.0]

    prior_centroid = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.zeros(len(prior_centroid_std), device=device, dtype=torch.float32),
        covariance_matrix=torch.diag(torch.tensor(prior_centroid_std, device=device, dtype=torch.float32) ** 2),
    )

    # Noise scales: fixed (scalar phase_unc only)
    phase_unc_list = params.get("phase_unc", [0.05, 0.08])
    scale_theta = torch.tensor(phase_unc_list, device=device, dtype=torch.float32)

    # Optimizer parameters (ΔX only; noise learning removed)
    opt_params: List[torch.nn.Parameter] = [dX_src]
    log_scale_theta: Optional[torch.nn.Parameter] = None

    optimizer = torch.optim.Adam(opt_params, lr=params["lr_warmup"])
    clamp_abs_dX = _parse_clamp_tensor(params, device)

    # Cluster analysis for connected components (used by gauge projection and graph-aware priors)
    # Build adjacency matrix from event pairings
    n_events = origins0.shape[0]
    row = II_cpu_np[:, 0]
    col = II_cpu_np[:, 1]
    # Undirected graph: adjacency is symmetric
    data = np.ones(len(row), dtype=int)
    adj = coo_matrix((data, (row, col)), shape=(n_events, n_events))
    # Note: connected_components treats the graph as undirected if connection='weak' (default for undirected graphs in concept)
    # But strictly for undirected we want symmetric. However, cs_graph.connected_components handles directed graphs
    # by finding strongly/weakly connected components. For our purpose (DD linkage), 'weak' connectivity
    # on the directed graph of pairs is sufficient (if A linked to B, they are in same cluster).
    n_components, labels = connected_components(csgraph=adj, directed=False, return_labels=True)

    info(f"Cluster analysis components={n_components}", section="GRAPH")
    cluster_ids = torch.tensor(labels, dtype=torch.int64, device=device)
    # Pre-compute counts for averaging
    # counts shape: (K,) -> (K, 1)
    counts = torch.bincount(cluster_ids, minlength=n_components).float().unsqueeze(1)
    cluster_counts = counts

    # Print cluster statistics
    if n_components > 0:
        c_sizes = counts.flatten().detach().cpu().numpy().astype(int)
        c_min, c_max, c_med = int(c_sizes.min()), int(c_sizes.max()), float(np.median(c_sizes))
        n_single = int(np.sum(c_sizes == 1))
        info(f"Cluster sizes min={c_min} max={c_max} median={c_med:.1f} singletons={n_single}", section="GRAPH")
        if n_components > 1:
            top_k = sorted(c_sizes, reverse=True)[:5]
            info(f"Largest clusters top5={top_k}", section="GRAPH")

    # Optional: DD graph k-hop clustering for shared_event_re (greedy non-overlapping balls).
    def _build_khop_clusters(indptr: np.ndarray, indices: np.ndarray, k: int) -> np.ndarray:
        n = int(indptr.shape[0] - 1)
        labels = -np.ones((n,), dtype=np.int64)
        if k <= 0:
            labels[:] = np.arange(n, dtype=np.int64)
            return labels
        from collections import deque
        cid = 0
        for i in range(n):
            if labels[i] != -1:
                continue
            labels[i] = cid
            dq = deque([i])
            depth = deque([0])
            while dq:
                v = dq.popleft()
                d = depth.popleft()
                if d >= k:
                    continue
                start = int(indptr[v])
                end = int(indptr[v + 1])
                for nb in indices[start:end]:
                    if labels[nb] == -1:
                        labels[nb] = cid
                        dq.append(int(nb))
                        depth.append(d + 1)
            cid += 1
        return labels

    try:
        se_cluster_mode = str(params.get("_shared_event_re_cluster_mode", "none")).strip().lower()
        se_cluster_k_raw = params.get("_shared_event_re_cluster_k", 1)
        se_cluster_k = int(se_cluster_k_raw) if se_cluster_k_raw is not None else 1
    except Exception:
        se_cluster_mode = "none"
        se_cluster_k = 1
    if se_cluster_mode == "component":
        try:
            params["_shared_event_re_cluster_ids"] = cluster_ids
            params["_shared_event_re_cluster_k"] = int(se_cluster_k)
            counts_k = np.bincount(labels, minlength=int(n_components))
            if counts_k.size > 0:
                cmin = int(counts_k.min())
                cmax = int(counts_k.max())
                cmed = float(np.median(counts_k))
                ns = int(np.sum(counts_k == 1))
                info(
                    f"shared_event_re component clusters "
                    f"n_clusters={int(counts_k.size)} min={cmin} max={cmax} median={cmed:.1f} singletons={ns}",
                    section="GRAPH",
                )
        except Exception as e:
            warn(f"shared_event_re component clustering failed; falling back to no clustering: {e}", section="GRAPH")
    elif se_cluster_mode == "dd_khop":
        try:
            row_np = II_cpu_np[:, 0]
            col_np = II_cpu_np[:, 1]
            if row_np.size > 0:
                row_all = np.concatenate([row_np, col_np], axis=0)
                col_all = np.concatenate([col_np, row_np], axis=0)
                data_all = np.ones((int(row_all.shape[0]),), dtype=np.int8)
                adj_csr = csr_matrix((data_all, (row_all, col_all)), shape=(n_events, n_events))
                khop_labels = _build_khop_clusters(adj_csr.indptr, adj_csr.indices, int(se_cluster_k))
            else:
                khop_labels = np.arange(int(n_events), dtype=np.int64)
            khop_ids = torch.tensor(khop_labels, dtype=torch.int64, device=device)
            params["_shared_event_re_cluster_ids"] = khop_ids
            params["_shared_event_re_cluster_k"] = int(se_cluster_k)
            # Stats
            counts_k = np.bincount(khop_labels, minlength=int(khop_labels.max() + 1) if khop_labels.size > 0 else 0)
            if counts_k.size > 0:
                cmin = int(counts_k.min())
                cmax = int(counts_k.max())
                cmed = float(np.median(counts_k))
                ns = int(np.sum(counts_k == 1))
                info(
                    f"shared_event_re k-hop clusters k={int(se_cluster_k)} "
                    f"n_clusters={int(counts_k.size)} min={cmin} max={cmax} median={cmed:.1f} singletons={ns}",
                    section="GRAPH",
                )
        except Exception as e:
            warn(f"shared_event_re k-hop clustering failed; falling back to no clustering: {e}", section="GRAPH")

    # Optional: unique event-event pair count distribution (diagnostics only).
    pair_count_stats_enable = bool(params.get("pair_count_stats_enable", False))

    pair_counts: Optional[pl.DataFrame] = None
    if pair_count_stats_enable:
        try:
            t0 = time.time()
            pairs = dtimes.select(
                [
                    pl.when(pl.col("evid1_idx") <= pl.col("evid2_idx"))
                    .then(pl.col("evid1_idx"))
                    .otherwise(pl.col("evid2_idx"))
                    .cast(pl.Int64)
                    .alias("u"),
                    pl.when(pl.col("evid1_idx") <= pl.col("evid2_idx"))
                    .then(pl.col("evid2_idx"))
                    .otherwise(pl.col("evid1_idx"))
                    .cast(pl.Int64)
                    .alias("v"),
                ]
            )
            pair_counts = pairs.group_by(["u", "v"]).len().rename({"len": "pair_count"})

            if pair_count_stats_enable:
                stats = pair_counts.select(
                    [
                        pl.len().alias("n_pairs"),
                        pl.col("pair_count").min().alias("min"),
                        pl.col("pair_count").quantile(0.50, interpolation="nearest").alias("p50"),
                        pl.col("pair_count").quantile(0.90, interpolation="nearest").alias("p90"),
                        pl.col("pair_count").quantile(0.95, interpolation="nearest").alias("p95"),
                        pl.col("pair_count").quantile(0.99, interpolation="nearest").alias("p99"),
                        pl.col("pair_count").max().alias("max"),
                    ]
                ).row(0)
                n_pairs, pc_min, pc_p50, pc_p90, pc_p95, pc_p99, pc_max = stats
                dt_sec = time.time() - t0
                info(
                    "Pair-count stats "
                    f"n_pairs={int(n_pairs):,} min={int(pc_min)} p50={int(pc_p50)} "
                    f"p90={int(pc_p90)} p95={int(pc_p95)} p99={int(pc_p99)} max={int(pc_max)} "
                    f"dt={dt_sec:.1f}s",
                    section="GRAPH",
                )
        except Exception as e:
            warn(f"Pair-count stats unavailable: {e}", section="GRAPH")
            pair_counts = None

    # No static graph partitioning is used for current sampler preconditioners.
    precond_block_members = None
    precond_block_sizes = None
    precond_n_blocks = 0

    # (Laplacian prior removed: no static Laplacian graph build or Laplacian hyperparameters.)

    # Initial precision matrix P0 from prior_event_std
    # prior_event_std is list/array of 4 stds
    # P0 = diag(1/std^2)
    # Use the resolved prior_event_std (may be placeholder if prior disabled)
    p_std = prior_event_std
    P0_init_diag = torch.diag(1.0 / (torch.tensor(p_std, device=device, dtype=torch.float32) ** 2))

    # Expand to (K, 4, 4) for cluster-specific priors
    # If no clusters found (n_components=0), we treat it as 1 cluster or fail gracefully
    n_clusters = max(1, n_components)
    P0_init = P0_init_diag.unsqueeze(0).expand(n_clusters, 4, 4).clone()

    # Hierarchical event prior is a refinement of the event prior; if event prior is disabled,
    # force hierarchical mode off as well. (Strict keys are materialized by validate_and_materialize_priors.)
    event_prior_enable = bool(params["prior_event_enable"])
    hierarchical_prior_enable = bool(params["hierarchical_event_prior"]) and event_prior_enable
    if hierarchical_prior_enable:
        info(f"Hierarchical event prior enabled n_clusters={n_clusters}", section="PRIORS")

    state = LocateState(
        params=params,
        device=device,
        projector=projector,
        origins0=origins0,
        dtimes=dtimes,
        X_src=X_src,
        dX_src=dX_src,
        II=II,
        YY=YY,
        row_station_index=row_station_index,
        n_stations=int(n_stations),
        model=model,
        prior_event=prior_event,
        prior_centroid=prior_centroid,
        optimizer=optimizer,
        N=N,
        batch_size_warmup=batch_size_warmup,
        batch_size_sgld=batch_size_sgld,
        scale_theta=scale_theta,
        nuisance_enable=False,
        nuisance_alpha=None,
        nuisance_k_index=None,
        nuisance_basis="poly1",
        nuisance_M=0,
        stats_tensor=torch.zeros(8, device=device),
        samples=[],
        sample_count=0,
        global_step_count=0,
        clamp_abs_dX=clamp_abs_dX,
        cluster_ids=cluster_ids,
        cluster_counts=cluster_counts,
        precond_block_members=precond_block_members,
        precond_block_sizes=precond_block_sizes,
        precond_n_blocks=int(precond_n_blocks),
        hierarchical_prior_enable=hierarchical_prior_enable,
        event_precision_matrix=P0_init,
    )
    # Expose a few runtime tensors in params so modeling.py can implement component-wise correlated
    # likelihoods without needing the full LocateState object.
    try:
        params["_runtime_event_cluster_ids"] = cluster_ids
        params["_runtime_n_stations"] = int(n_stations)
        params["_runtime_n_components"] = int(n_components)
    except Exception:
        pass
    # Initialize event-centric batching flag (mapping is built lazily when used)
    state.event_batch_enable = bool(params.get("event_batch_enable", False))
    # Cache CPU mirror of II for fast owner bucketing
    try:
        state._II_cpu = II_cpu_np
    except Exception:
        state._II_cpu = None
    # If nuisance field enabled in params (and SSST not in use), augment state and optimizer
    if bool(params.get("nuisance_enable_phase1", False)):
        # Build station-phase mapping
        sta_keys = (
            dtimes.select([
                (pl.col("network").cast(pl.Utf8) + pl.lit(".") + pl.col("station").cast(pl.Utf8)).alias("sta"),
                pl.col("phase").alias("phase"),
            ])
        )
        # Vectorized (sta, phase) -> index mapping via a join (order-preserving);
        # the previous per-row Python loop was O(n_dtimes) dict/tuple churn.
        unique_sp = sta_keys.unique(maintain_order=True).with_row_index("__sp_idx")
        n_sp = int(unique_sp.shape[0])
        k_np = (
            sta_keys.with_row_index("__row")
            .join(unique_sp, on=["sta", "phase"], how="left")
            .sort("__row")["__sp_idx"]
            .to_numpy()
            .astype(np.int64, copy=False)
        )
        nuisance_k_index = torch.tensor(k_np, dtype=torch.int64, device=device).contiguous()
        nuisance_basis = str(params.get("nuisance_basis", "poly1"))
        if nuisance_basis == "poly2":
            nuisance_M = 9
        else:
            nuisance_M = 3
        nuisance_alpha = torch.nn.Parameter(torch.zeros(n_sp, nuisance_M, dtype=torch.float32, device=device))
        state.nuisance_enable = True
        state.nuisance_alpha = nuisance_alpha
        state.nuisance_k_index = nuisance_k_index
        state.nuisance_basis = nuisance_basis
        state.nuisance_M = nuisance_M
        # Add parameters to optimizer (Phase 1)
        state.optimizer.add_param_group({"params": [nuisance_alpha]})
    return state


