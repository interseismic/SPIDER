from __future__ import annotations

from typing import List, Optional
import math
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from pyproj import Proj
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from spider.core.state import LocateState, _attach_dd_preconditioner_metric, _parse_clamp_tensor
from spider.utils.console import info, warn


def _build_initial_state(
    params: dict,
    origins0: pl.DataFrame,
    dtimes: pl.DataFrame,
    model: nn.Module,
    device: torch.device,
) -> LocateState:
    """Prepare tensors, priors, and optimizer."""

    evid_to_row = {row["evid"]: idx for idx, row in enumerate(origins0.iter_rows(named=True))}

    evid1_idx = [evid_to_row[x] for x in dtimes["evid1"]]
    evid2_idx = [evid_to_row[x] for x in dtimes["evid2"]]
    dtimes = dtimes.with_columns(
        [
            pl.Series(evid1_idx).alias("evid1_idx"),
            pl.Series(evid2_idx).alias("evid2_idx"),
            pl.Series(np.arange(dtimes.shape[0])).alias("arid"),
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

    II = torch.tensor(
        dtimes[["evid1_idx", "evid2_idx"]].to_numpy(),
        dtype=torch.int64,
        device=device,
    )
    dd_event_degree: Optional[torch.Tensor] = None
    if bool(params.get("dd_prec_enable", False)):
        dd_prec_dims = params.get("dd_prec_dims", [0, 1, 2, 3])
        if not isinstance(dd_prec_dims, list) or len(dd_prec_dims) == 0:
            dd_prec_dims = [0, 1, 2, 3]
        dd_prec_dims = [int(x) for x in dd_prec_dims if int(x) in (0, 1, 2, 3)]
        if len(dd_prec_dims) == 0:
            dd_prec_dims = [0, 1, 2, 3]
        deg_counts = (
            pl.concat(
                [
                    dtimes.select(pl.col("evid1").alias("evid")),
                    dtimes.select(pl.col("evid2").alias("evid")),
                ]
            )
            .group_by("evid")
            .len()
            .rename({"len": "deg"})
        )
        deg_map = {int(row["evid"]): max(1.0, float(row["deg"])) for row in deg_counts.iter_rows(named=True)}
        degree_vec = [
            deg_map.get(int(evid), 1.0)
            for evid in origins0["evid"].to_numpy()
        ]
        dd_event_degree = torch.tensor(degree_vec, dtype=torch.float32, device=device)
        # Normalize by the mean degree so the average preconditioner stays unchanged.
        if dd_event_degree.numel() > 0:
            mean_degree = float(dd_event_degree.mean().item())
            if not math.isfinite(mean_degree) or mean_degree <= 0.0:
                mean_degree = 1.0
            dd_event_degree = dd_event_degree / mean_degree
        # Optional: apply DD degree scaling only to selected ΔX dims.
        # Default (backward compatible) is all dims [0,1,2,3].
        try:
            if set(dd_prec_dims) != {0, 1, 2, 3}:
                Ne = int(X_src.shape[0])
                deg4 = torch.ones((Ne, 4), dtype=torch.float32, device=device)
                for d in dd_prec_dims:
                    deg4[:, int(d)] = dd_event_degree
                dd_event_degree = deg4
        except Exception:
            # Best-effort: keep scalar degree vector if shaping fails
            pass
    II_cpu_np = dtimes[["evid1_idx", "evid2_idx"]].to_numpy().astype(np.int64, copy=False)
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

    # Priors: strict enable keys are materialized by `validate_and_materialize_priors`.
    # If disabled, we create placeholder distributions purely to satisfy existing plumbing; they
    # are never used in the loss because compute_prior_loss gates by enable flags.
    event_prior_enable = bool(params["prior_event_enable"])
    centroid_prior_enable = bool(params["prior_centroid_enable"])

    prior_event_std = params.get("prior_event_std", None)
    if event_prior_enable and prior_event_std is None:
        raise KeyError("prior_event_std is required when prior_event_enable=true")
    if prior_event_std is None:
        prior_event_std = [9999.0, 9999.0, 9999.0, 9999.0]

    prior_centroid_std = params.get("prior_centroid_std", None)
    if centroid_prior_enable and prior_centroid_std is None:
        raise KeyError("prior_centroid_std is required when prior_centroid_enable=true")
    if prior_centroid_std is None:
        prior_centroid_std = [9999.0, 9999.0, 9999.0, 9999.0]

    prior_event = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.zeros(len(prior_event_std), device=device, dtype=torch.float32),
        covariance_matrix=torch.diag(torch.tensor(prior_event_std, device=device, dtype=torch.float32) ** 2),
    )
    prior_centroid = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.zeros(len(prior_centroid_std), device=device, dtype=torch.float32),
        covariance_matrix=torch.diag(torch.tensor(prior_centroid_std, device=device, dtype=torch.float32) ** 2),
    )

    # Noise scales: fixed or learnable
    phase_unc_list = params.get("phase_unc", [0.05, 0.08])
    scale_theta = torch.tensor(phase_unc_list, device=device, dtype=torch.float32)
    # Noise scale learning is controlled only by the hard-break schema key:
    #   model.likelihood.learn_noise_scale  -> materialized as params["learn_noise_scale"]
    #
    # Do NOT honor legacy aliases like `learn_phase_unc` here, because Phase-2 bundles may carry
    # stale keys from older runs. That can silently re-enable noise learning even when the current
    # config sets learn_noise_scale=false.
    learn_noise = bool(params.get("learn_noise_scale", False))

    # Optimizer parameters
    opt_params: List[torch.nn.Parameter] = [dX_src]
    log_scale_theta: Optional[torch.nn.Parameter] = None
    if learn_noise:
        # log σ parameters (ensure positivity via exp)
        log_scale_init = torch.log(scale_theta.clamp_min(1e-8)).detach()
        # If a noise prior is enabled, initialize to the *median* of that prior.
        # For LogNormal(loc, scale): median = exp(loc) ⇒ log(median) = loc.
        try:
            if bool(params.get("prior_noise_enable", False)):
                prior_type = str(params.get("noise_prior", "none")).strip().lower()
                if prior_type in {"lognormal", "log_normal"}:
                    loc = params.get("noise_prior_loc", None)
                    if isinstance(loc, (list, tuple)) and len(loc) == 2:
                        log_scale_init = torch.tensor(loc, device=device, dtype=torch.float32)
        except Exception:
            # Fall back to phase_unc-based initialization.
            pass
        log_scale_theta = torch.nn.Parameter(log_scale_init.clone())
        opt_params.append(log_scale_theta)

    optimizer = torch.optim.Adam(opt_params, lr=params["lr_warmup"])
    clamp_abs_dX = _parse_clamp_tensor(params, device)

    # Cluster analysis for centroid priors
    # Build adjacency matrix from event pairings
    n_events = origins0.shape[0]
    row = np.array(evid1_idx)
    col = np.array(evid2_idx)
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

    # Optional: unique event-event pair count distribution (useful for graph-aware features like blockdiag_fisher).
    pair_count_stats_enable = bool(params.get("pair_count_stats_enable", False))

    pair_counts: Optional[pl.DataFrame] = None
    # Also compute pair_counts when blockdiag_fisher partitioning is requested (static, pre-optimization).
    # This is only used by pSGLD's blockdiag_fisher preconditioner.
    sampler_backend = str(params.get("sampler_backend", "psgld")).strip().lower()
    want_blockdiag_partition = (
        sampler_backend == "psgld"
        and str(params.get("sampler_preconditioner", "")).strip().lower() in {"blockdiag_fisher", "matrix_ema"}
        and int(params.get("blockdiag_fisher_max_cluster_size", 1)) > 1
    )
    if pair_count_stats_enable or want_blockdiag_partition:
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
            if pair_count_stats_enable or want_blockdiag_partition:
                warn(f"Pair-count stats unavailable: {e}", section="GRAPH")
            pair_counts = None

    # --- Optional: static disjoint partition for blockdiag_fisher ---
    precond_block_members = None
    precond_block_sizes = None
    precond_n_blocks = 0
    if want_blockdiag_partition and pair_counts is not None:
        try:
            from spider.analysis.graph_partition import partition_graph_disjoint_blocks

            S = int(params.get("blockdiag_fisher_max_cluster_size", 1))
            u_np = pair_counts["u"].to_numpy().astype(np.int64, copy=False)
            v_np = pair_counts["v"].to_numpy().astype(np.int64, copy=False)
            w_np = pair_counts["pair_count"].to_numpy().astype(np.float64, copy=False)
            t_part0 = time.time()
            method = str(params.get("blockdiag_fisher_partition_method", "auto")).strip().lower()
            info(
                f"blockdiag_fisher partition start method={method} max_cluster_size={S} n_events={int(n_events):,} n_pairs={int(u_np.size):,}",
                section="GRAPH",
            )
            part = partition_graph_disjoint_blocks(
                n_nodes=int(n_events),
                u=u_np,
                v=v_np,
                w=w_np,
                max_cluster_size=int(S),
                method=method,
                min_balance=float(params.get("blockdiag_fisher_min_balance", 0.30)),
                seed=int(params.get("blockdiag_fisher_partition_seed", 0)),
            )
            blocks = part.blocks
            K = int(len(blocks))
            precond_n_blocks = K
            members = np.full((K, int(S)), -1, dtype=np.int64)
            sizes = np.zeros((K,), dtype=np.int64)
            for k, nd in enumerate(blocks):
                nd = np.asarray(nd, dtype=np.int64)
                sizes[k] = int(nd.size)
                members[k, : int(nd.size)] = nd
            precond_block_members = torch.tensor(members, dtype=torch.int64, device=device)
            precond_block_sizes = torch.tensor(sizes, dtype=torch.int64, device=device)

            if K > 0:
                srt = np.sort(sizes)
                dt_part = time.time() - t_part0
                info(
                    f"blockdiag_fisher partition: max_cluster_size={S} blocks={K} "
                    f"size_min={int(srt[0])} size_med={float(np.median(srt)):.1f} size_max={int(srt[-1])} "
                    f"dt={dt_part:.1f}s",
                    section="GRAPH",
                )
        except Exception as e:
            warn(f"blockdiag_fisher partitioning failed; falling back to per-event blocks: {e}", section="GRAPH")
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
        dd_event_degree=dd_event_degree,
        model=model,
        prior_event=prior_event,
        prior_centroid=prior_centroid,
        optimizer=optimizer,
        N=N,
        batch_size_warmup=batch_size_warmup,
        batch_size_sgld=batch_size_sgld,
        scale_theta=None if learn_noise else scale_theta,
        log_scale_theta=log_scale_theta,
        learn_noise_scale=learn_noise,
        nuisance_enable=False,
        nuisance_alpha=None,
        nuisance_k_index=None,
        nuisance_basis="poly1",
        nuisance_M=0,
        stats_tensor=torch.zeros(8, device=device),
        samples=[],
        noise_log_scales=[],
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
    _attach_dd_preconditioner_metric(state)
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
        unique_sp = sta_keys.unique(maintain_order=True)
        sp_keys = [(row["sta"], int(row["phase"])) for row in unique_sp.iter_rows(named=True)]
        sp_to_idx = {k: idx for idx, k in enumerate(sp_keys)}
        k_list = []
        for row in sta_keys.iter_rows(named=True):
            k_list.append(sp_to_idx[(row["sta"], int(row["phase"]))])
        nuisance_k_index = torch.tensor(k_list, dtype=torch.int64, device=device).contiguous()
        nuisance_basis = str(params.get("nuisance_basis", "poly1"))
        if nuisance_basis == "poly2":
            nuisance_M = 9
        else:
            nuisance_M = 3
        nuisance_alpha = torch.nn.Parameter(torch.zeros(len(sp_keys), nuisance_M, dtype=torch.float32, device=device))
        state.nuisance_enable = True
        state.nuisance_alpha = nuisance_alpha
        state.nuisance_k_index = nuisance_k_index
        state.nuisance_basis = nuisance_basis
        state.nuisance_M = nuisance_M
        # Add parameters to optimizer (Phase 1)
        state.optimizer.add_param_group({"params": [nuisance_alpha]})
    return state


