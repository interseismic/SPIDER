from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import math

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from spider.utils.console import info, warn

from pyproj import Proj



# Standardized stdout helper
def _log(*parts, section: str = "STATE", **_kwargs) -> None:
    msg = " ".join(str(p) for p in parts)
    low = msg.strip().lower()
    if low.startswith("warning") or low.startswith("error"):
        warn(msg, section=section)
    else:
        info(msg, section=section)

@dataclass
class LocateState:
    """Aggregated runtime state for earthquake location and sampling."""

    params: dict
    device: torch.device
    projector: Proj
    origins0: pl.DataFrame
    dtimes: pl.DataFrame

    # Tensors
    X_src: torch.Tensor
    dX_src: torch.nn.Parameter  # ΔX_src
    II: torch.Tensor
    YY: torch.Tensor

    # Model and priors
    model: nn.Module
    prior_event: torch.distributions.Distribution
    prior_centroid: torch.distributions.Distribution

    # Optim/SGD
    optimizer: torch.optim.Optimizer
    sampler: Optional[torch.optim.Optimizer] = None
    sampler_backend: str = "psgld"

    # Config
    N: int = 0
    # Per-row station index (stable integer id for (network,station), aligned with II/YY row order).
    row_station_index: Optional[torch.Tensor] = None    # shape (N,), int64 on device
    n_stations: int = 0
    batch_size_warmup: int = 0
    batch_size_sgld: int = 0
    scale_theta: Optional[torch.Tensor] = None  # [σ_p, σ_s] (fixed, if not learning)

    # Stats/samples
    stats_tensor: Optional[torch.Tensor] = None
    samples: Optional[List[torch.Tensor]] = None
    sample_count: int = 0
    global_step_count: int = 0
    # Noise learning removed; no per-step noise traces.

    II_epoch: Optional[torch.Tensor] = None
    YY_epoch: Optional[torch.Tensor] = None
    row_station_index_epoch: Optional[torch.Tensor] = None
    _perm_epoch: Optional[torch.Tensor] = None
    # Optional per-dimension clamp for ΔX_src (abs max for [dx, dy, dz, dt])
    clamp_abs_dX: Optional[torch.Tensor] = None
    dd_event_degree: Optional[torch.Tensor] = None
    # (receiver-centric residual scaling removed)
    # Nuisance field (Phase 1): station-phase basis expansion coefficients and mapping
    nuisance_enable: bool = False
    nuisance_alpha: Optional[torch.nn.Parameter] = None   # shape (K, M)
    nuisance_k_index: Optional[torch.Tensor] = None       # shape (N,), station-phase index per observation row
    nuisance_basis: str = "poly1"
    nuisance_M: int = 0
    # Cluster ids for connected components (used by gauge projection and graph-aware priors)
    cluster_ids: Optional[torch.Tensor] = None         # shape (Ne,) int64
    cluster_counts: Optional[torch.Tensor] = None      # shape (K, 1) float32

    # Sampler preconditioner partition (disjoint clusters for blockdiag_fisher)
    # This is separate from `cluster_ids` above (which is used for component detection / gauge projection).
    # If enabled, blocks partition each connected component into disjoint groups of size <= max_cluster_size
    # based on weighted event-event edges (pair_count).
    precond_block_members: Optional[torch.Tensor] = None    # shape (K, S) int64, padded with -1
    precond_block_sizes: Optional[torch.Tensor] = None      # shape (K,) int64
    precond_n_blocks: int = 0

    # Hierarchical Prior (Global Precision Matrix)
    hierarchical_prior_enable: bool = False
    event_precision_matrix: Optional[torch.Tensor] = None  # shape (K, 4, 4) P0 per cluster

    # SVRG State
    svrg_enable: bool = False
    svrg_dX_snapshot: Optional[torch.Tensor] = None        # Snapshot of params at start of epoch
    svrg_grad_full: Optional[torch.Tensor] = None          # Full gradient at snapshot

    # Event-centric batching
    event_batch_enable: bool = False
    _event_to_rows: Optional[List[np.ndarray]] = None   # CPU arrays of edge row indices per event
    # Owner-bucket batching storage (contiguous per-bucket slices)
    _bucket_rows_order: Optional[torch.Tensor] = None        # concatenated row order (device)
    _bucket_offsets: Optional[torch.Tensor] = None           # int64 offsets (len = num_buckets+1, device)
    _bucket_II: Optional[torch.Tensor] = None                # II reordered by bucket order (device)
    _bucket_YY: Optional[torch.Tensor] = None                # YY reordered by bucket order (device)
    _bucket_station_index: Optional[torch.Tensor] = None     # station index reordered by bucket order (device)
    _bucket_comp_index: Optional[torch.Tensor] = None        # component id per row reordered by bucket order (device)
    _bucket_p_counts: Optional[torch.Tensor] = None          # int64 per-bucket P-row counts (when reorder_all=True)
    # Legacy per-bucket per-phase precomputed event graph maps.
    # Whitening-first shared_event_re uses canonical grouping plans built at likelihood time.
    _bucket_nodes_p: Optional[list] = None
    _bucket_u_p: Optional[list] = None
    _bucket_v_p: Optional[list] = None
    _bucket_nodes_s: Optional[list] = None
    _bucket_u_s: Optional[list] = None
    _bucket_v_s: Optional[list] = None
    # Per-bucket pre-chunked phase blocks for correlated likelihood (station_phase grouping).
    # Each entry is a list of dicts with keys: i0,i1,nodes,u,v (all relative to bucket slice).
    _bucket_chunks_p: Optional[list] = None
    _bucket_chunks_s: Optional[list] = None

    # Optional shared_event_re whitening cache (static covariance operator).
    shared_event_re_whitening_cache: Optional[dict] = None
    # CPU mirror of II for fast owner bucketing (kept in sync on rebuilds)
    _II_cpu: Optional[np.ndarray] = None
    # Owner-bucket caching
    _bucket_last_epoch: Optional[int] = None                 # last epoch index we rebuilt buckets

    def begin_epoch_rr(self, *, seed: Optional[int] = None, use_full_N: bool = True, shuffle: bool = True) -> None:
        """
        Prepare random-reshuffled (without replacement) contiguous tensors for this epoch.
        After calling, use `state.II_epoch` and `state.YY_epoch` in place of `state.II`, `state.YY`
        for the duration of the epoch.

        Args:
            seed: Optional integer for reproducible reshuffles (e.g., pass `epoch`).
            use_full_N: If True, use `self.N` rows; otherwise infer from `self.II.shape[0]`.
            shuffle: If False, skip the random permutation and expose contiguous views in original order.
        """
        # Keep row counts consistent across all per-edge tensors.
        # Root fix for CUDA IndexKernel asserts: never allow `self.N` to drift away from tensor lengths.
        #
        # Canonical source of truth is the actual tensor lengths; if something drifted,
        # we repair it here once, then proceed safely.
        N_ii = int(self.II.shape[0])
        N_yy = int(self.YY.shape[0])
        N = int(min(N_ii, N_yy))
        if self.row_station_index is not None:
            try:
                N_sta = int(self.row_station_index.shape[0])
                N = int(min(N, N_sta))
            except Exception:
                self.row_station_index = None
                self.n_stations = 0
        # If lengths disagree, truncate all to the common prefix length to restore invariants.
        # (The correct long-term fix is to keep these aligned at every filtering step; this
        # makes the system robust even if a future filter forgets.)
        if (N_ii != N) or (N_yy != N) or (self.row_station_index is not None and int(self.row_station_index.shape[0]) != N):
            try:
                _log(
                    f"Warning: row-count mismatch repaired in begin_epoch_rr: "
                    f"II={N_ii} YY={N_yy} "
                    f"sta_idx={(int(self.row_station_index.shape[0]) if self.row_station_index is not None else 'None')} "
                    f"→ using N={N}",
                    flush=True,
                )
            except Exception:
                pass
            if N_ii != N:
                self.II = self.II[:N].contiguous()
            if N_yy != N:
                self.YY = self.YY[:N].contiguous()
            if self.row_station_index is not None and int(self.row_station_index.shape[0]) != N:
                try:
                    self.row_station_index = self.row_station_index[:N].contiguous()
                except Exception:
                    self.row_station_index = None
                    self.n_stations = 0
            # Keep optional polars df aligned if possible
            try:
                if hasattr(self, "dtimes") and isinstance(self.dtimes, pl.DataFrame) and int(self.dtimes.shape[0]) != N:
                    self.dtimes = self.dtimes.head(N)
            except Exception:
                pass
            # Invalidate any CPU mirror that might have stale length
            try:
                if getattr(self, "_II_cpu", None) is not None and int(getattr(self, "_II_cpu").shape[0]) != N:
                    self._II_cpu = None
            except Exception:
                pass
        # Authoritative N for this epoch is the tensor length.
        self.N = int(N)

        if not bool(shuffle):
            # No permutation: just expose per-epoch views in original order.
            # (Avoid `.contiguous()` here to prevent copying massive tensors.)
            self._perm_epoch = None
            self.II_epoch = self.II
            self.YY_epoch = self.YY
            if self.row_station_index is not None and int(self.row_station_index.shape[0]) == int(self.N):
                self.row_station_index_epoch = self.row_station_index
            else:
                self.row_station_index_epoch = None
            return

        # Build permutation once per epoch (CPU RNG for determinism across devices/processes).
        # This is especially important for torchrun/DDP where each rank uses a different CUDA device.
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(int(seed))
        perm = torch.randperm(int(self.N), generator=gen, device="cpu")
        perm = perm.to(device=self.II.device)

        # Create contiguous, permuted views so your [i_start:i_end] slicing stays valid
        self.II_epoch = self.II.index_select(0, perm).contiguous()
        self.YY_epoch = self.YY.index_select(0, perm).contiguous()

        # (optional) stash for debugging/repro
        self._perm_epoch = perm
        # Maintain station index per-epoch view if present
        if self.row_station_index is not None:
            try:
                # After repair above, this should always match.
                if int(self.row_station_index.shape[0]) == int(self.N):
                    self.row_station_index_epoch = self.row_station_index.index_select(0, perm).contiguous()
                else:
                    self.row_station_index_epoch = None
            except Exception:
                self.row_station_index_epoch = None


def _parse_clamp_tensor(params: dict, device: torch.device) -> Optional[torch.Tensor]:
    """Parse optional max_abs_dX clamp from params.

    Accepts either a scalar or a list of length 3 or 4. If a length-3 list is
    provided, the dt clamp is taken from `max_abs_dt` if present, otherwise
    left effectively disabled (inf).
    """
    v = params.get("max_abs_dX", None)
    if v is None:
        return None
    try:
        if isinstance(v, (int, float)):
            vals = [float(v), float(v), float(v), float(v)]
        else:
            vals = [float(x) for x in v]
            if len(vals) == 3:
                dtc = float(params.get("max_abs_dt", float("inf")))
                vals.append(dtc)
            elif len(vals) != 4:
                _log("Warning: max_abs_dX must be a scalar or list of length 3 or 4; disabling clamp.")
                return None
        t = torch.tensor(vals, device=device, dtype=torch.float32)
        t = torch.abs(t)
        return t
    except Exception as e:
        _log(f"Warning: could not parse max_abs_dX: {e}; disabling clamp.")
        return None


@torch.no_grad()
def _clamp_dX_inplace(state: LocateState) -> None:
    """Clamp ΔX_src in-place per-dimension if clamp is configured.

    Any non-finite or non-positive clamp entries are treated as disabled.
    """
    clamp = state.clamp_abs_dX
    if clamp is None:
        return
    for dim in range(4):
        c = float(clamp[dim].item())
        if not math.isfinite(c) or c <= 0.0:
            continue
        state.dX_src[:, dim].clamp_(-c, c)


def _attach_dd_preconditioner_metric(state: LocateState) -> None:
    """Attach per-event degree tensor to ΔX_src for DD preconditioning."""
    deg = state.dd_event_degree
    if deg is None:
        if hasattr(state.dX_src, "_dd_degree"):
            try:
                delattr(state.dX_src, "_dd_degree")
            except AttributeError:
                pass
        return
    try:
        # Support either:
        # - deg shape (Ne,)   -> broadcast across 4 dims
        # - deg shape (Ne,4)  -> per-dimension degree scaling (e.g., apply to XYZ only)
        if isinstance(deg, torch.Tensor) and deg.ndim == 2:
            setattr(state.dX_src, "_dd_degree", deg)
        else:
            setattr(state.dX_src, "_dd_degree", deg.view(-1, 1))
    except Exception as exc:
        _log(f"Warning: could not attach DD preconditioner tensor: {exc}")


def _current_noise_scales(state: LocateState) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (σ_p, σ_s) as tensors on the correct device."""
    assert state.scale_theta is not None, "Noise scales not initialized"
    return state.scale_theta[0], state.scale_theta[1]


