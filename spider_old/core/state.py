from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import math

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from pyproj import Proj


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

    # Optim/SGD
    optimizer: torch.optim.Optimizer
    sampler: Optional[torch.optim.Optimizer] = None
    sampler_backend: str = "psgld"

    # Config
    N: int = 0
    # Per-row station index (stable integer id for (network,station), aligned with II/YY row order).
    # This is used by station-dependent likelihood extensions (e.g. shared_event_latent).
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
    # Per-bucket per-phase precomputed event graph maps (when reorder_all=True).
    # These avoid per-batch torch.unique / remapping in correlated likelihood.
    _bucket_nodes_p: Optional[list] = None
    _bucket_u_p: Optional[list] = None
    _bucket_v_p: Optional[list] = None
    _bucket_nodes_s: Optional[list] = None
    _bucket_u_s: Optional[list] = None
    _bucket_v_s: Optional[list] = None
    # Per-bucket pre-chunked phase blocks for correlated likelihood (when grouping='phase').
    # Each entry is a list of dicts with keys: i0,i1,nodes,u,v (all relative to bucket slice).
    _bucket_chunks_p: Optional[list] = None
    _bucket_chunks_s: Optional[list] = None

    # --- Optional uncollapsed shared-event latent random effects b[s,event,phase] ---
    shared_event_latent_b: Optional[torch.nn.Parameter] = None    # shape (n_stations, n_events, 2)
    # --- Optional uncollapsed slowness_re latents (component + station, per phase) ---
    slowness_re_comp_p: Optional[torch.nn.Parameter] = None       # shape (n_components,)
    slowness_re_comp_s: Optional[torch.nn.Parameter] = None       # shape (n_components,)
    slowness_re_station_p: Optional[torch.nn.Parameter] = None    # shape (n_stations,)
    slowness_re_station_s: Optional[torch.nn.Parameter] = None    # shape (n_stations,)
    # --- Optional DD-graph random effects (event latents, per phase) ---
    dd_graph_re_b_p: Optional[torch.nn.Parameter] = None          # shape (n_events,)
    dd_graph_re_b_s: Optional[torch.nn.Parameter] = None          # shape (n_events,)
    # Optional fixed station-geometry basis for shared_event_latent (dimension reduction across stations).
    # When enabled (see config: model.likelihood.shared_event_latent.station_basis),
    # shared_event_latent_b uses rank-R coefficients rather than per-station coefficients:
    #   - inducing_gp: b has shape (R, M_total, 2) and station weights W have shape (n_stations, R)
    #   - full (optional future): b could be (R, n_events, 2) with the same station weights
    shared_event_latent_station_basis_W: Optional[torch.Tensor] = None  # (n_stations, R) float32
    shared_event_latent_station_basis_r: int = 0
    # Optional inducing-point GP interpolation data (for parameterization == "inducing_gp").
    # These are per-event neighbor lists into the concatenated inducing index list.
    shared_event_latent_inducing_neighbor_idx: Optional[torch.Tensor] = None  # (n_events, m) int64, padded with -1
    shared_event_latent_inducing_neighbor_k: Optional[torch.Tensor] = None    # (n_events, m) float32 kernel values
    # Inducing point locations in event index space (concatenated global inducing list).
    # Legacy: used to map inducing index -> event id (to fetch inducing XYZ from X_src).
    # For Option-B fixed inducing geometry, prefer `shared_event_latent_inducing_xyz_km`.
    shared_event_latent_inducing_event_idx: Optional[torch.Tensor] = None    # (M_total,) int64 event indices
    # Fixed inducing-point XYZ locations (km), concatenated across components.
    # Used by slowness_inducing_gp Option-B to keep kernel geometry consistent while events move.
    shared_event_latent_inducing_xyz_km: Optional[torch.Tensor] = None       # (M_total, 3) float32
    # Inducing prior blocks (component-wise). Offsets index into the concatenated inducing list.
    shared_event_latent_inducing_offsets: Optional[torch.Tensor] = None       # (n_blocks+1,) int64
    shared_event_latent_inducing_K_blocks: Optional[list] = None              # list[Tensor], each (M_c, M_c)

    # --- Optional correlated forward-model error latent (corr_error) ---
    # Low-rank station basis W (n_stations, R) and event latent b (n_events, R, 2).
    corr_error_b: Optional[torch.nn.Parameter] = None
    corr_error_station_basis_W: Optional[torch.Tensor] = None  # (n_stations, R) float32
    corr_error_r: int = 0
    corr_error_u: Optional[torch.Tensor] = None  # (E,) int64
    corr_error_v: Optional[torch.Tensor] = None  # (E,) int64
    corr_error_w: Optional[torch.Tensor] = None  # (E,) float32
    corr_error_q_diag: float = 0.0
    # Connected components of the corr_error event graph (for per-component gauge fixing).
    corr_error_component_id: Optional[torch.Tensor] = None  # (n_events,) int64 labels
    corr_error_n_components: int = 0
    # Optional FITC-style diagonal correction (Stage 5) for inducing_gp:
    # Q_ee ≈ K_eU K_UU^{-1} K_Ue (approximated using the same m-neighbor subset as interpolation),
    # residual diag Λ_ee = max(0, 1 - Q_ee).
    shared_event_latent_inducing_fitc_q_diag: Optional[torch.Tensor] = None   # (n_events,) float32
    shared_event_latent_inducing_fitc_resid: Optional[torch.Tensor] = None    # (n_events,) float32
    # Fixed event-kernel graph (kNN Laplacian) built from MAP once.
    shared_event_latent_u: Optional[torch.Tensor] = None          # (E,) int64 (CPU or device)
    shared_event_latent_v: Optional[torch.Tensor] = None          # (E,) int64
    shared_event_latent_w: Optional[torch.Tensor] = None          # (E,) float32 weights
    shared_event_latent_q_diag: float = 0.0
    # Optional shared_event_re whitening cache (static covariance operator).
    shared_event_re_whitening_cache: Optional[dict] = None
    # Optional Student-t scale-mixture per-row precision (lambda) for robust likelihoods.
    student_t_lambda: Optional[torch.Tensor] = None
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
                print(
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
                print("Warning: max_abs_dX must be a scalar or list of length 3 or 4; disabling clamp.")
                return None
        t = torch.tensor(vals, device=device, dtype=torch.float32)
        t = torch.abs(t)
        return t
    except Exception as e:
        print(f"Warning: could not parse max_abs_dX: {e}; disabling clamp.")
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


@torch.no_grad()
def _apply_shared_event_latent_constraints_inplace(state: LocateState) -> None:
    """
    Apply optional identifiability constraints to shared_event_latent parameters in-place.

    Motivation:
    shared_event_latent enters the likelihood as (b_{s,e2,phase} - b_{s,e1,phase}).
    Any component of b that is *constant across stations* for a given event/phase is
    indistinguishable from an origin-time shift Δt_e and can "absorb" it.

    When enabled, we project out that station-common mode so Δt remains identifiable.
    """
    try:
        if not bool(state.params.get("_shared_event_latent_enabled", False)):
            return
        b = getattr(state, "shared_event_latent_b", None)
        if not isinstance(b, torch.Tensor) or b.ndim != 3 or int(b.shape[2]) != 2:
            return
        n_stations = int(getattr(state, "n_stations", 0) or 0)
        if n_stations <= 0:
            return
        # Two parameterizations:
        # - Per-station coefficients: b shape (n_stations, K, 2) where K is n_events or M_inducing.
        # - Station-basis coefficients: b shape (R, K, 2) with W (n_stations, R).
        if int(b.shape[0]) == int(n_stations):
            # Remove station-mean per (event/inducing, phase): b <- b - mean_s(b)
            mu = b.mean(dim=0, keepdim=True)
            b.sub_(mu)
            return

        W = getattr(state, "shared_event_latent_station_basis_W", None)
        if not isinstance(W, torch.Tensor) or W.ndim != 2 or int(W.shape[0]) != int(n_stations):
            return
        R = int(W.shape[1])
        if int(b.shape[0]) != int(R):
            return

        # In basis mode, reconstructed station coefficients are C = W @ A where A=b (R,K,2).
        # Station-mean is (1/S) 1^T C = ((1/S) W^T 1)^T A. Let r = (1/S) W^T 1 (R,).
        # Enforce r^T A == 0 by projecting A onto the orthogonal complement of r.
        ones = torch.ones((n_stations,), device=W.device, dtype=W.dtype)
        r = (W.transpose(0, 1).matmul(ones)) / float(max(1, n_stations))  # (R,)
        denom = torch.dot(r, r).clamp_min(0.0)
        if not torch.isfinite(denom) or float(denom.item()) <= 0.0:
            return
        # dot[k,phase] = r^T A[:,k,phase]
        dot = torch.tensordot(r.to(dtype=b.dtype, device=b.device), b, dims=([0], [0]))  # (K,2)
        # A <- A - (r/||r||^2) * dot
        denom_b = denom.to(device=b.device, dtype=b.dtype)
        b.sub_((r.to(dtype=b.dtype, device=b.device) / denom_b).view(R, 1, 1) * dot.view(1, -1, 2))
    except Exception:
        # Constraints must never crash inference.
        return


@torch.no_grad()
def _apply_dd_graph_re_constraints_inplace(state: LocateState) -> None:
    """
    Enforce identifiability constraints for dd_graph_re by centering b per component.

    dd_graph_re enters the likelihood via (b_i - b_j). The per-component mean is a
    gauge mode that can trade off with ΔT. We remove that mode by zero-centering
    b within each connected component.
    """
    try:
        if not bool(state.params.get("_dd_graph_re_enabled", False)):
            return
        b_p = getattr(state, "dd_graph_re_b_p", None)
        b_s = getattr(state, "dd_graph_re_b_s", None)
        if not isinstance(b_p, torch.Tensor) and not isinstance(b_s, torch.Tensor):
            return
        cid = getattr(state, "cluster_ids", None)
        if not isinstance(cid, torch.Tensor) or cid.numel() == 0:
            return
        cid_dev = cid
        if b_p is not None and isinstance(b_p, torch.Tensor) and b_p.device != cid.device:
            cid_dev = cid.to(device=b_p.device)
        elif b_s is not None and isinstance(b_s, torch.Tensor) and b_s.device != cid.device:
            cid_dev = cid.to(device=b_s.device)
        K = int(cid_dev.max().item()) + 1 if cid_dev.numel() > 0 else 0
        if K <= 0:
            return
        counts = torch.bincount(cid_dev, minlength=K).clamp_min(1).to(dtype=torch.float32)

        def _center_inplace(b: torch.Tensor) -> None:
            if b.numel() == 0:
                return
            sums = torch.zeros((K,), device=b.device, dtype=b.dtype)
            sums.index_add_(0, cid_dev, b)
            means = sums / counts.to(device=b.device, dtype=b.dtype)
            b.sub_(means.index_select(0, cid_dev))

        if isinstance(b_p, torch.Tensor):
            _center_inplace(b_p)
        if isinstance(b_s, torch.Tensor):
            _center_inplace(b_s)
    except Exception:
        return

    # Also apply an identifiability constraint for corr_error:
    # corr_error enters the likelihood only via (b_j - b_i), so the mean of b across events is a gauge mode.
    try:
        if not bool(state.params.get("_corr_error_enabled", False)):
            return
        b2 = getattr(state, "corr_error_b", None)
        if not isinstance(b2, torch.Tensor) or b2.ndim != 3 or int(b2.shape[2]) != 2:
            return
        # Remove event-mean per (rank, phase): b <- b - mean_events(b)
        mu2 = b2.mean(dim=0, keepdim=True)
        b2.sub_(mu2)
    except Exception:
        return


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
        print(f"Warning: could not attach DD preconditioner tensor: {exc}")


def _current_noise_scales(state: LocateState) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (σ_p, σ_s) as tensors on the correct device."""
    assert state.scale_theta is not None, "Noise scales not initialized"
    return state.scale_theta[0], state.scale_theta[1]


