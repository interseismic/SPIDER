"""Location pipeline orchestration (warmup + SGLD) for SPIDER.

This module contains the high-level `locate_all` entrypoint and its helper
functions, organized into clear phases. It uses the lower-level modeling
primitives from `spider.core.modeling`.
"""

from typing import Optional, Tuple, List, Dict
import os
import time
import sys

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import collections, math

from spider.io.checkpoint import save_checkpoint, load_checkpoint, clear_checkpoint_files
from spider.io.samples import save_samples_periodic, get_next_sample_count, clear_samples_file
from spider.io.samples import save_map_locations
from spider.core.data import filter_by_pair_station_ratio
from spider.optim.backends import create_sampler_backend
from spider.diagnostics.spatial import run_spatial_diag_end_phase1
from spider.diagnostics.pathcorr import run_pathcorr_diag_end_phase1
from spider.utils.console import info, warn
from spider.utils.wandb_gates import want_wandb_group as _want_wandb_group, wb_add_if_finite as _wb_add_if_finite

# Core runtime state (moved out of this module to avoid locate<->epoch_runner cycles)
from spider.core.state import (
    LocateState,
    _clamp_dX_inplace,
    _attach_dd_preconditioner_metric,
    _current_noise_scales,
)

# Event-centric batching utilities (moved out of this module)
from spider.core.batching import (
    _build_event_to_row_map,
)

# Initial state construction (extracted from this module to reduce locate.py size)
from spider.core.init_state import _build_initial_state

# Phase boundary bundles (to skip Phase 1 when repeatedly sampling)
from spider.io.phase_bundle import save_phase2_bundle, load_phase2_bundle


# Pull modeling primitives we need
from .modeling import (
    posterior_loss,
    compute_residuals,
    compute_residuals_full,
    write_output,
    med_abs_dev_torch,
)


def _sampler_extra_metrics(optimizer: Optional[torch.optim.Optimizer]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if optimizer is None:
        return metrics

    if hasattr(optimizer, "drift_vs_noise_per_dim"):
        try:
            # ratios is (N, 4) or None
            ratios = optimizer.drift_vs_noise_per_dim()  # type: ignore[attr-defined]
            if ratios is not None:
                # Calculate stats for each dimension (0=x, 1=y, 2=z, 3=t)
                dim_names = ["x", "y", "z", "t"]
                for i, name in enumerate(dim_names):
                    dim_data = ratios[:, i]  # (N,)
                    
                    # Basic stats
                    metrics[f"drift_ratio_{name}_mean"] = float(dim_data.mean())
                    metrics[f"drift_ratio_{name}_median"] = float(dim_data.median())
                    
                    # Min/Max (robustness check)
                    metrics[f"drift_ratio_{name}_min"] = float(dim_data.min())
                    metrics[f"drift_ratio_{name}_max"] = float(dim_data.max())

                    # Log-mean (for order-of-magnitude tracking)
                    # Add epsilon to avoid log(0)
                    log_mean = torch.log10(dim_data + 1e-20).mean()
                    metrics[f"drift_ratio_{name}_log10_mean"] = float(log_mean)

        except Exception:
            pass

    return metrics


def _compute_phase_mads(
    state: LocateState, batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    residuals = compute_residuals_full(
        state.II, state.YY, state.X_src, state.dX_src, state.model, batch_size, state.N
    )
    idx_p = torch.nonzero(state.YY[:, 4] < 0.5)
    idx_s = torch.nonzero(state.YY[:, 4] > 0.5)
    mad_p = med_abs_dev_torch(residuals[idx_p])
    mad_s = med_abs_dev_torch(residuals[idx_s])
    return mad_p, mad_s

@torch.no_grad()
def _summarize_nuisance_amplitude(state: LocateState) -> None:
    """
    Print simple amplitude diagnostics for the nuisance field:
    - |alpha| distribution (mean/median/p90/max)
    - nuisance_delta magnitude on a sample of observations (RMS and median |.|) split by phase
    """
    try:
        if not (state.nuisance_enable and state.nuisance_alpha is not None and state.nuisance_k_index is not None):
            return
        # Alpha coefficient stats
        alpha_abs = torch.abs(state.nuisance_alpha).detach().flatten()
        aa = alpha_abs.cpu().numpy()
        if aa.size == 0:
            print("Nuisance: alpha coefficients empty.")
            return
        aa_mean = float(np.mean(aa))
        aa_med = float(np.median(aa))
        aa_p90 = float(np.quantile(aa, 0.90))
        aa_max = float(np.max(aa))
        # Nuisance delta over a capped sample of observations
        N = int(state.N)
        sample_cap = int(state.params.get("nuisance_summary_max_obs", 200000))
        B = min(N, max(sample_cap, 1))
        # Use existing batch size to compute deltas
        bs = max(int(state.batch_size_sgld or state.batch_size_warmup or 10000), 1)
        # Compute over first B rows (contiguous) for simplicity
        delta_vals = []
        # No nuisance field; skip per-batch Δb from nuisance
        if delta_vals:
            d_all = torch.cat(delta_vals, dim=0).numpy()
            # Phase masks for the same range [0:B]
            ph = state.YY[:B, 4].detach().cpu().numpy()
            mask_p = ph < 0.5
            mask_s = ph > 0.5
            def _rms(x: np.ndarray) -> float:
                if x.size == 0: return float("nan")
                return float(np.sqrt(np.mean(x * x)))
            def _mad(x: np.ndarray) -> float:
                if x.size == 0: return float("nan")
                return float(np.median(np.abs(x)))
            rms_p = _rms(d_all[mask_p]); rms_s = _rms(d_all[mask_s])
            mad_p = _mad(d_all[mask_p]); mad_s = _mad(d_all[mask_s])
            print(
                f"Nuisance amplitude: |alpha| mean/median/p90/max={aa_mean:.3e}/{aa_med:.3e}/{aa_p90:.3e}/{aa_max:.3e} | "
                f"Δb RMS(P/S)={rms_p:.3e}/{rms_s:.3e} s, median|Δb|(P/S)={mad_p:.3e}/{mad_s:.3e} s (sample={B})"
            )
        else:
            print(
                f"Nuisance amplitude: |alpha| mean/median/p90/max={aa_mean:.3e}/{aa_med:.3e}/{aa_p90:.3e}/{aa_max:.3e} | Δb not computed"
            )
    except Exception as e:
        print(f"Warning: nuisance amplitude summary failed: {e}")

@torch.no_grad()
def _shift_guard_check(state: LocateState, *, context: str = "") -> None:
    """
    If enabled, check per-event shifts against prior_event_std multiplied by a factor.
    If any event exceeds the threshold in any dimension (X,Y,Z,T), print its observations and exit.
    """
    if not bool(state.params.get("shift_guard_enable", False)):
        return
    factor = float(state.params.get("shift_guard_factor", 5.0))
    try:
        prior_std = torch.tensor(state.params["prior_event_std"], dtype=torch.float32, device=state.dX_src.device)
    except Exception:
        print("Shift guard: missing or invalid prior_event_std; skipping guard.")
        return
    if prior_std.numel() != 4:
        print("Shift guard: prior_event_std must have 4 entries; skipping guard.")
        return
    # Absolute per-dimension shifts
    abs_shift = torch.abs(state.dX_src)  # (Ne,4)
    thr = factor * prior_std[None, :]
    exceed_mask = (abs_shift > thr).any(dim=1).detach().cpu().numpy()
    if not exceed_mask.any():
        return
    idxs = np.nonzero(exceed_mask)[0]
    print(f"\nShift guard triggered ({context}): {len(idxs)} event(s) exceeded {factor}×prior_event_std. Printing observations and exiting.")
    # Compute residuals at initial locations (ΔX=0) once
    try:
        bs = max(int(state.batch_size_warmup), 1)
    except Exception:
        bs = 1
    zero_dX = torch.zeros_like(state.dX_src, device=state.dX_src.device)
    res_init = compute_residuals_full(state.II, state.YY, state.X_src, zero_dX, state.model, bs, state.N)
    res_np = res_init.detach().cpu().numpy()
    dt_with_resid = state.dtimes.with_columns(pl.Series("resid_init", res_np))
    # Print details for each offending event
    for idx in idxs:
        try:
            evid = state.origins0["evid"][idx]
        except Exception:
            # Fallback if direct indexing fails
            evid = list(state.origins0["evid"])[idx]
        try:
            print(f"\nEvent evid={evid} (row {idx})")
            shift_vec = abs_shift[idx].detach().cpu().numpy().tolist()
            thr_vec = (prior_std * factor).detach().cpu().numpy().tolist()
            print(f"Shift (dx,dy,dz,dt) = {shift_vec}")
            print(f"Thresholds (dx,dy,dz,dt) = {thr_vec}")
            dt_sub = dt_with_resid.filter(
                (pl.col("evid1").cast(pl.Utf8) == pl.lit(str(evid))) |
                (pl.col("evid2").cast(pl.Utf8) == pl.lit(str(evid)))
            )
            # Summary: number of unique (station, phase) pairs for this event
            try:
                sp_unique = dt_sub.select(["network", "station", "phase"]).unique(maintain_order=True)
                print(f"Unique (station, phase) count: {int(sp_unique.shape[0])}")
            except Exception:
                pass
            # Print a limited sample of rows with all columns (avoid dumping all rows)
            try:
                rows_to_show = min(50, int(dt_sub.shape[0]))
                cols_to_show = int(len(dt_sub.columns))
                with pl.Config(tbl_rows=rows_to_show, tbl_cols=cols_to_show):
                    print(dt_sub)
            except Exception:
                # Fallback to pandas full-column print
                try:
                    import pandas as _pd  # type: ignore
                    _pd.set_option("display.max_columns", None)
                    _pd.set_option("display.width", 0)
                    print(dt_sub.to_pandas().head(50).to_string(index=False))
                except Exception:
                    print(dt_sub)
        except Exception as e:
            print(f"Shift guard: failed to print observations for evid={evid}: {e}")
    sys.exit(2)

def _compute_nuisance_delta_batch(
    state: LocateState,
    i_start: int,
    i_end: int,
    use_epoch_perm: bool = True,
) -> Optional[torch.Tensor]:
    """
    Compute nuisance additive term for a batch of rows: Δb = (φ(ξ_j) - φ(ξ_i)) · α_{k(row)}.
    Returns a 1D tensor shape (B,) on the state's device, or None if nuisance disabled.
    """
    if not (state.nuisance_enable and state.nuisance_alpha is not None and state.nuisance_k_index is not None):
        return None
    # Event positions (Ne, 4) -> take first 3 dims (projected X,Y,Z in km, depth in km)
    Xtot = state.X_src + state.dX_src
    Xev = Xtot[:, 0:3]  # (Ne, 3)
    if state.nuisance_basis == "poly2":
        X, Y, Z = Xev[:, 0], Xev[:, 1], Xev[:, 2]
        Phi = torch.stack([X, Y, Z, X * X, Y * Y, Z * Z, X * Y, X * Z, Y * Z], dim=1)  # (Ne, 9)
    else:
        # default poly1
        Phi = Xev  # (Ne, 3)
    Iab = state.II_epoch[i_start:i_end, :] if (use_epoch_perm and state.II_epoch is not None) else state.II[i_start:i_end, :]
    a = Iab[:, 0]
    b = Iab[:, 1]
    dPhi = Phi.index_select(0, b) - Phi.index_select(0, a)  # (B, M)
    kk_all = state.nuisance_k_index
    kk = kk_all[i_start:i_end]
    Alpha = state.nuisance_alpha.index_select(0, kk)  # (B, M)
    return (dPhi * Alpha).sum(dim=1)  # (B,)


def _latent_map_noise_scales(state: LocateState) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (σ_p, σ_s) to use for the latent-field MAP fit.

    This exists because Phase-1 noise learning can inflate σ and effectively "turn off" the
    latent MAP fit. We allow selecting a different σ just for the MAP step via config.
    """
    mode = str(state.params.get("_latent_field_map_sigma_mode", "current")).strip().lower()
    if mode == "phase_unc":
        σp, σs = state.params.get("phase_unc", [0.05, 0.08])
        return (
            torch.tensor(float(σp), device=state.device, dtype=torch.float32),
            torch.tensor(float(σs), device=state.device, dtype=torch.float32),
        )
    if mode == "override":
        vv = state.params.get("_latent_field_map_sigma_override", None)
        if isinstance(vv, (list, tuple)) and len(vv) == 2:
            return (
                torch.tensor(float(vv[0]), device=state.device, dtype=torch.float32),
                torch.tensor(float(vv[1]), device=state.device, dtype=torch.float32),
            )
        # Fallback (should be prevented by schema validation)
        return _current_noise_scales(state)
    if mode == "clip_current":
        σp, σs = _current_noise_scales(state)
        vv = state.params.get("_latent_field_map_sigma_clip_max", None)
        if isinstance(vv, (list, tuple)) and len(vv) == 2:
            cap_p = float(vv[0])
            cap_s = float(vv[1])
            return (
                torch.minimum(σp, torch.tensor(cap_p, device=state.device, dtype=torch.float32)),
                torch.minimum(σs, torch.tensor(cap_s, device=state.device, dtype=torch.float32)),
            )
        return σp, σs
    # default: use current learned/fixed noise scales
    return _current_noise_scales(state)


@torch.no_grad()
def _latent_field_map_update(
    state: LocateState,
    *,
    epoch_index: int,
    refresh_every: int,
    map_iters: int,
    map_damping: float,
    edge_chunk: int,
    wandb_logger=None,
    wandb_phase: str = "phase1",
    wandb_epoch_for_log: Optional[int] = None,
) -> Dict[str, float]:
    """
    One z-block update step for block-coordinate descent:
      z <- approx argmax p(data | theta_fixed, z) p(z)

    Notes:
    - Independent components: solve x/y/z separately.
    - Midpoint/average correction: dt_lat = 0.5 * amp * (z_i + z_j) · (x_j - x_i)
    - NNGP/Vecchia prior uses the existing `NNGPStruct` (A, d, neigh_pos).
    - Runs a small number of Jacobi-style iterations per station within each connected component.
    """
    metrics_out: Dict[str, float] = {}
    if bool(state.params.get("_latent_field_enabled", False)):
        raise RuntimeError(
            "likelihood.latent_field has been removed from this codebase; delete the config block and use likelihood.shared_event_latent instead"
        )
    return metrics_out

    epoch_index = int(epoch_index)
    refresh_every = int(max(1, int(refresh_every)))
    map_iters = int(max(1, map_iters))
    map_damping = float(map_damping)
    map_damping = float(min(1.0, max(1e-6, map_damping)))
    edge_chunk = int(max(1024, edge_chunk))
    map_log_every = int(state.params.get("_latent_field_map_log_every_iter", 1))
    map_log_n_edges = int(state.params.get("_latent_field_map_log_n_edges", 200000))
    map_z_clip = float(state.params.get("_latent_field_map_z_clip", 10.0))
    map_log_every = int(max(0, map_log_every))
    map_log_n_edges = int(max(0, map_log_n_edges))
    map_z_clip = float(max(1e-6, map_z_clip))

    device = state.device
    n_events = int(state.X_src.shape[0])
    n_stations = int(getattr(state, "n_stations", 0))
    if state.latent_slowness_p is None or tuple(state.latent_slowness_p.shape) != (n_stations, n_events, 3):  # type: ignore[union-attr]
        state.latent_slowness_p = torch.zeros((n_stations, n_events, 3), device=device, dtype=torch.float32)
    if state.latent_slowness_s is None or tuple(state.latent_slowness_s.shape) != (n_stations, n_events, 3):  # type: ignore[union-attr]
        state.latent_slowness_s = torch.zeros((n_stations, n_events, 3), device=device, dtype=torch.float32)

    # Ensure CPU component lists exist (fixed over time for a given filtered dataset).
    labels = state.cluster_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    Kc = int(labels.max()) + 1 if labels.size > 0 else 0
    if not hasattr(state, "_latent_comp_event_ids"):
        comp_events: List[np.ndarray] = []
        for k in range(Kc):
            ev = np.nonzero(labels == k)[0].astype(np.int64, copy=False)
            ev.sort()
            comp_events.append(ev)
        setattr(state, "_latent_comp_event_ids", comp_events)
    if not hasattr(state, "_latent_comp_edge_idx"):
        if state._II_cpu is None or state._II_cpu.shape[0] != int(state.N):
            II_cpu = state.II.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            II_cpu = state._II_cpu
        e1 = II_cpu[:, 0]
        comp = labels[np.clip(e1, 0, labels.size - 1)]
        order = np.argsort(comp, kind="mergesort")
        comp_sorted = comp[order]
        counts = np.bincount(comp_sorted, minlength=Kc)
        offsets = np.zeros((Kc + 1,), dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        comp_edges: List[np.ndarray] = []
        for k in range(Kc):
            s = int(offsets[k]); e = int(offsets[k + 1])
            comp_edges.append(order[s:e].astype(np.int64, copy=False) if e > s else np.empty((0,), dtype=np.int64))
        setattr(state, "_latent_comp_edge_idx", comp_edges)

    # GP hyperparameters
    nu = float(state.params.get("_latent_field_nu", 2.5))
    ell_ps = state.params.get("_latent_field_ell_km", [1.0, 1.0])
    try:
        ell = float(0.5 * (float(ell_ps[0]) + float(ell_ps[1])))
    except Exception:
        ell = float(ell_ps[0]) if isinstance(ell_ps, list) and ell_ps else 1.0
    neighbor_m = int(state.params.get("_latent_field_neighbor_m", 40))

    # Refresh NNGP structures occasionally (geometry-dependent).
    need_refresh = (
        (state.latent_nngp_struct is None)
        or (int(getattr(state, "latent_last_refresh_epoch", -1)) < 0)
        or ((epoch_index - int(getattr(state, "latent_last_refresh_epoch", -1))) >= refresh_every)
    )
    if need_refresh:
        comp_events = getattr(state, "_latent_comp_event_ids")
        comp_edges = getattr(state, "_latent_comp_edge_idx")
        Xxyz = (state.X_src + state.dX_src).detach()[:, :3].float().cpu().numpy()
        n_comp = min(len(comp_events), len(comp_edges))
        nngp_map: Dict[int, dict] = {}
        for k in range(n_comp):
            ev = comp_events[k]
            if ev.size == 0:
                continue
            edge_idx = comp_edges[k]
            if edge_idx.size == 0:
                continue
            xyz = Xxyz[ev, :]
            try:
                struct = build_nngp_struct(xyz, nu=nu, ell_km=ell, m=neighbor_m)
            except Exception:
                continue
            ev_t = torch.tensor(ev, dtype=torch.int64, device=device).contiguous()      # local index order (sorted by global id)
            order_t = torch.tensor(struct.order, dtype=torch.int64, device=device)     # vecchia pos -> local index
            inv_pos_t = torch.tensor(struct.inv_order_pos, dtype=torch.int64, device=device)  # local index -> vecchia pos

            edge_idx_t = torch.tensor(edge_idx, dtype=torch.int64, device=device)
            IIe = state.II.index_select(0, edge_idx_t)
            e1g = IIe[:, 0].contiguous()
            e2g = IIe[:, 1].contiguous()
            e1_local = torch.searchsorted(ev_t, e1g)
            e2_local = torch.searchsorted(ev_t, e2g)
            pos1 = inv_pos_t.index_select(0, e1_local).contiguous()
            pos2 = inv_pos_t.index_select(0, e2_local).contiguous()

            ph = state.YY.index_select(0, edge_idx_t)[:, 4].contiguous()
            is_p_edge = (ph < 0.5)

            # Prior tensors
            neigh_pos_np = struct.neigh_pos.astype(np.int64, copy=False)
            A_np = struct.A.astype(np.float32, copy=False)
            d_np = struct.d.astype(np.float32, copy=False)
            neigh_pos = torch.tensor(neigh_pos_np, dtype=torch.int64, device=device)
            A = torch.tensor(A_np, dtype=torch.float32, device=device)
            inv_d = torch.tensor(1.0 / np.maximum(d_np, 1e-12), dtype=torch.float32, device=device)

            # Parent scatter indices for prior RHS / alpha.
            valid = (neigh_pos >= 0)
            child_idx = torch.arange(int(neigh_pos.shape[0]), device=device, dtype=torch.int64).unsqueeze(1).expand_as(neigh_pos)
            parent_idx_flat = neigh_pos[valid]
            child_idx_flat = child_idx[valid]
            A_flat = A[valid]
            inv_d_child_flat = inv_d.index_select(0, child_idx_flat)
            # alpha_i = 1/d_i + sum_{child k where i in N(k)} A_{k,i}^2 / d_k
            alpha = inv_d.clone()
            alpha.index_add_(0, parent_idx_flat, (A_flat * A_flat) * inv_d_child_flat)

            nngp_map[int(k)] = {
                "struct": struct,
                "ev_t": ev_t,
                "order_t": order_t,
                "inv_pos_t": inv_pos_t,
                "edge_idx": edge_idx_t,
                "pos1": pos1,
                "pos2": pos2,
                "is_p_edge": is_p_edge,
                "neigh_pos": neigh_pos,
                "A": A,
                "inv_d": inv_d,
                "alpha": alpha,
                "parent_idx_flat": parent_idx_flat,
                "child_idx_flat": child_idx_flat,
                "A_flat": A_flat,
                "inv_d_child_flat": inv_d_child_flat,
            }

        state.latent_nngp_struct = nngp_map  # type: ignore[assignment]
        state.latent_last_refresh_epoch = int(epoch_index)

    if not state.latent_nngp_struct:
        return metrics_out

    # Fixed geometry and residuals at end of Phase 1.
    X_total = (state.X_src + state.dX_src).detach()[:, :3]
    σp, σs = _latent_map_noise_scales(state)
    sigma_p = float(σp.item())
    sigma_s = float(σs.item())
    w_p = float(1.0 / max(sigma_p * sigma_p, 1e-12))
    w_s = float(1.0 / max(sigma_s * sigma_s, 1e-12))
    sl_ps = state.params.get("_latent_field_slowness_amp_s_per_km", [0.0, 0.0])
    amp_p = float(sl_ps[0]) if isinstance(sl_ps, list) and len(sl_ps) > 0 else float(sl_ps)
    amp_s = float(sl_ps[1]) if isinstance(sl_ps, list) and len(sl_ps) > 1 else float(sl_ps)
    min_edges_per_station = int(state.params.get("_latent_field_min_edges_per_station", 0))

    t0 = time.time()
    n_comp_done = 0
    n_sta_done = 0

    def _prior_rhs(z_ord_1d: torch.Tensor, cc: dict) -> torch.Tensor:
        """Compute prior RHS = (A z_nb)/d + sum_parent (A_child,parent/d_child * eps_child)."""
        neigh_pos = cc["neigh_pos"]
        A = cc["A"]
        inv_d = cc["inv_d"]
        # gather neighbors (clamp invalid to 0, mask them out by zeroing A)
        valid = (neigh_pos >= 0)
        neigh_pos0 = neigh_pos.clamp_min(0)
        z_nb = z_ord_1d.index_select(0, neigh_pos0.reshape(-1)).reshape_as(neigh_pos0)
        mu = (A * z_nb * valid.to(A.dtype)).sum(dim=1)
        eps = z_ord_1d - mu
        rhs = mu * inv_d
        # parent contribution
        parent_idx = cc["parent_idx_flat"]
        child_idx = cc["child_idx_flat"]
        A_flat = cc["A_flat"]
        inv_d_child = cc["inv_d_child_flat"]
        vals = (A_flat * inv_d_child) * eps.index_select(0, child_idx)
        rhs.index_add_(0, parent_idx, vals)
        return rhs

    # Precompute per-component cached tensors once so MAP iterations don't repeatedly re-evaluate residuals.
    comp_cache: list[dict] = []
    E_total = 0
    for _, cc in state.latent_nngp_struct.items():  # type: ignore[union-attr]
        ev_t: torch.Tensor = cc["ev_t"]
        edge_idx_t: torch.Tensor = cc["edge_idx"]
        Nc = int(ev_t.numel())
        E = int(edge_idx_t.numel())
        if Nc < 2 or E < 4:
            continue
        # Base residual r0 (no latent correction) for this component.
        r0 = torch.empty((E,), device=device, dtype=torch.float32)
        for i0 in range(0, E, edge_chunk):
            i1 = min(i0 + edge_chunk, E)
            rows = edge_idx_t[i0:i1]
            II_b = state.II.index_select(0, rows)
            YY_b = state.YY.index_select(0, rows)
            rb = compute_residuals(II_b, YY_b, state.X_src, state.dX_src, state.model).detach().to(torch.float32)
            r0[i0:i1] = rb
        # Station id for each edge
        sta_edge = state.row_station_index.index_select(0, edge_idx_t).to(torch.int64)  # type: ignore[union-attr]
        sta_np = sta_edge.detach().cpu().numpy().astype(np.int64, copy=False)
        order_sta = np.argsort(sta_np, kind="mergesort")
        sta_sorted = sta_np[order_sta]
        uniq, start_idx, counts = np.unique(sta_sorted, return_index=True, return_counts=True)
        # Geometry for each edge
        II_comp = state.II.index_select(0, edge_idx_t)
        e1g = II_comp[:, 0].contiguous()
        e2g = II_comp[:, 1].contiguous()
        dx_all = (X_total.index_select(0, e2g) - X_total.index_select(0, e1g)).to(torch.float32)
        comp_cache.append(
            {
                "cc": cc,
                "r0": r0,
                "sta_edge": sta_edge,
                "order_sta": order_sta,
                "uniq": uniq,
                "start_idx": start_idx,
                "counts": counts,
                "e1g": e1g,
                "e2g": e2g,
                "dx_all": dx_all,
                "E": E,
                "log_sel": None,
            }
        )
        E_total += int(E)
        n_comp_done += 1

    # Precompute a fixed logging subsample of edges per component so per-iteration curves are comparable.
    # (Otherwise, sampling new edges each iter can look like a drift even when the latent is converging.)
    if map_log_n_edges > 0 and map_log_every > 0 and E_total > 0:
        for ci, pack in enumerate(comp_cache):
            try:
                E = int(pack.get("E", 0))
                if E <= 0:
                    pack["log_sel"] = None
                    continue
                m = int(max(1, int(map_log_n_edges * float(E) / float(E_total))))
                m = int(min(m, E, map_log_n_edges))
                gen = torch.Generator(device=device)
                gen.manual_seed(int(12345 + ci) & 0x7FFFFFFF)
                pack["log_sel"] = torch.randint(low=0, high=E, size=(m,), device=device, dtype=torch.int64, generator=gen)
            except Exception:
                pack["log_sel"] = None

    # Outer MAP iterations (global passes).
    for it in range(map_iters):
        # Per-iter diagnostic accumulators (estimated on a subsample of edges).
        sum_sq_delta_p = 0.0
        sum_sq_delta_s = 0.0
        sum_sq_resid_p = 0.0
        sum_sq_resid_s = 0.0
        sum_sq_r0_p = 0.0
        sum_sq_r0_s = 0.0
        n_p_tot = 0
        n_s_tot = 0

        # MAP sweep over all components/stations.
        max_stations_per_update = int(state.params.get("_latent_field_max_stations_per_update", 0))
        for ci, pack in enumerate(comp_cache):
            cc = pack["cc"]
            ev_t: torch.Tensor = cc["ev_t"]
            order_t: torch.Tensor = cc["order_t"]
            inv_pos_t: torch.Tensor = cc["inv_pos_t"]
            pos1: torch.Tensor = cc["pos1"]
            pos2: torch.Tensor = cc["pos2"]
            is_p_edge: torch.Tensor = cc["is_p_edge"]
            alpha: torch.Tensor = cc["alpha"]

            r0: torch.Tensor = pack["r0"]
            sta_edge: torch.Tensor = pack["sta_edge"]
            order_sta = pack["order_sta"]
            uniq = pack["uniq"]
            start_idx = pack["start_idx"]
            counts = pack["counts"]
            e1g: torch.Tensor = pack["e1g"]
            e2g: torch.Tensor = pack["e2g"]
            dx_all: torch.Tensor = pack["dx_all"]
            E = int(pack["E"])

            # Iterate MAP updates station-by-station (independent across stations).
            sta_list = uniq.tolist()
            if max_stations_per_update > 0 and len(sta_list) > max_stations_per_update:
                # Deterministic subsample to cap per-update work (scales to many stations).
                rng = np.random.default_rng(int((epoch_index * 1000003 + 17 * ci + 97 * it) & 0x7FFFFFFF))
                sta_list = rng.choice(np.asarray(sta_list, dtype=np.int64), size=int(max_stations_per_update), replace=False).tolist()

            for j, sta_id in enumerate(sta_list):
                c_edges = int(counts[j])
                if min_edges_per_station > 0 and c_edges < min_edges_per_station:
                    continue
                s0 = int(start_idx[j]); s1 = int(s0 + c_edges)
                rel_edges_np = order_sta[s0:s1].astype(np.int64, copy=False)
                rel_edges = torch.tensor(rel_edges_np, dtype=torch.int64, device=device)

                # Split this station's edges by phase once.
                p_mask = is_p_edge.index_select(0, rel_edges)
                rel_p = rel_edges[p_mask]
                rel_s = rel_edges[~p_mask]

                def _pack_edges(rel: torch.Tensor):
                    if int(rel.numel()) <= 0:
                        return None
                    return {
                        "pos1": pos1.index_select(0, rel),
                        "pos2": pos2.index_select(0, rel),
                        "r0": r0.index_select(0, rel),
                        "dx": dx_all.index_select(0, rel),  # [E,3]
                    }

                pack_p = _pack_edges(rel_p)
                pack_s = _pack_edges(rel_s)
                if pack_p is None and pack_s is None:
                    continue

                # Pull current z for this station and component (local-index order), then map to Vecchia order.
                zP_loc3 = state.latent_slowness_p[sta_id].index_select(0, ev_t)  # [Nc,3]
                zS_loc3 = state.latent_slowness_s[sta_id].index_select(0, ev_t)  # [Nc,3]
                zP_ord3 = zP_loc3.index_select(0, order_t).contiguous()
                zS_ord3 = zS_loc3.index_select(0, order_t).contiguous()

                for pck, z_ord3, amp, w in (
                    (pack_p, zP_ord3, amp_p, w_p),
                    (pack_s, zS_ord3, amp_s, w_s),
                ):
                    if pck is None or amp <= 0.0:
                        continue
                    pos1_e = pck["pos1"]
                    pos2_e = pck["pos2"]
                    r_e = pck["r0"]
                    dx_e = pck["dx"]
                    w_t = torch.tensor(w, device=device, dtype=torch.float32)
                    for d in range(3):
                        z = z_ord3[:, d]
                        rhs_prior = _prior_rhs(z, cc)
                        H = torch.zeros((int(ev_t.numel()),), device=device, dtype=torch.float32)
                        g = torch.zeros((int(ev_t.numel()),), device=device, dtype=torch.float32)
                        dx_d = dx_e[:, d]
                        c = (0.5 * float(amp)) * dx_d
                        c2w = (c * c) * w_t
                        # IMPORTANT: condition on the other two components by subtracting their
                        # current contributions from the residual, like the ESS update does.
                        # Otherwise each component tries to explain the full residual and the
                        # iterates can "blow up".
                        other = torch.zeros_like(r_e)
                        for d2 in range(3):
                            if d2 == d:
                                continue
                            z2v = z_ord3[:, d2]
                            z1_other = z2v.index_select(0, pos1_e)
                            z2_other = z2v.index_select(0, pos2_e)
                            other = other + (0.5 * float(amp) * (z1_other + z2_other) * dx_e[:, d2])
                        base_r = (r_e - other).to(torch.float32)
                        crw = (c * base_r) * w_t
                        H.index_add_(0, pos1_e, c2w)
                        H.index_add_(0, pos2_e, c2w)
                        z1 = z.index_select(0, pos1_e)
                        z2 = z.index_select(0, pos2_e)
                        g.index_add_(0, pos1_e, crw - c2w * z2)
                        g.index_add_(0, pos2_e, crw - c2w * z1)
                        denom = (H + alpha).clamp_min(1e-12)
                        z_new = (g + rhs_prior) / denom
                        if map_damping < 1.0:
                            z_new = (1.0 - map_damping) * z + map_damping * z_new
                        # Safety clamp on the *unit* latent to avoid numerical divergence.
                        # (Amplitude in s/km is applied outside; typical unit-GP values are O(1).)
                        z_new = z_new.clamp(min=-map_z_clip, max=map_z_clip)
                        z_ord3[:, d] = z_new

                zP_local_new = zP_ord3.index_select(0, inv_pos_t)
                zS_local_new = zS_ord3.index_select(0, inv_pos_t)
                state.latent_slowness_p[sta_id].index_copy_(0, ev_t, zP_local_new)
                state.latent_slowness_s[sta_id].index_copy_(0, ev_t, zS_local_new)
                n_sta_done += 1

            # Sampled diagnostics for this component (budgeted across comps).
            try:
                if map_log_n_edges > 0 and map_log_every > 0 and ((it + 1) % map_log_every == 0) and E_total > 0:
                    sel = pack.get("log_sel", None)
                    if sel is None:
                        continue
                    sta_b = sta_edge.index_select(0, sel)
                    e1b = e1g.index_select(0, sel)
                    e2b = e2g.index_select(0, sel)
                    dx_b = dx_all.index_select(0, sel)
                    r0_b = r0.index_select(0, sel)
                    is_p_b = is_p_edge.index_select(0, sel)

                    r2 = (r0_b * r0_b).to(torch.float32)
                    if bool(is_p_b.any().item()):
                        sum_sq_r0_p += float(r2[is_p_b].sum().item())
                        n_p_tot += int(is_p_b.sum().item())
                    if bool((~is_p_b).any().item()):
                        sum_sq_r0_s += float(r2[~is_p_b].sum().item())
                        n_s_tot += int((~is_p_b).sum().item())

                    if amp_p > 0.0 and bool(is_p_b.any().item()):
                        dsP1 = state.latent_slowness_p[sta_b, e1b, :] * float(amp_p)
                        dsP2 = state.latent_slowness_p[sta_b, e2b, :] * float(amp_p)
                        dtP = (0.5 * (dsP1 + dsP2) * dx_b).sum(dim=1).to(torch.float32)
                        rrP = (r0_b - dtP).to(torch.float32)
                        sum_sq_delta_p += float((dtP[is_p_b] * dtP[is_p_b]).sum().item())
                        sum_sq_resid_p += float((rrP[is_p_b] * rrP[is_p_b]).sum().item())
                    if amp_s > 0.0 and bool((~is_p_b).any().item()):
                        dsS1 = state.latent_slowness_s[sta_b, e1b, :] * float(amp_s)
                        dsS2 = state.latent_slowness_s[sta_b, e2b, :] * float(amp_s)
                        dtS = (0.5 * (dsS1 + dsS2) * dx_b).sum(dim=1).to(torch.float32)
                        rrS = (r0_b - dtS).to(torch.float32)
                        maskS = (~is_p_b)
                        sum_sq_delta_s += float((dtS[maskS] * dtS[maskS]).sum().item())
                        sum_sq_resid_s += float((rrS[maskS] * rrS[maskS]).sum().item())
            except Exception:
                pass

        # Per-iteration logging/printing (sampled).
        do_log = (map_log_every > 0) and (((it + 1) % map_log_every) == 0) and ((n_p_tot + n_s_tot) > 0)
        if do_log:
            try:
                rms_dp = math.sqrt(sum_sq_delta_p / max(1, n_p_tot)) if n_p_tot > 0 else float("nan")
                rms_ds = math.sqrt(sum_sq_delta_s / max(1, n_s_tot)) if n_s_tot > 0 else float("nan")
                rms_rp = math.sqrt(sum_sq_resid_p / max(1, n_p_tot)) if n_p_tot > 0 else float("nan")
                rms_rs = math.sqrt(sum_sq_resid_s / max(1, n_s_tot)) if n_s_tot > 0 else float("nan")
                rms_r0p = math.sqrt(sum_sq_r0_p / max(1, n_p_tot)) if n_p_tot > 0 else float("nan")
                rms_r0s = math.sqrt(sum_sq_r0_s / max(1, n_s_tot)) if n_s_tot > 0 else float("nan")
                # Standardized residual RMS (use the σ used by the latent MAP update)
                # NOTE: RMS values do NOT "add up" linearly: r = r0 - Δ, so
                #   E[r^2] = E[r0^2] + E[Δ^2] - 2E[r0Δ]
                # The cross-term matters because Δ is chosen to correlate with r0.
                rrms_p = (rms_rp / max(sigma_p, 1e-12)) if math.isfinite(rms_rp) else float("nan")
                rrms_s = (rms_rs / max(sigma_s, 1e-12)) if math.isfinite(rms_rs) else float("nan")

                # Terminal output (requested): show RMS latent correction and RMS residuals each iteration.
                # Keep this line compact since map_log_every_iter can be 1.
                print(
                    f"[latent MAP] iter {it + 1}/{map_iters} "
                    f"rms_delta_p={rms_dp:.6g} rms_delta_s={rms_ds:.6g} "
                    f"rms_resid_p={rms_rp:.6g} rms_resid_s={rms_rs:.6g} "
                    f"rrms_p={rrms_p:.3g} rrms_s={rrms_s:.3g} "
                    f"(sigma_p={sigma_p:.3g} sigma_s={sigma_s:.3g} n_p={int(n_p_tot)} n_s={int(n_s_tot)})",
                    flush=True,
                )

                if wandb_logger is not None:
                    iter_metrics = {
                        "latent/map_iter": float(it + 1),
                        "latent/rms_delta_p": float(rms_dp),
                        "latent/rms_delta_s": float(rms_ds),
                        "latent/rms_resid_p": float(rms_rp),
                        "latent/rms_resid_s": float(rms_rs),
                        "latent/rms_resid_p_over_sigma": float(rrms_p),
                        "latent/rms_resid_s_over_sigma": float(rrms_s),
                        "latent/rms_r0_p": float(rms_r0p),
                        "latent/rms_r0_s": float(rms_r0s),
                        "latent/n_edges_p": float(int(n_p_tot)),
                        "latent/n_edges_s": float(int(n_s_tot)),
                        "latent/map_log_n_edges": float(map_log_n_edges),
                    }
                    ep = int(wandb_epoch_for_log) if wandb_epoch_for_log is not None else int(epoch_index)
                    if str(wandb_phase).strip().lower() == "phase1":
                        wandb_logger.log_phase1_metrics(ep, iter_metrics, global_step=int(state.global_step_count))
                    elif str(wandb_phase).strip().lower() == "phase2":
                        wandb_logger.log_phase2_metrics(ep, iter_metrics, global_step=int(state.global_step_count))
                    elif str(wandb_phase).strip().lower() == "phase3":
                        wandb_logger.log_phase3_metrics(ep, iter_metrics, global_step=int(state.global_step_count))
                    else:
                        wandb_logger.log_phase4_metrics(ep, iter_metrics, global_step=int(state.global_step_count))
                    # Ensure monotonically increasing W&B step.
                    state.global_step_count += 1
            except Exception:
                pass

    metrics_out.update(
        {
            "latent/map_ran": 1.0,
            "latent/map_iters": float(map_iters),
            "latent/map_damping": float(map_damping),
            "latent/map_comps": float(n_comp_done),
            "latent/map_stations_updated": float(n_sta_done),
            "latent/map_time_s": float(time.time() - t0),
            "latent/map_refresh": float(1.0 if need_refresh else 0.0),
        }
    )
    return metrics_out


    # (latent_field ESS removed; we use MAP (Jacobi) z-block updates instead)

def _pre_filter_outlier_residuals(state: LocateState) -> None:
    """Optionally drop dtimes with large residuals at initial locations (ΔX=0).

    Controlled by params:
      - residual_filter_enable: bool (default False)
      - residual_filter_method: 'mad' or 'abs' (default 'mad')
      - residual_filter_mad_sigma: float (default 6.0)
      - residual_filter_abs_max: float seconds (default 1.0)
      - residual_filter_log_every: int (unused here, reserved)
    """
    if not bool(state.params.get("residual_filter_enable", False)):
        return

    method = str(state.params.get("residual_filter_method", "mad")).lower()
    sigma = float(state.params.get("residual_filter_mad_sigma", 6.0))
    abs_max = float(state.params.get("residual_filter_abs_max", 1.0))

    print("Residual pre-filter: computing initial residuals for outlier detection…")
    # Use a generous residual batch size to speed up pass
    bs = max(int(state.batch_size_warmup), 1)
    with torch.no_grad():
        # Evaluate residuals at ΔX=0 (i.e., current X_src + 0)
        zero_dX = torch.zeros_like(state.dX_src, device=state.dX_src.device)
        residuals = compute_residuals_full(
            state.II, state.YY, state.X_src, zero_dX, state.model, bs, state.N
        )
        # Build keep mask
        abs_thr = abs_max if (abs_max is not None and float(abs_max) > 0.0) else float("inf")
        if method == "abs":
            thr = abs_thr
            mask_keep = torch.abs(residuals) <= thr
        else:
            # Robust MAD threshold about the residual median, but also respect abs_max:
            # use thr = min(abs_max, mad_sigma * MAD) when abs_max > 0.
            r_med = torch.median(residuals)
            r_mad = med_abs_dev_torch(residuals)
            if not torch.isfinite(r_mad) or float(r_mad.item()) <= 0.0:
                thr = abs_thr
                print(
                    f"Residual pre-filter: MAD not useful (mad={r_mad.item():.3e}); using abs_max={thr:.3f}s"
                )
                mask_keep = torch.abs(residuals - r_med) <= thr
            else:
                mad_thr = sigma * float(r_mad.item())
                thr = min(abs_thr, mad_thr)
                mask_keep = torch.abs(residuals - r_med) <= thr

        keep_count = int(mask_keep.sum().item())
        drop_count = int(state.N - keep_count)
        if drop_count <= 0:
            print("Residual pre-filter: no outliers detected; keeping all rows.")
            return

        # Materialize indices/mask on CPU for polars filtering
        mask_cpu = mask_keep.detach().cpu().numpy()
        if method == "abs":
            thr_msg = f"thr={thr:.3f}s (abs_max)"
        else:
            thr_msg = f"thr={thr:.3f}s (min(abs_max={abs_thr:.3f}s, mad_sigma*mad={mad_thr:.3f}s))"
        print(
            f"Residual pre-filter: dropping {drop_count} / {state.N} dtimes ({thr_msg}, method={method})."
        )

        # Filter tensors
        idx_keep = torch.nonzero(mask_keep).squeeze(-1)
        state.II = state.II.index_select(0, idx_keep).contiguous()
        state.YY = state.YY.index_select(0, idx_keep).contiguous()
        try:
            if state.row_station_index is not None:
                state.row_station_index = state.row_station_index.index_select(0, idx_keep).contiguous()
        except Exception:
            state.row_station_index = None
            state.n_stations = 0
        state.N = int(state.II.shape[0])

        # Filter the polars dtimes (same order as tensors)
        try:
            state.dtimes = state.dtimes.filter(pl.Series(mask_cpu))
        except Exception as e:
            print(f"Warning: could not filter dtimes DataFrame: {e}")
        print(f"Residual pre-filter: remaining dtimes = {state.N}")

        # IMPORTANT: filtering changes row order/length; invalidate any cached batching structures
        # that reference previous row indices (owner buckets, CPU II mirror, per-epoch views).
        try:
            state._II_cpu = None
        except Exception:
            pass
        state.II_epoch = None
        state.YY_epoch = None
        state.row_station_index_epoch = None
        state._perm_epoch = None
        state._bucket_rows_order = None
        state._bucket_offsets = None
        state._bucket_II = None
        state._bucket_YY = None
        state._bucket_station_index = None
        state._bucket_p_counts = None
        state._bucket_nodes_p = None
        state._bucket_u_p = None
        state._bucket_v_p = None
        state._bucket_nodes_s = None
        state._bucket_u_s = None
        state._bucket_v_s = None
        state._bucket_chunks_p = None
        state._bucket_chunks_s = None
        state._bucket_last_epoch = None
        if state.event_batch_enable:
            try:
                _build_event_to_row_map(state)
            except Exception as e:
                print(f"Warning: could not rebuild event->row map after residual pre-filter: {e}")


@torch.no_grad()
def _print_initial_residual_stats(state: LocateState) -> None:
    """Compute and print initial residual statistics (ΔX=0) for P and S separately."""
    if state.N <= 0:
        print("Initial residual stats: no observations.")
        return
    bs = max(int(state.batch_size_warmup), 1)
    # Evaluate residuals at ΔX=0 (i.e., current X_src + 0)
    zero_dX = torch.zeros_like(state.dX_src, device=state.dX_src.device)
    residuals = compute_residuals_full(
        state.II, state.YY, state.X_src, zero_dX, state.model, bs, state.N
    )
    idx_p = torch.nonzero(state.YY[:, 4] < 0.5).squeeze(-1)
    idx_s = torch.nonzero(state.YY[:, 4] > 0.5).squeeze(-1)

    def _stats(mask: torch.Tensor) -> Tuple[float, float, float, int]:
        if mask.numel() == 0:
            return float("nan"), float("nan"), float("nan"), 0
        r = residuals.index_select(0, mask)
        mse = torch.mean(r * r).item()
        mae = torch.mean(torch.abs(r)).item()
        mad = med_abs_dev_torch(r).item()
        return mse, mae, mad, int(mask.numel())

    mse_p, mae_p, mad_p, n_p = _stats(idx_p)
    mse_s, mae_s, mad_s, n_s = _stats(idx_s)
    print(
        f"Initial residual stats (ΔX=0): "
        f"P(n={n_p}) MSE={mse_p:.6e} s^2 MAE={mae_p:.6e} s MAD={mad_p:.6e} s | "
        f"S(n={n_s}) MSE={mse_s:.6e} s^2 MAE={mae_s:.6e} s MAD={mad_s:.6e} s"
    )

from .epoch_runner import _run_epoch

def _format_sampler_status(opt: Optional[torch.optim.Optimizer]) -> str:
    """
    Compact one-liner describing sampler preconditioning + noise settings.
    Works for Adam too (will show precond=none, noise=off).
    """
    if opt is None or not hasattr(opt, "param_groups") or len(getattr(opt, "param_groups", [])) == 0:
        return "precond=unknown noise=unknown"
    g = opt.param_groups[0]
    preconditioning = bool(g.get("preconditioning", False))
    precond_type = str(g.get("preconditioner", "none")).strip().lower()
    # NOTE: we intentionally do not reflect freeze_preconditioner in the console string
    # (it is noisy and not actionable per-epoch). We still read it here so the logic
    # remains easy to extend if needed.
    freeze_precond = bool(g.get("freeze_preconditioner", False))
    if (not preconditioning) or (precond_type in {"none", "false", ""}):
        precond_s = "none"
    else:
        precond_s = precond_type
    add_noise = bool(g.get("add_noise", False))
    noise_scale = float(g.get("noise_scale", 0.0))
    temp = float(g.get("temperature", 1.0))
    noise_on = add_noise and (noise_scale > 0.0) and (temp > 0.0)
    if noise_on:
        noise_s = f"on(scale={noise_scale:g},T={temp:g})"
    else:
        noise_s = "off"
    return f"precond={precond_s} noise={noise_s}"

def _format_epoch_line(*, phase: str, step: int, total: int, metrics: Dict[str, float], opt: Optional[torch.optim.Optimizer], extra: str = "") -> str:
    """
    Standardized per-epoch/iter console line.
    """
    parts = [
        f"{phase}",
        f"{step}/{total}",
        f"L={metrics.get('loss', float('nan')):.4e}",
        f"dx={metrics.get('dx_med_abs', float('nan')):.3e}",
        f"dy={metrics.get('dy_med_abs', float('nan')):.3e}",
        f"dz={metrics.get('dz_med_abs', float('nan')):.3e}",
        f"dr_max={metrics.get('dr_max', float('nan')):.3e}",
        f"dr_90={metrics.get('dr_90', float('nan')):.3e}",
        f"t={metrics.get('epoch_time', float('nan')):.1f}s",
        _format_sampler_status(opt),
    ]
    if extra:
        parts.append(extra)
    return " | ".join(parts)

def _phase1_map_warmup(state: LocateState, start_epoch: int = 0, wandb_logger=None) -> None:
    """Noise-free MAP warmup using Adam on ΔX_src."""
    print(f"Phase 1: MAP (optimizer=adam) | {_format_sampler_status(state.optimizer)}")
    checkpoint_interval = state.params.get("checkpoint_interval", 50)
    # Optional cosine LR scheduler
    use_cosine = bool(state.params.get("phase1_use_cosine", False))
    if use_cosine:
        # T_max: total epochs for cosine; allow override, else use phase1_epochs
        t_max = int(state.params.get("phase1_cosine_T_max", state.params.get("phase1_epochs", 0)))
        eta_min = float(state.params.get("phase1_cosine_eta_min", 0.0))
        scheduler = CosineAnnealingLR(state.optimizer, T_max=max(1, t_max), eta_min=eta_min)
    else:
        scheduler = None
    for epoch in range(start_epoch, state.params["phase1_epochs"]):
        grad_clip_norm = float(state.params.get("grad_clip_norm", 100.0))
        
        metrics = _run_epoch(
            state,
            epoch,
            state.optimizer,
            grad_clip_norm=grad_clip_norm,
            write_map_csv=True,
        )

        # Step LR scheduler at end of epoch
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception as e:
                print(f"Warning: cosine scheduler step failed: {e}")

        
        # Log metrics to wandb if enabled
        if wandb_logger:
            # metrics dict already has dx_mean etc. But logger expects "mad_p" etc?
            # _run_epoch returns simplified metrics dict.
            # Original code computed MADs explicitly.
            # We need to compute MADs if not in _run_epoch.
            # For simplicity, _run_epoch does NOT compute MADs by default to save time.
            # We can add it there or do it here. 
            # The original code did it: if epoch % 10 == 0 or last epoch.
            # We'll replicate that here.
            mad_p_val: Optional[float] = None
            mad_s_val: Optional[float] = None
            if epoch % 10 == 0 or epoch == state.params["phase1_epochs"] - 1:
                 mp, ms = _compute_phase_mads(state, state.batch_size_warmup)
                 mad_p_val, mad_s_val = mp.item(), ms.item()
            
            wandb_metrics = {k:v for k,v in metrics.items()}
            # Noise scales (learned or fixed)
            try:
                σp_now, σs_now = _current_noise_scales(state)
                wandb_metrics.update({
                    "noise/learn_noise_scale": int(bool(state.learn_noise_scale)),
                    "noise/sigma_p": float(σp_now.detach().cpu().item()),
                    "noise/sigma_s": float(σs_now.detach().cpu().item()),
                    "noise/log_sigma_p": float(torch.log(σp_now).detach().cpu().item()),
                    "noise/log_sigma_s": float(torch.log(σs_now).detach().cpu().item()),
                })
            except Exception:
                pass
            # Only include MADs when computed to avoid periodic zero spikes
            if mad_p_val is not None and mad_s_val is not None:
                wandb_metrics.update({
                    "mad_p": mad_p_val,
                    "mad_s": mad_s_val,
                })
            wandb_metrics.update({
                "learning_rate": state.optimizer.param_groups[0]['lr']
            })
            wandb_logger.log_phase1_metrics(epoch, wandb_metrics, global_step=state.global_step_count)
        
        # Report current posterior noise scales instead of MADs
        σp_now, σs_now = _current_noise_scales(state)
        print(_format_epoch_line(
            phase="phase1",
            step=epoch + 1,
            total=int(state.params.get("phase1_epochs", 0)),
            metrics=metrics,
            opt=state.optimizer,
            extra=f"sigma_p={float(σp_now):.4f} sigma_s={float(σs_now):.4f}",
        ))
        _shift_guard_check(state, context=f"phase1 epoch {epoch}")

        # periodic checkpoint for MAP phase
        if checkpoint_interval > 0 and epoch > 0 and (epoch % checkpoint_interval == 0):
            save_checkpoint(
                state.params,
                state.optimizer,
                epoch,
                state.N,
                state.dX_src,
                [],
                state.stats_tensor,
                phase="phase1",
                global_step_count=state.global_step_count,
                noise_log_scale=(state.log_scale_theta.detach() if state.log_scale_theta is not None else None),
                event_precision_matrix=state.event_precision_matrix,
            )

    _finalize_phase1(state)

def _apply_linearization_filter(state: LocateState, target_phase: str) -> None:
    """
    Compute linearization error across the dataset and optionally filter dtimes.
    Target phase is compared against state.params['linearization_error_phase'].
    """
    try:
        lin_enable = bool(state.params.get("linearization_error_enable", False))
        lin_phase = str(state.params.get("linearization_error_phase", "after_phase1")).strip().lower()
        if not lin_enable or lin_phase != target_phase:
            return

        lin_bs = int(state.params.get("linearization_error_batch_size", 50000))
        lin_sample = int(state.params.get("linearization_error_sample_size", 200000))
        lin_log_every = int(state.params.get("linearization_error_log_every_batches", 25))
        lin_max_ratio = state.params.get("linearization_error_max_ratio", None)
        lin_max_ratio_f = float(lin_max_ratio) if lin_max_ratio is not None else None
    except Exception:
        return

    from spider.core.modeling import compute_linearization_error_ratio

    N = int(state.N)
    if N > 0:
        info(
            f"Computing {target_phase} linearization ratio over all rows (N={N}, bs={lin_bs}, max_ratio={lin_max_ratio_f})...",
            section="FILTER",
        )
        # Temporarily disable parameter grads to reduce overhead (we only need input grads)
        req_grad_prev = []
        try:
            for p in state.model.parameters():
                req_grad_prev.append(p.requires_grad)
                p.requires_grad_(False)
        except Exception:
            req_grad_prev = []

        # Streaming stats + optional reservoir sample for percentiles
        import numpy as _np
        rng = _np.random.default_rng(int(state.params.get("linearization_error_seed", 0)))
        sample_buf = []  # type: ignore[var-annotated]
        sample_cap = max(int(lin_sample), 0)
        sample_ratio_buf = []  # type: ignore[var-annotated]
        sample_ratio_cap = max(int(lin_sample), 0)
        seen = 0
        sum_e = 0.0
        max_e = 0.0
        sum_ratio = 0.0
        max_ratio_seen = 0.0
        # Ratio-only filter; schema ensures max_ratio exists and >0 when enabled.
        if not (lin_max_ratio_f is not None and lin_max_ratio_f > 0.0):
            return
        keep_mask = _np.zeros((N,), dtype=_np.bool_)

        t0 = time.time()
        n_batches = (N + lin_bs - 1) // lin_bs
        for b in range(n_batches):
            i0 = b * lin_bs
            i1 = min(i0 + lin_bs, N)
            II_b = state.II[i0:i1]
            YY_b = state.YY[i0:i1]
            # Compute error + ratio on device
            e_b, ratio_b, _, _ = compute_linearization_error_ratio(
                idx=II_b,
                y=YY_b,
                X_src=state.X_src,
                ΔX_src=state.dX_src,
                model=state.model,
            )
            e_cpu = e_b.detach().float().cpu().numpy()
            ratio_cpu = ratio_b.detach().float().cpu().numpy()
            seen += int(e_cpu.size)
            sum_e += float(e_cpu.sum())
            m = float(e_cpu.max()) if e_cpu.size > 0 else 0.0
            if m > max_e:
                max_e = m
            if ratio_cpu.size > 0:
                sum_ratio += float(ratio_cpu.sum())
                mr = float(ratio_cpu.max())
                if mr > max_ratio_seen:
                    max_ratio_seen = mr
            keep_mask[i0:i1] = (ratio_cpu <= lin_max_ratio_f)
            if sample_cap > 0:
                # Reservoir sampling (uniform over all rows)
                for x in e_cpu.tolist():
                    if len(sample_buf) < sample_cap:
                        sample_buf.append(float(x))
                    else:
                        j = int(rng.integers(0, seen))
                        if j < sample_cap:
                            sample_buf[j] = float(x)
            if sample_ratio_cap > 0:
                for x in ratio_cpu.tolist():
                    if len(sample_ratio_buf) < sample_ratio_cap:
                        sample_ratio_buf.append(float(x))
                    else:
                        j = int(rng.integers(0, seen))
                        if j < sample_ratio_cap:
                            sample_ratio_buf[j] = float(x)
            if lin_log_every > 0 and ((b + 1) % lin_log_every == 0 or (b + 1) == n_batches):
                dt_s = time.time() - t0
                rate = float(seen) / max(dt_s, 1e-6)
                extra = f" ratio_mean={sum_ratio/max(seen,1):.3g} ratio_max={max_ratio_seen:.3g}"
                info(
                    f"Linearization error sweep: {b+1}/{n_batches} rows={seen}/{N} "
                    f"mean={sum_e/max(seen,1):.3g} max={max_e:.3g}{extra} rows/s={rate:.3g}",
                    section="FILTER",
                )

        # Restore requires_grad flags
        try:
            if req_grad_prev:
                for p, rg in zip(state.model.parameters(), req_grad_prev):
                    p.requires_grad_(bool(rg))
        except Exception:
            pass

        stats = {
            "mean": float(sum_e / max(seen, 1)),
            "max": float(max_e),
            "n": int(seen),
        }
        if sample_buf:
            s = _np.asarray(sample_buf, dtype=_np.float64)
            try:
                stats.update({
                    "p50": float(_np.quantile(s, 0.50)),
                    "p90": float(_np.quantile(s, 0.90)),
                    "p99": float(_np.quantile(s, 0.99)),
                })
            except Exception:
                pass
        if sample_ratio_buf:
            sr = _np.asarray(sample_ratio_buf, dtype=_np.float64)
            try:
                stats.update({
                    "ratio_mean": float(sum_ratio / max(seen, 1)),
                    "ratio_max": float(max_ratio_seen),
                    "ratio_p50": float(_np.quantile(sr, 0.50)),
                    "ratio_p90": float(_np.quantile(sr, 0.90)),
                    "ratio_p99": float(_np.quantile(sr, 0.99)),
                })
            except Exception:
                pass
        state.params["_linearization_error_stats"] = stats
        info(
            f"Linearization ratio stats ({target_phase}): "
            + ", ".join(f"{k}={v:.3g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in stats.items()),
            section="FILTER",
        )

        before_n = int(state.dtimes.shape[0])
        keep_idx = _np.nonzero(keep_mask)[0].astype(_np.int64, copy=False)
        after_n = int(keep_idx.size)
        info(
            f"Applied linearization_error ({target_phase}) max_ratio={lin_max_ratio_f}; kept {after_n}/{before_n} dtimes.",
            section="FILTER",
        )
        # Polars row selection compatibility (DataFrame.take vs gather vs fallback filter)
        if hasattr(state.dtimes, "take"):
            try:
                state.dtimes = state.dtimes.take(keep_idx.tolist())  # type: ignore[attr-defined]
            except Exception:
                state.dtimes = state.dtimes.take(pl.Series(keep_idx))  # type: ignore[attr-defined]
        elif hasattr(state.dtimes, "gather"):
            state.dtimes = state.dtimes.gather(keep_idx.tolist())  # type: ignore[attr-defined]
        else:
            # Slow fallback: add row index and filter.
            if hasattr(state.dtimes, "with_row_index"):
                tmp = state.dtimes.with_row_index("__row")  # type: ignore[attr-defined]
            else:
                tmp = state.dtimes.with_row_count("__row")  # type: ignore[attr-defined]
            state.dtimes = tmp.filter(pl.col("__row").is_in(pl.Series(keep_idx))).drop("__row")
        state.II = state.II.index_select(0, torch.as_tensor(keep_idx, dtype=torch.int64, device=state.II.device)).contiguous()
        state.YY = state.YY.index_select(0, torch.as_tensor(keep_idx, dtype=torch.int64, device=state.YY.device)).contiguous()
        # Rebuild station indices to ensure they are contiguous 0..n_stations-1 after filtering.
        # This is important for station-dependent latent models (e.g., shared_event_latent) and
        # avoids non-contiguous/gappy sta_idx causing GPU index OOB.
        try:
            sta_keys = (
                state.dtimes.select([pl.col("network"), pl.col("station")])
                .unique(maintain_order=True)
                .with_row_index("sta_idx")
            )
            state.dtimes = state.dtimes.drop("sta_idx") if "sta_idx" in state.dtimes.columns else state.dtimes
            state.dtimes = state.dtimes.join(sta_keys, on=["network", "station"], how="left")
            # If network/station contains nulls, Polars join may produce null sta_idx. Guard explicitly.
            if state.dtimes["sta_idx"].null_count() > 0:
                state.dtimes = state.dtimes.with_columns(pl.col("sta_idx").fill_null(-1))
            sta_idx_np = state.dtimes["sta_idx"].to_numpy().astype(np.int64, copy=False)
            if sta_idx_np.size > 0:
                mn = int(sta_idx_np.min())
                mx = int(sta_idx_np.max())
                n_st = int(sta_keys.shape[0])
                if mn < 0 or mx >= n_st:
                    raise ValueError(f"Invalid sta_idx after linearization filter: min={mn} max={mx} n_stations={n_st}")
            state.row_station_index = torch.tensor(sta_idx_np, dtype=torch.int64, device=state.device).contiguous()
            state.n_stations = int(sta_keys.shape[0])
        except Exception:
            # Fallback: keep the filtered row_station_index slice (if it exists) but ensure n_stations is consistent.
            try:
                if state.row_station_index is not None:
                    state.row_station_index = state.row_station_index.index_select(
                        0, torch.as_tensor(keep_idx, dtype=torch.int64, device=state.row_station_index.device)
                    ).contiguous()
                    mx = int(state.row_station_index.max().item()) if state.row_station_index.numel() > 0 else -1
                    state.n_stations = int(mx + 1)
            except Exception:
                state.row_station_index = None
                state.n_stations = 0
        state.N = after_n
        if state.event_batch_enable:
            try:
                _build_event_to_row_map(state)
            except Exception as e:
                print(f"Warning: could not rebuild event->row map after filtering: {e}")
        try:
            state._II_cpu = _np.column_stack([keep_idx, keep_idx]).astype(_np.int64, copy=False) # Placeholder for re-indexing logic if needed? Actually II is already updated.
            # Actually _II_cpu is used for owner buckets. It needs to be correctly mapped from keep_idx if we use owner buckets.
            # Let's just null it out so it gets rebuilt if needed.
            state._II_cpu = None
        except Exception:
            state._II_cpu = None
        state._bucket_rows_order = None
        state._bucket_offsets = None
        state._bucket_II = None
        state._bucket_YY = None
        state._bucket_p_counts = None
        state._bucket_nodes_p = None
        state._bucket_u_p = None
        state._bucket_v_p = None
        state._bucket_nodes_s = None
        state._bucket_u_s = None
        state._bucket_v_s = None
        state._bucket_chunks_p = None
        state._bucket_chunks_s = None
        state._bucket_last_epoch = None

def _finalize_phase1(state: LocateState) -> None:
    if state.params["phase1_epochs"] <= 0:
        return
    X_src1 = (state.X_src + state.dX_src).detach().cpu().numpy()
    origins = write_output(
        state.origins0, X_src1, state.X_src.detach().cpu().numpy(), state.projector
    )
    origins.write_csv(f"{state.params['catalog_outfile']}_MAP.csv")
    try:
        save_map_locations(state.params, state.origins0, state.X_src, state.dX_src, state.projector)
    except Exception as e:
        print(f"Warning: could not write MAP locations to HDF5: {e}")
    try:
        ratio_phase = str(state.params.get("ratio_filter_phase", "before")).strip().lower()
        ratio_thr = float(state.params.get("max_pair_station_ratio", 0.0))
        if ratio_phase == "after_phase1" and ratio_thr > 0.0:
            before_n = int(state.dtimes.shape[0])
            filtered_dt = filter_by_pair_station_ratio(
                state.params, state.dtimes, origins,
                lat_min=state.params["lat_min"], lon_min=state.params["lon_min"]
            )
            after_n = int(filtered_dt.shape[0])
            print(f"Applied max_pair_station_ratio={ratio_thr} after Phase 1; kept {after_n}/{before_n} dtimes.")
            evid_to_row = {row["evid"]: idx for idx, row in enumerate(state.origins0.iter_rows(named=True))}
            e1_idx = np.array([evid_to_row[x] for x in filtered_dt["evid1"]], dtype=np.int64)
            e2_idx = np.array([evid_to_row[x] for x in filtered_dt["evid2"]], dtype=np.int64)
            state.II = torch.tensor(np.column_stack([e1_idx, e2_idx]), dtype=torch.int64, device=state.device).contiguous()
            YY_np = filtered_dt[["dt", "X", "Y", "Z", "phase"]].to_numpy()
            state.YY = torch.tensor(YY_np, dtype=torch.float32, device=state.device).contiguous()
            state.dtimes = filtered_dt
            state.N = after_n
            # Rebuild station indices (aligned with new dtimes row order)
            try:
                sta_keys = (
                    state.dtimes.select([pl.col("network"), pl.col("station")])
                    .unique(maintain_order=True)
                    .with_row_index("sta_idx")
                )
                state.dtimes = state.dtimes.join(sta_keys, on=["network", "station"], how="left")
                if state.dtimes["sta_idx"].null_count() > 0:
                    state.dtimes = state.dtimes.with_columns(pl.col("sta_idx").fill_null(-1))
                sta_idx_np = state.dtimes["sta_idx"].to_numpy().astype(np.int64, copy=False)
                if sta_idx_np.size > 0:
                    mn = int(sta_idx_np.min())
                    mx = int(sta_idx_np.max())
                    n_st = int(sta_keys.shape[0])
                    if mn < 0 or mx >= n_st:
                        raise ValueError(f"Invalid sta_idx after post-phase1 filtering: min={mn} max={mx} n_stations={n_st}")
                state.row_station_index = torch.tensor(sta_idx_np, dtype=torch.int64, device=state.device).contiguous()
                state.n_stations = int(sta_keys.shape[0])
            except Exception:
                state.row_station_index = None
                state.n_stations = 0
            if state.event_batch_enable:
                try:
                    _build_event_to_row_map(state)
                except Exception as e:
                    print(f"Warning: could not rebuild event->row map after filtering: {e}")
            try:
                state._II_cpu = np.column_stack([e1_idx, e2_idx]).astype(np.int64, copy=False)
            except Exception:
                state._II_cpu = None
            state._bucket_rows_order = None
            state._bucket_offsets = None
            state._bucket_II = None
            state._bucket_YY = None
            state._bucket_p_counts = None
            state._bucket_nodes_p = None
            state._bucket_u_p = None
            state._bucket_v_p = None
            state._bucket_nodes_s = None
            state._bucket_u_s = None
            state._bucket_v_s = None
            state._bucket_chunks_p = None
            state._bucket_chunks_s = None
            state._bucket_last_epoch = None
        # Apply linearization error filter if configured for after Phase 1
        _apply_linearization_filter(state, "after_phase1")

        # Run one-shot spatial diagnostic after final filtering
        try:
            run_spatial_diag_end_phase1(state)
        except Exception as e:
            warn(f"Spatial diagnostic failed: {e}", section="DIAG")
        # Path-correlation vs path-similarity diagnostic (station–phase)
        try:
            run_pathcorr_diag_end_phase1(state)
        except Exception as e:
            warn(f"Path-correlation diagnostic failed: {e}", section="DIAG")

        # Optional: estimate a reasonable station-basis length scale ell_km from MAP residual structure.
        # This is useful for tuning model.likelihood.shared_event_latent.station_basis.ell_km.
        try:
            from spider.diagnostics.station_basis_ell import maybe_estimate_station_basis_ell_after_phase1
            maybe_estimate_station_basis_ell_after_phase1(state=state)
        except Exception as e:
            warn(f"Station-basis ell_km estimate failed: {e}", section="DIAG")

        # Optional: estimate a reasonable event-space ell_km for shared_event_latent from MAP residual structure.
        # This is useful for tuning model.likelihood.shared_event_latent.ell_km (the event kernel length scale).
        try:
            from spider.diagnostics.shared_event_latent_event_ell import (
                maybe_estimate_shared_event_latent_event_ell_after_phase1,
            )
            maybe_estimate_shared_event_latent_event_ell_after_phase1(state=state)
        except Exception as e:
            warn(f"Shared-event-latent event ell_km estimate failed: {e}", section="DIAG")

        # Optional: estimate a reasonable shared-event-latent amplitude tau_s (and tau_p) from MAP residuals.
        # This is useful for tuning model.likelihood.shared_event_latent.tau_s.
        try:
            from spider.diagnostics.shared_event_latent_tau import (
                maybe_estimate_shared_event_latent_tau_after_phase1,
            )
            maybe_estimate_shared_event_latent_tau_after_phase1(state=state)
        except Exception as e:
            warn(f"Shared-event-latent tau estimate failed: {e}", section="DIAG")
    except Exception as e:
        warn(f"Post-Phase1 filters failed: {e}", section="FILTER")
    
    save_checkpoint(
        state.params,
        state.optimizer,
        state.params["phase1_epochs"] - 1,
        state.N,
        state.dX_src,
        [],
        state.stats_tensor,
        phase="phase1",
        global_step_count=state.global_step_count,
        noise_log_scale=(state.log_scale_theta.detach() if state.log_scale_theta is not None else None),
        event_precision_matrix=state.event_precision_matrix,
    )


def _setup_sampler(state: LocateState) -> torch.optim.Optimizer:
    backend_name, sampler = create_sampler_backend(state.params, state)
    info(f"Sampler backend={backend_name}", section="SAMP")
    state.sampler_backend = backend_name
    state.sampler = sampler
    _apply_sampler_group_overrides(state, sampler)
    return sampler


def _resume_or_initialize(state: LocateState):
    """Handle checkpoint reset/resume.

    Returns a tuple: (phase, start_epoch, skip_saving_first_epoch, ckpt_data)
    where phase is one of {"phase1","phase2","phase3","phase4"}.
    start_epoch is the next epoch/iter to run within the resumed phase.
    ckpt_data is the raw checkpoint dict if resuming, else None.
    """
    reset_batch_numbers = state.params.get("reset_batch_numbers", False)
    clear_samples_on_reset = state.params.get("clear_samples_on_reset", False)
    # If we're not resetting batch numbers but we are clearing samples, we want to
    # restart Phase 2 from the Phase 1 MAP locations. Prune any checkpoints saved
    # during phases 2–4 so that the latest checkpoint is from Phase 1.
    if (not reset_batch_numbers) and clear_samples_on_reset:
        try:
            from spider.io.checkpoint import prune_checkpoints_after_phase1 as _prune_p1
            deleted, kept = _prune_p1(state.params)
            if deleted > 0:
                info(f"Pruned checkpoints deleted={deleted} kept={kept}", section="CKPT")
        except Exception as e:
            warn(f"Prune of non-phase1 checkpoints failed: {e}", section="CKPT")
    
    if reset_batch_numbers:
        info("Resetting batch numbers to 0 (reset_batch_numbers=True)", section="RUN")
        clear_checkpoint_files(state.params)
        if clear_samples_on_reset:
            clear_samples_file(state.params)
        state.sample_count = 0
        return "phase1", 0, False, None

    ckpt = load_checkpoint(state.params, state.device)
    if ckpt is not None:
        # no-op: SSST removed entirely
        pass
    
    # Check if we should clear samples even when resuming from checkpoint
    if clear_samples_on_reset and not reset_batch_numbers:
        info("Clearing samples file while resuming from checkpoint (clear_samples_on_reset=True)", section="SAMPLES")
        clear_samples_file(state.params)
        
    if ckpt is None:
        # fresh run
        state.sample_count = get_next_sample_count(state.params)
        if state.sample_count > 0:
            info(f"Continuing from batch number {state.sample_count} (existing samples file found)", section="SAMPLES")
        else:
            info("Starting with batch number 0 (no existing samples file)", section="SAMPLES")
        return "phase1", 0, False, None

    # adopt tensors and stats from checkpoint
    state.dX_src = ckpt["ΔX_src"]  # type: ignore[assignment]
    _attach_dd_preconditioner_metric(state)
    # Ensure the optimizer (used in phase1) points at the resumed tensor
    try:
        state.optimizer.param_groups[0]['params'][0] = state.dX_src  # type: ignore[index]
    except Exception as e:
        warn(f"Could not reset optimizer parameter reference: {e}", section="RUN")
    # Adopt noise log-scale if learning is enabled and checkpoint provides it
    if state.learn_noise_scale:
        nls = ckpt.get("noise_log_scale", None)
        if nls is not None:
            try:
                nls_t = nls.to(device=state.device, dtype=torch.float32)
            except Exception:
                nls_t = torch.as_tensor(nls, dtype=torch.float32, device=state.device)
            state.log_scale_theta = torch.nn.Parameter(nls_t.detach().clone())
            try:
                if len(state.optimizer.param_groups[0]['params']) == 1:
                    state.optimizer.param_groups[0]['params'].append(state.log_scale_theta)  # type: ignore[attr-defined]
                else:
                    state.optimizer.param_groups[0]['params'][1] = state.log_scale_theta  # type: ignore[index]
            except Exception as e:
                print(f"Warning: could not reset optimizer noise parameter reference: {e}")
    state.stats_tensor = ckpt.get("stats_tensor", state.stats_tensor)
    # SSST removed entirely (no backward compatibility): do not load or compute SSST from checkpoints.
        
    # Adopt Hierarchical Prior P0
    try:
        epm = ckpt.get("event_precision_matrix", None)
        if epm is not None and state.hierarchical_prior_enable:
            state.event_precision_matrix = epm.to(device=state.device, dtype=torch.float32)
            print("Resumed Hierarchical Event Precision Matrix (P0) from checkpoint.")
    except Exception as e:
        print(f"Warning: could not adopt Hierarchical Prior P0 from checkpoint: {e}")

    # (Laplacian prior removed: ignore any Laplacian fields that might exist in old checkpoints.)

    state.samples = []
    state.global_step_count = ckpt.get("global_step_count", 0)
    state.sample_count = get_next_sample_count(state.params)
    # Clamp any resumed parameters to respect current config bounds
    try:
        _clamp_dX_inplace(state)
    except Exception as e:
        print(f"Warning: could not clamp resumed parameters: {e}")

    # If clearing samples but not resetting batch numbers, start from phase 2
    if clear_samples_on_reset and not reset_batch_numbers:
        print("Starting fresh from phase 2 (as if phase 1 just finished)")
        phase = "phase2"
        start_epoch = 0
    else:
        phase = str(ckpt.get("phase", "phase4"))
        # We save checkpoints at the end of an epoch/iter, so resume from the *next* one.
        try:
            start_epoch = int(ckpt.get("epoch", -1)) + 1
        except Exception:
            start_epoch = 0

    # New schema only: checkpoints must store one of {"phase1","phase2","phase3","phase4"}.
    phase = phase.strip().lower()
    if phase not in {"phase1", "phase2", "phase3", "phase4"}:
        raise ValueError(f"Invalid checkpoint phase label '{phase}'. Expected one of phase1..phase4.")

    # For phase1, we need to load optimizer state here
    if phase == "phase1":
        try:
            state.optimizer.load_state_dict(ckpt.get("optimizer_state_dict", {}))
            # One-time fix: override LR with current JSON so restarts pick up changes
            if "lr_warmup" in state.params:
                for g in state.optimizer.param_groups:
                    g['lr'] = float(state.params.get("lr_warmup", g.get('lr', 0.0)))
        except Exception as e:
            print(f"Warning: could not load optimizer state: {e}")

    # For SGLD phases we will create sampler later and load then
    skip_saving_first_epoch = True
    print(f"Resuming from phase '{phase}', next epoch/iter {start_epoch}")
    return phase, start_epoch, skip_saving_first_epoch, ckpt


def _phase2_preconditioner(
    state: LocateState, start_epoch: int = 0, skip_saving_first_epoch: bool = False, wandb_logger=None
) -> None:
    assert state.sampler is not None
    sampler = state.sampler
    sampler_backend = str(state.params["sampler_backend"]).strip().lower()
    
    # ... existing comments ...

    # Respect user intent: only force RMSProp warmup if preconditioning is enabled
    user_preconditioning = bool(state.params["sampler_preconditioning"])
    user_precond_type = str(state.params["sampler_preconditioner"]).lower()
    
    for g in sampler.param_groups:
        if 'add_noise' in g:
            g['add_noise'] = False
        if 'noise_scale' in g:
            g['noise_scale'] = 0.0

        # AdaptiveSGHMC has its own internal preconditioner (v_hat) and burn-in adaptation.
        # Do NOT clobber its identity/flags with the generic "rmsprop/adam/matrix" Phase-2 logic.
        if sampler_backend == "adaptive_sghmc":
            g["preconditioner"] = "adaptive_sghmc"
            g["preconditioning"] = True
            g["freeze_preconditioner"] = False
            g["is_burnin"] = True
            continue
            
        # include blockdiag_fisher (alias: matrix_ema) as a valid Phase 2 preconditioner
        if user_preconditioning and user_precond_type in {"rmsprop", "adam", "blockdiag_fisher", "matrix_ema"}:
            g['preconditioner'] = user_precond_type
            g['preconditioning'] = True
            g['freeze_preconditioner'] = False
            g['is_burnin'] = True
        else:
            g['preconditioner'] = 'none'
            g['preconditioning'] = False
            g['freeze_preconditioner'] = True

    print(f"Phase 2: drift-only (noise=off) | {_format_sampler_status(state.sampler)}")
    for epoch in range(start_epoch, state.params["phase2_epochs"]):
        metrics = _run_epoch(state, epoch, sampler, noise_scale_factor=0.0)
        
        # --- per-epoch summary (like phase 3 style) ---
        # Control MAD computation frequency to avoid full-dataset passes
        phase2_interval = int(state.params.get("phase2_mads_interval", 0 if state.event_batch_enable else 10))
        mad_p_val: Optional[float] = None
        mad_s_val: Optional[float] = None
        if phase2_interval > 0 and (epoch % phase2_interval == 0 or epoch == state.params["phase2_epochs"] - 1):
            mp, ms = _compute_phase_mads(state, state.batch_size_sgld)
            mad_p_val, mad_s_val = mp.item(), ms.item()

        # Grad noise vs Langevin diagnostic (geometric mean, median, p10, p90)
        gnoise_gm = float('nan')
        gnoise_med = float('nan')
        gnoise_p10 = float('nan')
        gnoise_p90 = float('nan')
        teff_gm = float('nan')
        teff_med = float('nan')
        teff_var_gm = float('nan')
        teff_var_med = float('nan')
        teff_over = float('nan')
        teff_var_over = float('nan')
        try:
            if bool(state.params.get("sgld_log_gnoise", False)) and _want_wandb_group(state.params, "sampler"):
                stats = sampler.grad_vs_noise_stats()  # type: ignore[attr-defined]
                gnoise_gm = float(stats["gm"])
                gnoise_med = float(stats["median"])
                gnoise_p10 = float(stats.get("p10", float("nan")))
                gnoise_p90 = float(stats.get("p90", float("nan")))
        except Exception:
            pass
        try:
            if bool(state.params.get("sgld_log_temperature", False)) and _want_wandb_group(state.params, "sampler") and hasattr(sampler, "temperature_stats"):
                tstats = sampler.temperature_stats()  # type: ignore[attr-defined]
                teff_gm = float(tstats.get("msq_gm", float("nan")))
                teff_med = float(tstats.get("msq_median", float("nan")))
                teff_var_gm = float(tstats.get("var_gm", float("nan")))
                teff_var_med = float(tstats.get("var_median", float("nan")))
                teff_over = float(tstats.get("msq_median_over_target", float("nan")))
                teff_var_over = float(tstats.get("var_median_over_target", float("nan")))
        except Exception:
            pass

        # Log metrics to wandb if enabled
        if wandb_logger and _want_wandb_group(state.params, "core"):
            wandb_metrics = {k:v for k,v in metrics.items()}
            # Noise scales (learned or fixed)
            try:
                if _want_wandb_group(state.params, "noise"):
                    σp_now, σs_now = _current_noise_scales(state)
                    wandb_metrics.update({
                        "noise/learn_noise_scale": int(bool(state.learn_noise_scale)),
                        "noise/sigma_p": float(σp_now.detach().cpu().item()),
                        "noise/sigma_s": float(σs_now.detach().cpu().item()),
                        "noise/log_sigma_p": float(torch.log(σp_now).detach().cpu().item()),
                        "noise/log_sigma_s": float(torch.log(σs_now).detach().cpu().item()),
                    })
            except Exception:
                pass
            if mad_p_val is not None and mad_s_val is not None:
                wandb_metrics.update({
                    "mad_p": mad_p_val,
                    "mad_s": mad_s_val,
                })
            
            # Hierarchical prior stats are logged centrally in epoch_runner; avoid duplicating here.

            lr0 = float(sampler.param_groups[0]['lr'])
            noise_enabled = bool(sampler.param_groups[0].get('add_noise', sampler.param_groups[0].get('noise_scale', 0.0) > 0.0))
            if _want_wandb_group(state.params, "sampler"):
                wandb_metrics.update({
                    "learning_rate": lr0,
                    "noise_enabled": int(noise_enabled),
                })
            # Only log these diagnostics when noise is enabled; otherwise they are undefined / misleading.
            if noise_enabled and _want_wandb_group(state.params, "sampler"):
                if bool(state.params.get("sgld_log_gnoise", False)):
                    _wb_add_if_finite(wandb_metrics, "grad_noise_to_langevin_gm", gnoise_gm)
                    _wb_add_if_finite(wandb_metrics, "grad_noise_to_langevin_med", gnoise_med)
                    _wb_add_if_finite(wandb_metrics, "grad_noise_to_langevin_p10", gnoise_p10)
                    _wb_add_if_finite(wandb_metrics, "grad_noise_to_langevin_p90", gnoise_p90)
                if bool(state.params.get("sgld_log_temperature", False)):
                    _wb_add_if_finite(wandb_metrics, "sghmc_teff_gm", teff_gm)
                    _wb_add_if_finite(wandb_metrics, "sghmc_teff_med", teff_med)
                    _wb_add_if_finite(wandb_metrics, "sghmc_teff_var_gm", teff_var_gm)
                    _wb_add_if_finite(wandb_metrics, "sghmc_teff_var_med", teff_var_med)
                    _wb_add_if_finite(wandb_metrics, "sghmc_teff_over_target", teff_over)
                    _wb_add_if_finite(wandb_metrics, "sghmc_teff_var_over_target", teff_var_over)
            if _want_wandb_group(state.params, "sampler"):
                wandb_metrics.update(_sampler_extra_metrics(sampler))
            wandb_logger.log_phase2_metrics(epoch, wandb_metrics, global_step=state.global_step_count)

        # Report current posterior noise scales instead of MADs
        σp_now, σs_now = _current_noise_scales(state)
        print(_format_epoch_line(
            phase="phase2",
            step=epoch + 1,
            total=int(state.params.get("phase2_epochs", 0)),
            metrics=metrics,
            opt=sampler,
            # Keep console output compact: avoid printing sampler diagnostics every epoch.
            extra=f"sigma_p={float(σp_now):.4f} sigma_s={float(σs_now):.4f}",
        ))
        
        # Hierarchical prior (Wishart) Gibbs update is handled centrally inside `_run_epoch`
        # so it can run consistently in phases 1–4 without duplication.

        _shift_guard_check(state, context=f"phase2 epoch {epoch+1}")

        # periodic checkpointing (no samples written in phase 2)
        checkpoint_interval = int(state.params.get("checkpoint_interval", 50))
        if checkpoint_interval > 0 and epoch > 0 and (epoch % checkpoint_interval == 0) and (not skip_saving_first_epoch):
            save_checkpoint(
                state.params,
                state.sampler,  # type: ignore[arg-type]
                epoch=epoch,
                N=state.N,
                ΔX_src=state.dX_src,
                samples=[],
                stats_tensor=state.stats_tensor,
                phase="phase2",
                global_step_count=state.global_step_count,
                noise_log_scale=(state.log_scale_theta.detach() if state.log_scale_theta is not None else None),
                event_precision_matrix=state.event_precision_matrix,
            )

        if skip_saving_first_epoch:
            skip_saving_first_epoch = False

    # Save checkpoint at end of phase 2
    phase2_last_epoch = int(state.params.get("phase2_epochs", 0)) - 1
    if phase2_last_epoch < 0:
        phase2_last_epoch = 0
    save_checkpoint(
        state.params,
        state.sampler,  # type: ignore[arg-type]
        epoch=phase2_last_epoch,
        N=state.N,
        ΔX_src=state.dX_src,
        samples=[],
        stats_tensor=state.stats_tensor,
        phase="phase2",
        global_step_count=state.global_step_count,
        noise_log_scale=(state.log_scale_theta.detach() if state.log_scale_theta is not None else None),
        event_precision_matrix=state.event_precision_matrix,
    )
    
    # Compute FIM diagnostics and/or install FIM Preconditioner
    try:
        # Check standard config parameter for preconditioner type
        precond_type = str(state.params["sampler_preconditioner"]).lower()
        # Full-dataset FIM workflow is only enabled explicitly via 'fim'.
        # (User-facing preconditioner type 'matrix' was removed; use 'blockdiag_fisher' (alias: 'matrix_ema')
        # for online 4x4 block-diagonal preconditioning.)
        use_fim = (precond_type in {"fim"})
        
        # Also run diagnostics if explicitly requested, even if not using it for sampling
        run_diag = bool(state.params.get("fim_enable_phase2", False))
        
        if run_diag or use_fim:
            from spider.diagnostics.fim import compute_block_fim, analyze_fim_stability, filter_unstable_events
            print("\n--- Fisher Information Matrix Diagnostics (End of Phase 2) ---")
            
            # Compute only diagonal blocks if we just want to filter (much faster)
            # Only compute sparse if we are actually using matrix preconditioner
            # Or if user debug requested
            # NOTE: Current block-diagonal implementation only uses fim_diag.
            # Sparse is only needed if we implement SVRG-2nd-order or similar.
            need_sparse = False 
            
            fim_diag, fim_sparse = compute_block_fim(state, batch_size=4096, return_sparse=need_sparse)
            
            # Run stability analysis on diagonal blocks
            analyze_fim_stability(fim_diag)
            
            # Filter Unstable Events if configured
            # This physically removes them from the state before Phase 3
            filter_thr = float(state.params.get("fim_filter_threshold", 0.0))
            if filter_thr > 0.0:
                n_dropped = filter_unstable_events(state, fim_diag, threshold=filter_thr)
                if n_dropped > 0:
                    # If we dropped events, we must re-compute FIM if we plan to use it for preconditioning!
                    # Because indices have shifted.
                    print("Events dropped. Re-computing FIM for preconditioner...")
                    fim_diag, fim_sparse = compute_block_fim(state, batch_size=4096, return_sparse=need_sparse)
                    
                    # Re-initialize the sampler optimizer completely
                    # This ensures no stale state (momentum, RMSprop) from old parameters exists.
                    print("Re-initializing sampler optimizer due to parameter change...")
                    
                    # _setup_sampler is defined in THIS file, just call it directly.
                    # No need to import.
                    new_sampler = _setup_sampler(state)
                    # We may need to transplant state if we wanted to keep it, but here we explicitly WANT a reset.
                    # However, if Phase 2 had built up useful preconditioner stats (RMSProp), we lose them.
                    # But if we use FIM preconditioner (Matrix), it is injected below anyway.
                    # If using RMSProp, we restart warm-up in Phase 3. This is acceptable.
                    state.sampler = new_sampler
            
            print("----------------------------------------------------------\n")

            # Install FIM as preconditioner if requested
            if use_fim:
                print("Installing FIM-based Block-Diagonal Preconditioner for Phase 3/4...")
                # fim_diag is (N, 4, 4) (Sum over observations)
                
                # Scale FIM by 1/N to get "Average Fisher Information"
                # This aligns the scale of the inverse with the scale of the gradients (if using total gradient)
                # RMSProp effectively scales updates by N.
                # Natural Gradient (FIM) scales updates by 1.
                # To match RMSProp magnitude convention (so 'lr' means similar thing), we scale by N.
                # Scaling M^{-1} by N is equivalent to dividing M by N.
                # M_avg = F_total / N + damping
                
                # Damping logic:
                # We apply damping to M before inversion.
                damping = float(state.params.get("fim_damping", 1e-2))
                I_eye = torch.eye(4, device=state.device, dtype=fim_diag.dtype).unsqueeze(0)
                M = fim_diag + damping * I_eye
                
                try:
                    # M is the curvature (Precision). We need M^{-1} for drift and noise cov.
                    # M = L_M @ L_M.T
                    L_M = torch.linalg.cholesky(M)
                    
                    # We need L_fac such that L_fac @ L_fac.T = M^{-1}
                    # M^{-1} = (L_M @ L_M.T)^{-1} = L_M^{-T} @ L_M^{-1}
                    # Let L_fac = L_M^{-T} (Upper triangular)
                    L_M_inv = torch.linalg.inv(L_M)
                    L_fac = L_M_inv.mT
                    
                    # Also need M^{-1} for drift term
                    # M^{-1} = L_fac @ L_fac.T
                    M_inv = L_fac @ L_fac.mT
                    
                    # No artificial scaling by N. We use the raw FIM inverse.
                    # This is the true Natural Gradient scaling.
                    # Note: You may need a larger LR in config if steps are too small.
                    
                    # Install into sampler state
                    p_dX = state.dX_src
                    # IMPORTANT: Use the sampler instance from state, which might have been updated!
                    # The local variable 'sampler' (from top of function) might be stale if we did _setup_sampler(state).
                    # Always use state.sampler
                    
                    # Ensure state exists
                    if p_dX not in state.sampler.state:
                        state.sampler.state[p_dX] = {}
                    
                    state.sampler.state[p_dX]['matrix_inv'] = M_inv
                    state.sampler.state[p_dX]['matrix_L'] = L_fac
                    
                    # Switch param group to 'matrix' mode
                    found = False
                    for g in state.sampler.param_groups:
                        # Check if p_dX is in this group
                        # Note: 'params' is a list of tensors
                        if any(p is p_dX for p in g['params']):
                            g['preconditioner'] = 'matrix'
                            # Also ensure preconditioning is True so fallback works for other params
                            g['preconditioning'] = True
                            found = True
                    
                    if found:
                        print(f"FIM Preconditioner installed. Damping={damping}")
                    else:
                        print("Warning: dX_src not found in any sampler param group.")

                except Exception as e:
                    print(f"Error computing/installing FIM preconditioner (singular?): {e}")

    except Exception as e:
        print(f"Warning: FIM computation failed: {e}")



def _phase3_noise_ramp(
    state: LocateState, start_epoch: int = 0, skip_saving_first_epoch: bool = False, wandb_logger=None
) -> None:
    assert state.sampler is not None
    sampler = state.sampler
    print(f"Phase 3: noise ramp | {_format_sampler_status(state.sampler)}")
    ramp_len = int(state.params.get("phase3_epochs", 500))

    # Re-verify FIM installation before starting Phase 3
    # If the user enabled FIM, it should be installed after Phase 2.
    # We check if 'matrix_inv' is present in the sampler state.
    has_fim = False
    try:
        p_first = state.dX_src
        if p_first in sampler.state and 'matrix_inv' in sampler.state[p_first]:
            has_fim = True
            print("Verified: FIM Preconditioner is active for Phase 3.")
        else:
            # Check user intent
            precond_type = str(state.params["sampler_preconditioner"]).lower()
            if precond_type in {"matrix", "fim"}:
                print("Warning: FIM requested but not found in sampler state! Re-running FIM computation...")
                # Re-run installation logic (copied from end of Phase 2)
                from spider.diagnostics.fim import compute_block_fim
                fim_diag, _ = compute_block_fim(state, batch_size=4096, return_sparse=False)
                
                damping = float(state.params.get("fim_damping", 1e-2))
                I_eye = torch.eye(4, device=state.device, dtype=fim_diag.dtype).unsqueeze(0)
                M = fim_diag + damping * I_eye
                L_M = torch.linalg.cholesky(M)
                L_M_inv = torch.linalg.inv(L_M)
                L_fac = L_M_inv.mT
                M_inv = L_fac @ L_fac.mT
                
                # Install
                if p_first not in sampler.state: sampler.state[p_first] = {}
                sampler.state[p_first]['matrix_inv'] = M_inv
                sampler.state[p_first]['matrix_L'] = L_fac
                for g in sampler.param_groups:
                    if any(p is p_first for p in g['params']):
                        g['preconditioner'] = 'matrix'
                        g['preconditioning'] = True
                print("FIM Preconditioner installed (late).")
    except Exception as e:
        print(f"Warning checking FIM status: {e}")

    # Set constant learning rate
    # NOTE on lr_mode='per_obs':
    # We interpret lr_sampler as a per-observation knob for pSGLD/SGHMC/AdaptiveSGHMC by applying
    # lr_eff = lr_sampler / N. This must be applied consistently whenever we overwrite sampler LR
    # (Phase 3/4 entry and resume). The backend factory does the same at construction time.
    lr_user = float(state.params["lr_sampler"])
    lr_mode = str(state.params["sampler_lr_mode"]).strip().lower()
    sampler_backend = str(state.params["sampler_backend"]).lower()
    if sampler_backend in {"psgld", "sghmc", "adaptive_sghmc"} and lr_mode == "per_obs":
        base_lr = lr_user / float(max(1, int(state.N)))
    else:
        base_lr = lr_user
    sampler.set_lr(base_lr)
    _apply_sampler_group_overrides(state, sampler)
    
    for g in sampler.param_groups:
        # Phase 3 is burn-in / noise-ramp. We intentionally allow the preconditioner to adapt here
        # for *all* backends (including pSGLD/SGHMC). Freezing the preconditioner too early can
        # lock in poorly-initialized statistics (e.g., v≈0 for RMSProp), which effectively makes
        # G≈1/eps and can cause catastrophic step/noise amplification once Phase 3 injects noise.
        #
        # The user-facing config key is `sampler.freeze_preconditioner_sampling`, and we apply it
        # in Phase 4 (sampling) only.
        if sampler_backend == "adaptive_sghmc":
            g["freeze_preconditioner"] = False
            g["preconditioner"] = "adaptive_sghmc"
            g["preconditioning"] = True
        else:
            g["freeze_preconditioner"] = False
        # Scale-adapted SGHMC needs to adapt its statistics during the burn-in phase
        # regardless of whether the preconditioning is frozen for sampling.
        g['is_burnin'] = True
    if sampler_backend == "adaptive_sghmc":
        print(f"Phase 3: burn-in | {_format_sampler_status(sampler)}")
    else:
        print(f"Phase 3: noise ramp | {_format_sampler_status(sampler)}")

    for t in range(start_epoch, ramp_len):
        # Noise scale ramp: skip for adaptive samplers because they expect full noise during burn-in/adaptation.
        if sampler_backend == "adaptive_sghmc":
            progress = 1.0
        else:
            progress = float(min(1.0, (t + 1) / float(ramp_len)))
        
        metrics = _run_epoch(state, t, sampler, noise_scale_factor=progress)

        # --- per-iteration (ramp step) summary ---
        phase3_interval = int(state.params.get("phase3_mads_interval", 0 if state.event_batch_enable else 10))
        mad_p_val: Optional[float] = None
        mad_s_val: Optional[float] = None
        if phase3_interval > 0 and (t % phase3_interval == 0 or t == ramp_len - 1):
            mp, ms = _compute_phase_mads(state, state.batch_size_sgld)
            mad_p_val, mad_s_val = mp.item(), ms.item()

        # Grad noise vs Langevin diagnostic (geometric mean, median, p10, p90)
        gnoise_gm = float('nan')
        gnoise_med = float('nan')
        gnoise_p10 = float('nan')
        gnoise_p90 = float('nan')
        teff_gm = float('nan')
        teff_med = float('nan')
        teff_var_gm = float('nan')
        teff_var_med = float('nan')
        teff_over = float('nan')
        teff_var_over = float('nan')
        tau_mean = float('nan')
        tau_med = float('nan')
        # Only compute sampler diagnostics if we're actually logging them.
        if wandb_logger and _want_wandb_group(state.params, "sampler"):
            try:
                if hasattr(sampler, "grad_vs_noise_stats"):
                    stats = sampler.grad_vs_noise_stats()  # type: ignore[attr-defined]
                    gnoise_gm = float(stats.get("gm", float("nan")))
                    gnoise_med = float(stats.get("median", float("nan")))
                    gnoise_p10 = float(stats.get("p10", float("nan")))
                    gnoise_p90 = float(stats.get("p90", float("nan")))
            except Exception:
                pass
            try:
                if hasattr(sampler, "temperature_stats"):
                    tstats = sampler.temperature_stats()  # type: ignore[attr-defined]
                    teff_gm = float(tstats.get("msq_gm", float("nan")))
                    teff_med = float(tstats.get("msq_median", float("nan")))
                    teff_var_gm = float(tstats.get("var_gm", float("nan")))
                    teff_var_med = float(tstats.get("var_median", float("nan")))
                    teff_over = float(tstats.get("msq_median_over_target", float("nan")))
                    teff_var_over = float(tstats.get("var_median_over_target", float("nan")))
            except Exception:
                pass
            try:
                if hasattr(sampler, "tau_stats"):
                    taustats = sampler.tau_stats()  # type: ignore[attr-defined]
                    tau_mean = float(taustats.get("mean", float("nan")))
                    tau_med = float(taustats.get("median", float("nan")))
            except Exception:
                pass

        # Log metrics to wandb if enabled
        if wandb_logger and _want_wandb_group(state.params, "core"):
            wandb_metrics = {k:v for k,v in metrics.items()}
            # Noise scales (learned or fixed)
            try:
                if _want_wandb_group(state.params, "noise"):
                    σp_now, σs_now = _current_noise_scales(state)
                    wandb_metrics.update({
                        "noise/learn_noise_scale": int(bool(state.learn_noise_scale)),
                        "noise/sigma_p": float(σp_now.detach().cpu().item()),
                        "noise/sigma_s": float(σs_now.detach().cpu().item()),
                        "noise/log_sigma_p": float(torch.log(σp_now).detach().cpu().item()),
                        "noise/log_sigma_s": float(torch.log(σs_now).detach().cpu().item()),
                    })
            except Exception:
                pass
            if mad_p_val is not None and mad_s_val is not None:
                wandb_metrics.update({
                    "mad_p": mad_p_val,
                    "mad_s": mad_s_val,
                })
            
            # Hierarchical prior stats are logged centrally in epoch_runner; avoid duplicating here.

            lr0 = float(sampler.param_groups[0]['lr'])
            noise_enabled = bool(sampler.param_groups[0].get('add_noise', sampler.param_groups[0].get('noise_scale', 1.0) > 0.0))
            if _want_wandb_group(state.params, "sampler"):
                wandb_metrics.update({
                    "learning_rate": lr0,
                    "noise_enabled": int(noise_enabled),
                    "ramp_progress": (t + 1) / ramp_len,
                })
                if noise_enabled:
                    _wb_add_if_finite(wandb_metrics, "grad_noise_to_langevin_gm", gnoise_gm)
                    _wb_add_if_finite(wandb_metrics, "grad_noise_to_langevin_med", gnoise_med)
                    _wb_add_if_finite(wandb_metrics, "grad_noise_to_langevin_p10", gnoise_p10)
                    _wb_add_if_finite(wandb_metrics, "grad_noise_to_langevin_p90", gnoise_p90)
                    _wb_add_if_finite(wandb_metrics, "teff_gm", teff_gm)
                    _wb_add_if_finite(wandb_metrics, "teff_med", teff_med)
                    _wb_add_if_finite(wandb_metrics, "teff_var_gm", teff_var_gm)
                    _wb_add_if_finite(wandb_metrics, "teff_var_med", teff_var_med)
                    _wb_add_if_finite(wandb_metrics, "teff_over_target", teff_over)
                    _wb_add_if_finite(wandb_metrics, "teff_var_over_target", teff_var_over)
                # tau_* only exists for samplers that expose tau_stats(); skip NaNs.
                _wb_add_if_finite(wandb_metrics, "tau_mean", tau_mean)
                _wb_add_if_finite(wandb_metrics, "tau_med", tau_med)
                wandb_metrics.update(_sampler_extra_metrics(sampler))
            wandb_logger.log_phase3_metrics(t, wandb_metrics, global_step=state.global_step_count)

        # Report current posterior noise scales instead of MADs
        σp_now, σs_now = _current_noise_scales(state)
        print(_format_epoch_line(
            phase="phase3",
            step=t + 1,
            total=int(ramp_len),
            metrics=metrics,
            opt=sampler,
            # Keep console output compact: avoid printing sampler diagnostics every epoch.
            extra=f"sigma_p={float(σp_now):.4f} sigma_s={float(σs_now):.4f} ramp={progress:.3f}",
        ))
        
        # Hierarchical prior (Wishart) Gibbs update is handled centrally inside `_run_epoch`
        # so it can run consistently in phases 1–4 without duplication.

        _shift_guard_check(state, context=f"phase3 iter {t+1}")

        # periodic checkpointing during phase 3 ramp (no samples written)
        checkpoint_interval = int(state.params.get("checkpoint_interval", 50))
        if checkpoint_interval > 0 and t > 0 and (t % checkpoint_interval == 0) and (not skip_saving_first_epoch):
            save_checkpoint(
                state.params,
                state.sampler,  # type: ignore[arg-type]
                epoch=t,
                N=state.N,
                ΔX_src=state.dX_src,
                samples=[],
                stats_tensor=state.stats_tensor,
                phase="phase3",
                global_step_count=state.global_step_count,
                noise_log_scale=(state.log_scale_theta.detach() if state.log_scale_theta is not None else None),
                event_precision_matrix=state.event_precision_matrix,
            )

        if skip_saving_first_epoch:
            skip_saving_first_epoch = False

    # Save checkpoint at end of phase 3
    phase3_last_iter = int(ramp_len) - 1
    if phase3_last_iter < 0:
        phase3_last_iter = 0
    save_checkpoint(
        state.params,
        state.sampler,  # type: ignore[arg-type]
        epoch=phase3_last_iter,
        N=state.N,
        ΔX_src=state.dX_src,
        samples=[],
        stats_tensor=state.stats_tensor,
        phase="phase3",
        global_step_count=state.global_step_count,
        noise_log_scale=(state.log_scale_theta.detach() if state.log_scale_theta is not None else None),
        event_precision_matrix=state.event_precision_matrix,
    )


def _phase4_sampling(
    state: LocateState, start_epoch: int, skip_saving_first_epoch: bool, wandb_logger=None
) -> None:
    assert state.sampler is not None
    sampler = state.sampler
    # Ensure LR is set from config.
    # For lr_mode='per_obs' we apply lr_eff = lr_sampler / N for pSGLD/SGHMC/AdaptiveSGHMC.
    try:
        lr_user = float(state.params["lr_sampler"])
        lr_mode = str(state.params["sampler_lr_mode"]).strip().lower()
        backend = str(state.params["sampler_backend"]).strip().lower()
        if backend in {"psgld", "sghmc", "adaptive_sghmc"} and lr_mode == "per_obs":
            base_lr = lr_user / float(max(1, int(state.N)))
        else:
            base_lr = lr_user
        if hasattr(sampler, "set_lr"):
            sampler.set_lr(base_lr)  # type: ignore[attr-defined]
        else:
            for g in sampler.param_groups:
                g['lr'] = base_lr
    except Exception:
        pass

    _apply_sampler_group_overrides(state, sampler)

    # Phase 4 might be configured as a no-op (epochs=0). Handle cleanly.
    try:
        n_epochs = int(state.params.get("phase4_epochs", 0))
    except Exception:
        n_epochs = 0
    if n_epochs <= 0:
        print("Phase 4: sampling skipped (phase4_epochs=0)")
        return
    if int(start_epoch) >= int(n_epochs):
        print(f"Phase 4: sampling skipped (start_epoch={start_epoch} >= phase4_epochs={n_epochs})")
        return

    freeze_precond = bool(state.params.get("freeze_preconditioner_sampling", True))
    for g in sampler.param_groups:
        g["freeze_preconditioner"] = freeze_precond
        g["is_burnin"] = False
    _apply_sampler_group_overrides(state, sampler)
    print(f"Phase 4: sampling | {_format_sampler_status(sampler)}")

    # Track relative parameter changes over the last N and N2 epochs
    rel_window = int(state.params.get("rel_change_window", 10))
    rel_window2 = int(state.params.get("rel_change_window2", 50))
    param_snapshots = collections.deque(maxlen=max(rel_window, rel_window2) + 1)

    for epoch in range(int(start_epoch), int(n_epochs)):
        metrics = _run_epoch(
            state,
            epoch,
            sampler,
            is_sampling=True,
            noise_scale_factor=1.0,
        )

        # --- per-epoch summary ---
        phase4_interval = int(state.params.get("phase4_mads_interval", 0 if state.event_batch_enable else 10))
        mad_p_val: Optional[float] = None
        mad_s_val: Optional[float] = None
        if phase4_interval > 0 and (epoch % phase4_interval == 0 or epoch == n_epochs - 1):
            mp, ms = _compute_phase_mads(state, state.batch_size_sgld)
            mad_p_val, mad_s_val = mp.item(), ms.item()

        # Grad noise vs Langevin diagnostic (geometric mean, median, p10, p90)
        gnoise_gm = float('nan')
        gnoise_med = float('nan')
        gnoise_p10 = float('nan')
        gnoise_p90 = float('nan')
        teff_gm = float('nan')
        teff_med = float('nan')
        teff_var_gm = float('nan')
        teff_var_med = float('nan')
        teff_over = float('nan')
        teff_var_over = float('nan')
        tau_mean = float('nan')
        tau_med = float('nan')
        # Only compute sampler diagnostics if we're actually logging them.
        if wandb_logger and _want_wandb_group(state.params, "sampler"):
            try:
                if hasattr(sampler, "grad_vs_noise_stats"):
                    stats = sampler.grad_vs_noise_stats()  # type: ignore[attr-defined]
                    gnoise_gm = float(stats.get("gm", float("nan")))
                    gnoise_med = float(stats.get("median", float("nan")))
                    gnoise_p10 = float(stats.get("p10", float("nan")))
                    gnoise_p90 = float(stats.get("p90", float("nan")))
            except Exception:
                pass
            try:
                if hasattr(sampler, "temperature_stats"):
                    tstats = sampler.temperature_stats()  # type: ignore[attr-defined]
                    teff_gm = float(tstats.get("msq_gm", float("nan")))
                    teff_med = float(tstats.get("msq_median", float("nan")))
                    teff_var_gm = float(tstats.get("var_gm", float("nan")))
                    teff_var_med = float(tstats.get("var_median", float("nan")))
                    teff_over = float(tstats.get("msq_median_over_target", float("nan")))
                    teff_var_over = float(tstats.get("var_median_over_target", float("nan")))
            except Exception:
                pass
            try:
                if hasattr(sampler, "tau_stats"):
                    taustats = sampler.tau_stats()  # type: ignore[attr-defined]
                    tau_mean = float(taustats.get("mean", float("nan")))
                    tau_med = float(taustats.get("median", float("nan")))
            except Exception:
                pass

        # Log metrics to wandb if enabled
        if wandb_logger and _want_wandb_group(state.params, "core"):
            wandb_metrics = {k:v for k,v in metrics.items()}
            # Noise scales (learned or fixed)
            try:
                if _want_wandb_group(state.params, "noise"):
                    σp_now, σs_now = _current_noise_scales(state)
                    wandb_metrics.update({
                        "noise/learn_noise_scale": int(bool(state.learn_noise_scale)),
                        "noise/sigma_p": float(σp_now.detach().cpu().item()),
                        "noise/sigma_s": float(σs_now.detach().cpu().item()),
                        "noise/log_sigma_p": float(torch.log(σp_now).detach().cpu().item()),
                        "noise/log_sigma_s": float(torch.log(σs_now).detach().cpu().item()),
                    })
            except Exception:
                pass
            if mad_p_val is not None and mad_s_val is not None:
                wandb_metrics.update({
                    "mad_p": mad_p_val,
                    "mad_s": mad_s_val,
                })
            
            # Hierarchical prior stats are logged centrally in epoch_runner; avoid duplicating here.

            lr0 = float(sampler.param_groups[0]['lr'])
            noise_enabled = bool(sampler.param_groups[0].get('add_noise', sampler.param_groups[0].get('noise_scale', 1.0) > 0.0))
            if _want_wandb_group(state.params, "sampler"):
                wandb_metrics.update({
                    "learning_rate": lr0,
                    "noise_enabled": int(noise_enabled),
                })
                if noise_enabled:
                    _wb_add_if_finite(wandb_metrics, "grad_noise_to_langevin_gm", gnoise_gm)
                    _wb_add_if_finite(wandb_metrics, "grad_noise_to_langevin_med", gnoise_med)
                    _wb_add_if_finite(wandb_metrics, "grad_noise_to_langevin_p10", gnoise_p10)
                    _wb_add_if_finite(wandb_metrics, "grad_noise_to_langevin_p90", gnoise_p90)
                    _wb_add_if_finite(wandb_metrics, "teff_gm", teff_gm)
                    _wb_add_if_finite(wandb_metrics, "teff_med", teff_med)
                    _wb_add_if_finite(wandb_metrics, "teff_var_gm", teff_var_gm)
                    _wb_add_if_finite(wandb_metrics, "teff_var_med", teff_var_med)
                    _wb_add_if_finite(wandb_metrics, "teff_over_target", teff_over)
                    _wb_add_if_finite(wandb_metrics, "teff_var_over_target", teff_var_over)
                # tau_* only exists for samplers that expose tau_stats(); skip NaNs.
                _wb_add_if_finite(wandb_metrics, "tau_mean", tau_mean)
                _wb_add_if_finite(wandb_metrics, "tau_med", tau_med)
                wandb_metrics.update(_sampler_extra_metrics(sampler))
            wandb_logger.log_phase4_metrics(epoch, wandb_metrics, global_step=state.global_step_count)

        # Compact console line
        σp_now, σs_now = _current_noise_scales(state)
        print(_format_epoch_line(
            phase="phase4",
            step=epoch + 1,
            total=int(n_epochs),
            metrics=metrics,
            opt=sampler,
            extra=f"sigma_p={float(σp_now):.4f} sigma_s={float(σs_now):.4f}",
        ))
        _shift_guard_check(state, context=f"phase4 epoch {epoch}")

        # Periodic checkpointing / sample flush
        checkpoint_interval = int(state.params.get("checkpoint_interval", 50))
        if checkpoint_interval > 0 and epoch > 0 and (epoch % checkpoint_interval == 0) and (not skip_saving_first_epoch):
            save_checkpoint(
                state.params,
                sampler,
                epoch,
                state.N,
                state.dX_src,
                state.samples,
                state.stats_tensor,
                phase="phase4",
                global_step_count=state.global_step_count,
                noise_log_scale=(state.log_scale_theta.detach() if state.log_scale_theta is not None else None),
                event_precision_matrix=state.event_precision_matrix,
            )
            state.sample_count = save_samples_periodic(
                state.params, state.origins0, state.X_src,
                state.samples, state.projector, state.sample_count,
                noise_log_scales=state.noise_log_scales
            )
            state.samples = []
            state.noise_log_scales = []

        if skip_saving_first_epoch:
            skip_saving_first_epoch = False

    # Final flush + checkpoint
    state.sample_count = save_samples_periodic(
        state.params, state.origins0, state.X_src, state.samples, state.projector, state.sample_count,
        noise_log_scales=state.noise_log_scales
    )
    last_epoch = int(n_epochs) - 1
    save_checkpoint(
        state.params,
        sampler,
        last_epoch,
        state.N,
        state.dX_src,
        [],
        state.stats_tensor,
        phase="phase4",
        global_step_count=state.global_step_count,
        noise_log_scale=(state.log_scale_theta.detach() if state.log_scale_theta is not None else None),
        event_precision_matrix=state.event_precision_matrix,
    )  # type: ignore[arg-type]
    return


def _apply_sampler_group_overrides(state: "LocateState", sampler: Optional[torch.optim.Optimizer]) -> None:
    """
    Apply per-parameter-group overrides for the uncollapsed shared_event_latent b field.
    This is necessary because some call sites overwrite lr/temperature for *all* groups.
    """
    if sampler is None:
        return
    if not bool(state.params.get("_shared_event_latent_enabled", False)):
        return
    # If the user did not explicitly provide any overrides, do NOT touch group hyperparams.
    # This avoids surprising behavior and ensures global flags like sampler.freeze_preconditioner_sampling
    # apply uniformly to all parameter groups.
    if not bool(state.params.get("_shared_event_latent_sampler_overrides_active", False)):
        return
    try:
        lr_mult = float(state.params.get("_shared_event_latent_lr_mult", 0.05))
        t_mult = float(state.params.get("_shared_event_latent_temperature_mult", 0.25))
        eps_b = float(state.params.get("_shared_event_latent_eps", 1e-3))
        inc_gamma = bool(state.params.get("_shared_event_latent_include_gamma", False))
        freeze_b = bool(state.params.get("_shared_event_latent_freeze_preconditioner_sampling", False))
    except Exception:
        lr_mult, t_mult, eps_b, inc_gamma, freeze_b = 0.05, 0.25, 1e-3, False, False

    for g in sampler.param_groups:
        if str(g.get("group_name", "")).strip().lower() != "shared_event_latent":
            continue
        try:
            # Keep base lr/temperature as whatever caller set, then apply multipliers.
            # We store base values on the group to avoid compounding.
            if "base_lr" not in g:
                g["base_lr"] = float(g.get("lr", 0.0))
            if "base_temperature" not in g:
                g["base_temperature"] = float(g.get("temperature", 1.0))
            g["lr"] = float(g["base_lr"]) * float(lr_mult)
            g["temperature"] = float(g["base_temperature"]) * float(t_mult)
            # Stabilize RMSProp preconditioner/noise amplification
            g["eps"] = float(eps_b)
            # Default to disabling gamma correction for b (less drift / fewer instabilities)
            g["include_gamma"] = bool(inc_gamma)
            # Do not clobber the global freeze flag; if the run is in a frozen phase (Phase 4),
            # keep it frozen even if this group override isn't requesting freezing.
            g["freeze_preconditioner"] = bool(g.get("freeze_preconditioner", False)) or bool(freeze_b)
        except Exception:
            pass
    return


@torch.no_grad()
def _maybe_report_shared_event_latent_inducing_plan(state: "LocateState") -> None:
    """
    Stage-1 diagnostic for an inducing-point GP approximation of shared_event_latent.

    This does NOT change inference. It only reports per-connected-component geometry and a
    simple suggested inducing-point budget based on a target coverage radius r = cover_frac * ell_km
    (default: ell/2), capped by max_inducing_per_component.
    """
    try:
        if not bool(state.params.get("_shared_event_latent_enabled", False)):
            return
        if not bool(state.params.get("_shared_event_latent_inducing_plan_enable", False)):
            return
        ell = float(state.params.get("_shared_event_latent_ell_km", 0.0))
        if not (ell > 0.0):
            return
        cover_frac = float(state.params.get("_shared_event_latent_inducing_plan_cover_frac_of_ell", 0.5))
        if not (cover_frac > 0.0) or (not math.isfinite(cover_frac)):
            cover_frac = 0.5
        r = float(cover_frac) * float(ell)
        min_m = int(state.params.get("_shared_event_latent_inducing_plan_min_inducing_per_component", 1))
        max_m = int(state.params.get("_shared_event_latent_inducing_plan_max_inducing_per_component", 1024))
        top_k = int(state.params.get("_shared_event_latent_inducing_plan_top_k", 10))
        top_k = max(1, top_k)
        min_m = max(1, min_m)
        max_m = max(min_m, max_m)
    except Exception:
        return

    if getattr(state, "cluster_ids", None) is None or getattr(state, "cluster_counts", None) is None:
        return

    try:
        comp = state.cluster_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        counts = state.cluster_counts.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
    except Exception:
        return
    if comp.size == 0 or counts.size == 0:
        return

    n_comp = int(counts.size)
    # MAP coordinates in km
    try:
        X_map = (state.X_src.detach() + state.dX_src.detach())[:, :3].to(torch.float32).cpu().numpy()
    except Exception:
        return
    if X_map.shape[0] != comp.shape[0]:
        return

    x = X_map[:, 0].astype(np.float32, copy=False)
    y = X_map[:, 1].astype(np.float32, copy=False)
    z = X_map[:, 2].astype(np.float32, copy=False)

    # Per-component bounding boxes via indexed ufunc-at (O(N))
    inf = np.float32(np.inf)
    ninf = np.float32(-np.inf)
    min_x = np.full(n_comp, inf, dtype=np.float32); max_x = np.full(n_comp, ninf, dtype=np.float32)
    min_y = np.full(n_comp, inf, dtype=np.float32); max_y = np.full(n_comp, ninf, dtype=np.float32)
    min_z = np.full(n_comp, inf, dtype=np.float32); max_z = np.full(n_comp, ninf, dtype=np.float32)
    np.minimum.at(min_x, comp, x); np.maximum.at(max_x, comp, x)
    np.minimum.at(min_y, comp, y); np.maximum.at(max_y, comp, y)
    np.minimum.at(min_z, comp, z); np.maximum.at(max_z, comp, z)

    dx = (max_x - min_x).astype(np.float32, copy=False)
    dy = (max_y - min_y).astype(np.float32, copy=False)
    dz = (max_z - min_z).astype(np.float32, copy=False)
    diag_xy = np.sqrt(dx * dx + dy * dy).astype(np.float32, copy=False)
    diag_xyz = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32, copy=False)

    # Simple budget heuristic: cover 2D diameter with disks of radius r (roughly)
    # M ~ ceil((D/r)^2), capped. If D <= r, M=1.
    ratio = diag_xy / np.float32(max(r, 1e-6))
    m_suggest = np.where(ratio <= 1.0, 1.0, np.ceil(ratio * ratio)).astype(np.int64)
    m_suggest = np.maximum(m_suggest, int(min_m))
    m_suggest = np.minimum(m_suggest, int(max_m))
    # Cannot exceed component size
    m_suggest = np.minimum(m_suggest, np.maximum(counts, 1))

    # Summaries
    total_events = int(counts.sum())
    total_m = int(m_suggest.sum())
    n_m1 = int(np.sum(m_suggest == 1))
    # Largest components by size
    idx_sorted = np.argsort(-counts)
    idx_top = idx_sorted[: min(int(top_k), int(n_comp))]

    info(
        f"Inducing-plan (shared_event_latent): components={n_comp} events={total_events:,} "
        f"ell_km={ell:g} cover_r={r:g} (= {cover_frac:g}*ell) min_M={min_m} max_M={max_m} "
        f"sum_M={total_m:,} comps_M1={n_m1}",
        section="LIKELIHOOD",
    )
    for j, ci in enumerate(idx_top.tolist()):
        info(
            f"  comp[{j}] id={ci} n={int(counts[ci]):,} "
            f"diag_xy={float(diag_xy[ci]):.3g}km diag_xyz={float(diag_xyz[ci]):.3g}km "
            f"diag_xy/ell={float(diag_xy[ci]/max(ell,1e-6)):.3g} M={int(m_suggest[ci])}",
            section="LIKELIHOOD",
        )

    # Optional CSV output
    out_path = state.params.get("_shared_event_latent_inducing_plan_outfile", None)
    if out_path:
        try:
            # If relative, place next to checkpoint_dir for convenience
            out_path_s = str(out_path)
            if not os.path.isabs(out_path_s):
                base = str(state.params.get("checkpoint_dir", "."))
                out_path_s = os.path.join(base, out_path_s)
            os.makedirs(os.path.dirname(out_path_s) or ".", exist_ok=True)
            import csv
            with open(out_path_s, "w", newline="") as f:
                wtr = csv.writer(f)
                wtr.writerow([
                    "component_id",
                    "n_events",
                    "diag_xy_km",
                    "diag_xyz_km",
                    "ell_km",
                    "cover_r_km",
                    "diag_xy_over_cover_r",
                    "suggested_M",
                ])
                for ci in range(n_comp):
                    if int(counts[ci]) <= 0:
                        continue
                    wtr.writerow([
                        int(ci),
                        int(counts[ci]),
                        float(diag_xy[ci]),
                        float(diag_xyz[ci]),
                        float(ell),
                        float(r),
                        float(diag_xy[ci] / max(r, 1e-6)),
                        int(m_suggest[ci]),
                    ])
            info(f"Wrote inducing-plan CSV: {out_path_s}", section="LIKELIHOOD")
        except Exception as e:
            warn(f"Could not write inducing-plan CSV: {e}", section="LIKELIHOOD")
    return


@torch.no_grad()
def _maybe_select_shared_event_latent_inducing_points(state: "LocateState") -> None:
    """
    Stage-2 diagnostic for an inducing-point approximation of shared_event_latent.

    This selects inducing event indices per connected component using a simple greedy farthest-point
    (k-center) heuristic with a target coverage radius r = cover_frac * ell_km (default ell/2),
    capped by max_inducing_per_component.

    It does NOT change inference. It only writes a .npz so you can inspect the selection.
    """
    try:
        if not bool(state.params.get("_shared_event_latent_enabled", False)):
            return
        if not bool(state.params.get("_shared_event_latent_inducing_plan_enable", False)):
            return
        if not bool(state.params.get("_shared_event_latent_inducing_plan_select", False)):
            return
        ell = float(state.params.get("_shared_event_latent_ell_km", 0.0))
        if not (ell > 0.0):
            return
        cover_frac = float(state.params.get("_shared_event_latent_inducing_plan_cover_frac_of_ell", 0.5))
        if not (cover_frac > 0.0) or (not math.isfinite(cover_frac)):
            cover_frac = 0.5
        r = float(cover_frac) * float(ell)
        min_m = int(state.params.get("_shared_event_latent_inducing_plan_min_inducing_per_component", 1))
        max_m = int(state.params.get("_shared_event_latent_inducing_plan_max_inducing_per_component", 1024))
        top_k = int(state.params.get("_shared_event_latent_inducing_plan_top_k", 10))
        seed_strategy = str(state.params.get("_shared_event_latent_inducing_plan_seed_strategy", "max_degree")).strip().lower()
        use_xyz = bool(state.params.get("_shared_event_latent_inducing_plan_use_xyz", False))
        out_path = state.params.get("_shared_event_latent_inducing_plan_selection_outfile", None)
        if out_path is None or str(out_path).strip() == "":
            out_path = "shared_event_latent_inducing_selection.npz"
        top_k = max(1, top_k)
        min_m = max(1, min_m)
        max_m = max(min_m, max_m)
        if seed_strategy not in {"max_degree", "random"}:
            seed_strategy = "max_degree"
    except Exception:
        return

    if getattr(state, "cluster_ids", None) is None or getattr(state, "cluster_counts", None) is None:
        return

    try:
        comp = state.cluster_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        counts = state.cluster_counts.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
    except Exception:
        return
    if comp.size == 0 or counts.size == 0:
        return
    n_comp = int(counts.size)

    # MAP coordinates in km (XY or XYZ)
    try:
        X_map = (state.X_src.detach() + state.dX_src.detach())[:, :3].to(torch.float32).cpu().numpy()
    except Exception:
        return
    if X_map.shape[0] != comp.shape[0]:
        return
    if use_xyz:
        P = X_map[:, :3].astype(np.float32, copy=False)
        dim_label = "xyz"
    else:
        P = X_map[:, :2].astype(np.float32, copy=False)
        dim_label = "xy"

    # Optional per-event degree for linkage-aware seeding
    deg = None
    try:
        if getattr(state, "dd_event_degree", None) is not None:
            deg = state.dd_event_degree.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
            if deg.shape[0] != comp.shape[0]:
                deg = None
    except Exception:
        deg = None

    rng = np.random.default_rng(int(state.params.get("runtime_seed", 0)))
    t0 = time.time()

    # Select inducing points for each component
    comp_ids = np.arange(n_comp, dtype=np.int64)
    # Only components with positive counts
    valid = counts > 0
    comp_ids = comp_ids[valid]

    # Process larger components first for logging
    comp_ids = comp_ids[np.argsort(-counts[comp_ids])]

    inducing_idx_all: list[np.ndarray] = []
    inducing_off = [0]
    comp_out = []
    m_out = []
    cov_out = []
    for ci in comp_ids.tolist():
        n = int(counts[ci])
        if n <= 0:
            continue
        idxs = np.flatnonzero(comp == int(ci)).astype(np.int64, copy=False)
        if idxs.size == 0:
            continue
        m_cap = int(min(max_m, int(idxs.size)))
        # Seed
        if seed_strategy == "max_degree" and deg is not None:
            seed_local = int(np.argmax(deg[idxs]))
        else:
            seed_local = int(rng.integers(0, int(idxs.size)))

        Pc = P[idxs]  # (n, d)
        sel_local = [seed_local]
        # min squared distance to any selected point
        d0 = Pc - Pc[seed_local]
        min_d2 = (d0 * d0).sum(axis=1).astype(np.float32, copy=False)
        # Greedy farthest-point until covered or cap reached
        while len(sel_local) < m_cap:
            max_d2 = float(min_d2.max()) if min_d2.size > 0 else 0.0
            if math.sqrt(max_d2) <= r:
                break
            j = int(np.argmax(min_d2))
            if j in sel_local:
                # Shouldn't happen, but guard against degenerate zero distances
                break
            sel_local.append(j)
            dj = Pc - Pc[j]
            d2 = (dj * dj).sum(axis=1).astype(np.float32, copy=False)
            min_d2 = np.minimum(min_d2, d2)

        max_dist = math.sqrt(float(min_d2.max())) if min_d2.size > 0 else 0.0
        sel_global = idxs[np.asarray(sel_local, dtype=np.int64)]
        inducing_idx_all.append(sel_global.astype(np.int64, copy=False))
        inducing_off.append(int(inducing_off[-1] + int(sel_global.size)))
        comp_out.append(int(ci))
        m_out.append(int(sel_global.size))
        cov_out.append(float(max_dist))

    if not inducing_idx_all:
        warn("Inducing selection produced no points; skipping write.", section="LIKELIHOOD")
        return

    inducing_idx = np.concatenate(inducing_idx_all, axis=0).astype(np.int64, copy=False)
    offsets = np.asarray(inducing_off, dtype=np.int64)
    comp_out_a = np.asarray(comp_out, dtype=np.int64)
    m_out_a = np.asarray(m_out, dtype=np.int64)
    cov_out_a = np.asarray(cov_out, dtype=np.float32)

    dt_s = time.time() - t0
    info(
        f"Inducing selection (shared_event_latent): comps={int(comp_out_a.size)} "
        f"sum_M={int(inducing_idx.size):,} ell_km={ell:g} cover_r={r:g} dims={dim_label} "
        f"seed={seed_strategy} dt={dt_s:.1f}s",
        section="LIKELIHOOD",
    )
    # Log top-k by component size
    order = np.argsort(-counts[comp_out_a])
    for j in range(int(min(int(top_k), int(order.size)))):
        ci = int(comp_out_a[order[j]])
        info(
            f"  comp[{j}] id={ci} n={int(counts[ci]):,} M={int(m_out_a[order[j]])} "
            f"max_dist_to_inducing={float(cov_out_a[order[j]]):.3g}km "
            f"(target_r={r:.3g}km)",
            section="LIKELIHOOD",
        )

    # Write NPZ (relative -> checkpoint_dir)
    try:
        out_path_s = str(out_path)
        if not os.path.isabs(out_path_s):
            base = str(state.params.get("checkpoint_dir", "."))
            out_path_s = os.path.join(base, out_path_s)
        os.makedirs(os.path.dirname(out_path_s) or ".", exist_ok=True)
        np.savez_compressed(
            out_path_s,
            component_id=comp_out_a,
            component_n_events=counts[comp_out_a].astype(np.int64, copy=False),
            component_offsets=offsets,
            inducing_event_idx=inducing_idx,
            ell_km=np.asarray([float(ell)], dtype=np.float32),
            cover_r_km=np.asarray([float(r)], dtype=np.float32),
            use_xyz=np.asarray([int(bool(use_xyz))], dtype=np.int8),
            seed_strategy=np.asarray([seed_strategy], dtype=object),
            max_dist_to_inducing_km=cov_out_a,
        )
        info(f"Wrote inducing selection NPZ: {out_path_s}", section="LIKELIHOOD")
        # Expose in-memory selection for subsequent stages (diagnostic; not user-facing)
        try:
            state.params["_shared_event_latent_inducing_component_id"] = comp_out_a
            state.params["_shared_event_latent_inducing_component_offsets"] = offsets
            state.params["_shared_event_latent_inducing_event_idx"] = inducing_idx
            state.params["_shared_event_latent_inducing_cover_r_km"] = float(r)
            state.params["_shared_event_latent_inducing_use_xyz_runtime"] = bool(use_xyz)
            state.params["_shared_event_latent_inducing_selection_file_runtime"] = str(out_path_s)
        except Exception:
            pass
    except Exception as e:
        warn(f"Could not write inducing selection NPZ: {e}", section="LIKELIHOOD")
    return


@torch.no_grad()
def _maybe_build_shared_event_latent_inducing_interpolation(state: "LocateState") -> None:
    """
    Stage-3 diagnostic for inducing-point GP approximation of shared_event_latent.

    Given a per-component inducing selection (Stage 2), build a sparse interpolation structure:
      - for each event e, choose its nearest m inducing points within its component
      - store (global inducing index, kernel value) pairs

    This does NOT change inference. It writes an .npz for inspection and later integration.
    """
    try:
        if not bool(state.params.get("_shared_event_latent_enabled", False)):
            return
        if not bool(state.params.get("_shared_event_latent_inducing_plan_enable", False)):
            return
        if not bool(state.params.get("_shared_event_latent_inducing_plan_interpolation_enable", False)):
            return
        ell = float(state.params.get("_shared_event_latent_ell_km", 0.0))
        if not (ell > 0.0):
            return
        m = int(state.params.get("_shared_event_latent_inducing_plan_interpolation_m", 16))
        m = max(1, m)
        store_dist = bool(state.params.get("_shared_event_latent_inducing_plan_interpolation_store_distances", False))
        use_xyz = bool(state.params.get("_shared_event_latent_inducing_plan_use_xyz", False))
        out_path = state.params.get("_shared_event_latent_inducing_plan_interpolation_outfile", None)
        if out_path is None or str(out_path).strip() == "":
            out_path = "shared_event_latent_inducing_interpolation.npz"
    except Exception:
        return

    # Load selection arrays from memory or from file
    comp_ids = state.params.get("_shared_event_latent_inducing_component_id", None)
    comp_off = state.params.get("_shared_event_latent_inducing_component_offsets", None)
    inducing_event_idx = state.params.get("_shared_event_latent_inducing_event_idx", None)
    if comp_ids is None or comp_off is None or inducing_event_idx is None:
        # Try to load from NPZ file
        sel_path = state.params.get("_shared_event_latent_inducing_selection_file_runtime", None)
        if sel_path is None:
            sel_path = state.params.get("_shared_event_latent_inducing_plan_selection_outfile", None)
        if sel_path is None:
            warn("Inducing interpolation requested but no inducing selection is available; enable inducing_plan.select first.", section="LIKELIHOOD")
            return
        try:
            sel_path_s = str(sel_path)
            if not os.path.isabs(sel_path_s):
                base = str(state.params.get("checkpoint_dir", "."))
                sel_path_s = os.path.join(base, sel_path_s)
            data = np.load(sel_path_s, allow_pickle=True)
            comp_ids = data["component_id"]
            comp_off = data["component_offsets"]
            inducing_event_idx = data["inducing_event_idx"]
            # prefer runtime use_xyz from file if present
            if "use_xyz" in data:
                try:
                    use_xyz = bool(int(np.asarray(data["use_xyz"]).reshape(-1)[0]))
                except Exception:
                    pass
        except Exception as e:
            warn(f"Inducing interpolation requested but could not load selection NPZ: {e}", section="LIKELIHOOD")
            return

    try:
        comp_ids = np.asarray(comp_ids, dtype=np.int64)
        comp_off = np.asarray(comp_off, dtype=np.int64)
        inducing_event_idx = np.asarray(inducing_event_idx, dtype=np.int64)
    except Exception:
        return
    if comp_ids.size == 0 or comp_off.size != comp_ids.size + 1 or inducing_event_idx.size == 0:
        return

    # Event -> component mapping
    if getattr(state, "cluster_ids", None) is None:
        return
    try:
        ev_comp = state.cluster_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    except Exception:
        return
    n_events = int(ev_comp.shape[0])

    # MAP coordinates in km (XY or XYZ)
    try:
        X_map = (state.X_src.detach() + state.dX_src.detach())[:, :3].to(torch.float32).cpu().numpy()
    except Exception:
        return
    if X_map.shape[0] != n_events:
        return
    if use_xyz:
        P = X_map[:, :3].astype(np.float32, copy=False)
        dim_label = "xyz"
    else:
        P = X_map[:, :2].astype(np.float32, copy=False)
        dim_label = "xy"

    # Build a mapping from component id -> block index in comp_ids
    n_comp_total = int(max(ev_comp.max() + 1, comp_ids.max() + 1)) if n_events > 0 else int(comp_ids.max() + 1)
    comp_to_block = np.full((n_comp_total,), -1, dtype=np.int64)
    for bi, cid in enumerate(comp_ids.tolist()):
        if cid >= 0 and cid < comp_to_block.shape[0]:
            comp_to_block[int(cid)] = int(bi)

    # Output arrays (fixed m per event; padded with -1/0)
    neigh_idx = np.full((n_events, m), -1, dtype=np.int64)
    neigh_k = np.zeros((n_events, m), dtype=np.float32)
    neigh_d = np.zeros((n_events, m), dtype=np.float32) if store_dist else None

    t0 = time.time()
    use_scipy = True

    for bi, cid in enumerate(comp_ids.tolist()):
        i0 = int(comp_off[bi]); i1 = int(comp_off[bi + 1])
        if i1 <= i0:
            continue
        U_ev = inducing_event_idx[i0:i1]
        M = int(U_ev.size)
        if M <= 0:
            continue
        # All events in this component
        ev_idx = np.flatnonzero(ev_comp == int(cid)).astype(np.int64, copy=False)
        if ev_idx.size == 0:
            continue
        Pc = P[ev_idx]
        Uc = P[U_ev]
        kq = int(min(m, M))
        if kq <= 0:
            continue
        # Query nearest inducing points
        try:
            from scipy.spatial import cKDTree  # type: ignore
            tree = cKDTree(Uc.astype("float64", copy=False))
            try:
                dists, nbrs = tree.query(Pc.astype("float64", copy=False), k=kq, workers=-1)
            except TypeError:
                try:
                    dists, nbrs = tree.query(Pc.astype("float64", copy=False), k=kq, n_jobs=-1)  # type: ignore[call-arg]
                except TypeError:
                    dists, nbrs = tree.query(Pc.astype("float64", copy=False), k=kq)
            # Normalize shapes to (n,kq)
            if kq == 1:
                dists = np.asarray(dists).reshape(-1, 1)
                nbrs = np.asarray(nbrs).reshape(-1, 1)
        except Exception:
            use_scipy = False
            # Fallback: brute force with chunking (can be slow for large components)
            Uc_f = Uc.astype(np.float32, copy=False)
            Pc_f = Pc.astype(np.float32, copy=False)
            nbrs = np.empty((Pc_f.shape[0], kq), dtype=np.int64)
            dists = np.empty((Pc_f.shape[0], kq), dtype=np.float32)
            chunk = 8192
            for s0 in range(0, Pc_f.shape[0], chunk):
                s1 = min(s0 + chunk, Pc_f.shape[0])
                Q = Pc_f[s0:s1]  # (B,d)
                # (B,M) squared distances
                # Use (x-y)^2 = x^2 + y^2 - 2 x y for speed
                q2 = (Q * Q).sum(axis=1, keepdims=True)
                u2 = (Uc_f * Uc_f).sum(axis=1, keepdims=True).T
                d2 = q2 + u2 - 2.0 * (Q @ Uc_f.T)
                d2 = np.maximum(d2, 0.0)
                # partial sort
                part = np.argpartition(d2, kth=kq - 1, axis=1)[:, :kq]
                d2_part = np.take_along_axis(d2, part, axis=1)
                ord2 = np.argsort(d2_part, axis=1)
                part_sorted = np.take_along_axis(part, ord2, axis=1)
                d2_sorted = np.take_along_axis(d2_part, ord2, axis=1)
                nbrs[s0:s1, :] = part_sorted
                dists[s0:s1, :] = np.sqrt(d2_sorted).astype(np.float32, copy=False)

        # Convert to global inducing indices [0..sum_M)
        gidx = (np.asarray(nbrs, dtype=np.int64) + int(i0)).astype(np.int64, copy=False)
        # Kernel values (RBF)
        d_f = np.asarray(dists, dtype=np.float32)
        k_val = np.exp(-0.5 * (d_f / float(ell)) ** 2).astype(np.float32, copy=False)

        neigh_idx[ev_idx, :kq] = gidx
        neigh_k[ev_idx, :kq] = k_val
        if store_dist and neigh_d is not None:
            neigh_d[ev_idx, :kq] = d_f

    dt_s = time.time() - t0
    info(
        f"Inducing interpolation (shared_event_latent): events={n_events:,} m={m} dims={dim_label} "
        f"ell_km={ell:g} backend={'scipy_ckdtree' if use_scipy else 'bruteforce'} dt={dt_s:.1f}s",
        section="LIKELIHOOD",
    )

    # Write NPZ (relative -> checkpoint_dir)
    try:
        out_path_s = str(out_path)
        if not os.path.isabs(out_path_s):
            base = str(state.params.get("checkpoint_dir", "."))
            out_path_s = os.path.join(base, out_path_s)
        os.makedirs(os.path.dirname(out_path_s) or ".", exist_ok=True)
        payload = dict(
            component_id=comp_ids,
            component_offsets=comp_off,
            inducing_event_idx=inducing_event_idx,
            event_component_id=ev_comp.astype(np.int64, copy=False),
            neighbor_inducing_global_idx=neigh_idx,
            neighbor_kernel=neigh_k,
            ell_km=np.asarray([float(ell)], dtype=np.float32),
            use_xyz=np.asarray([int(bool(use_xyz))], dtype=np.int8),
        )
        if store_dist and neigh_d is not None:
            payload["neighbor_dist_km"] = neigh_d
        np.savez_compressed(out_path_s, **payload)
        info(f"Wrote inducing interpolation NPZ: {out_path_s}", section="LIKELIHOOD")
        try:
            state.params["_shared_event_latent_inducing_interpolation_file_runtime"] = str(out_path_s)
        except Exception:
            pass
    except Exception as e:
        warn(f"Could not write inducing interpolation NPZ: {e}", section="LIKELIHOOD")
    return


@torch.no_grad()
def _maybe_init_shared_event_latent(state: "LocateState") -> None:
    """
    Initialize uncollapsed shared-event latent random effects b[s,event,phase] and a fixed event-space
    kernel graph (kNN Laplacian) built from the current MAP (post Phase-1).

    This is intended for small/medium problems where the full latent b is feasible to sample.
    The event graph is fixed after Phase 1 under the assumption event locations won't move much.
    """
    try:
        if not bool(state.params.get("_shared_event_latent_enabled", False)):
            return
    except Exception:
        return

    # Require stable station indices
    if getattr(state, "row_station_index", None) is None or int(getattr(state, "n_stations", 0)) <= 0:
        warn("shared_event_latent enabled but station indices are missing; disabling shared_event_latent.", section="LIKELIHOOD")
        state.params["_shared_event_latent_enabled"] = False
        return

    mode = str(state.params.get("_shared_event_latent_parameterization", "full")).strip().lower()
    if mode not in {"full", "inducing_gp", "graph_gmrf"}:
        mode = "full"

    # If already initialized (e.g., resume), keep existing
    if mode == "inducing_gp":
        fitc_enable = bool(state.params.get("_shared_event_latent_inducing_fitc_enable", False))
        already = (
            getattr(state, "shared_event_latent_b", None) is not None
            and getattr(state, "shared_event_latent_inducing_neighbor_idx", None) is not None
            and getattr(state, "shared_event_latent_inducing_offsets", None) is not None
            and getattr(state, "shared_event_latent_inducing_K_blocks", None) is not None
        )
        if already:
            # If station_basis is enabled, ensure the fixed basis W is present on resume.
            # Note: W is not a Parameter and is not stored in checkpoints, so we must rebuild it.
            try:
                use_sta_basis = bool(state.params.get("_shared_event_latent_station_basis_enabled", False))
            except Exception:
                use_sta_basis = False
            if use_sta_basis:
                try:
                    b_lat = getattr(state, "shared_event_latent_b", None)
                    n_stations = int(getattr(state, "n_stations", 0))
                    if (
                        isinstance(b_lat, torch.Tensor)
                        and b_lat.ndim == 3
                        and int(b_lat.shape[2]) == 2
                        and int(n_stations) > 0
                        and getattr(state, "shared_event_latent_station_basis_W", None) is None
                    ):
                        # We expect basis-rank coefficients: b_lat.shape[0] == R (not n_stations).
                        r_sta = int(b_lat.shape[0])
                        ell_sta = float(state.params.get("_shared_event_latent_station_basis_ell_km", 0.0))
                        jitter_sta = float(state.params.get("_shared_event_latent_station_basis_jitter", 1e-6))
                        method = str(state.params.get("_shared_event_latent_station_basis_method", "eigh_rbf")).strip().lower()
                        if r_sta >= 1 and r_sta <= n_stations and method == "eigh_rbf" and (ell_sta > 0.0):
                            dt = state.dtimes
                            if "sta_idx" in dt.columns and "X" in dt.columns and "Y" in dt.columns:
                                sta_xy = (
                                    dt.select([pl.col("sta_idx"), pl.col("X"), pl.col("Y")])
                                    .unique(subset=["sta_idx"], maintain_order=True)
                                    .sort("sta_idx")
                                )
                                xy_np = sta_xy.select([pl.col("X"), pl.col("Y")]).to_numpy().astype(np.float32, copy=False)
                                XY = torch.from_numpy(xy_np).to(device=state.device, dtype=torch.float32)
                                D_sta = torch.cdist(XY, XY).to(torch.float32)
                                K_sta = torch.exp(-0.5 * (D_sta / float(ell_sta)).square())
                                j = 1e-6 if (not math.isfinite(jitter_sta) or jitter_sta <= 0.0) else float(jitter_sta)
                                K_sta = K_sta + (j * torch.eye(int(K_sta.shape[0]), device=state.device, dtype=K_sta.dtype))
                                evals, evecs = torch.linalg.eigh(K_sta)
                                evals = evals.clamp_min(0.0)
                                evals_r = evals[-r_sta:]
                                evecs_r = evecs[:, -r_sta:]
                                W_sta = evecs_r * torch.sqrt(evals_r).unsqueeze(0)  # (S,R)
                                state.shared_event_latent_station_basis_W = W_sta.to(device=state.device, dtype=torch.float32)
                                state.shared_event_latent_station_basis_r = int(r_sta)
                                state.params["_shared_event_latent_station_basis_r_runtime"] = int(r_sta)
                                info(f"Rebuilt shared_event_latent station_basis W on resume (R={r_sta})", section="LIKELIHOOD")
                except Exception as e:
                    warn(f"Could not rebuild shared_event_latent station_basis on resume: {e}", section="LIKELIHOOD")

            # If FITC is enabled but residual diag wasn't computed (e.g., resume from older checkpoint),
            # compute it now and keep the existing parameter tensor.
            if fitc_enable and getattr(state, "shared_event_latent_inducing_fitc_resid", None) is None:
                try:
                    nei_idx = getattr(state, "shared_event_latent_inducing_neighbor_idx", None)
                    nei_k = getattr(state, "shared_event_latent_inducing_neighbor_k", None)
                    offs_t = getattr(state, "shared_event_latent_inducing_offsets", None)
                    K_blocks = getattr(state, "shared_event_latent_inducing_K_blocks", None)
                    if (
                        isinstance(nei_idx, torch.Tensor)
                        and isinstance(nei_k, torch.Tensor)
                        and isinstance(offs_t, torch.Tensor)
                        and isinstance(K_blocks, list)
                        and K_blocks
                    ):
                        t_fitc0 = time.time()
                        idx_np = nei_idx.detach().cpu().numpy().astype(np.int64, copy=False)
                        k_np = nei_k.detach().cpu().numpy().astype(np.float32, copy=False)
                        offs_np = offs_t.detach().cpu().numpy().astype(np.int64, copy=False)
                        K_np = [
                            (Kb.detach().cpu().numpy().astype(np.float32, copy=False) if isinstance(Kb, torch.Tensor) else None)
                            for Kb in K_blocks
                        ]
                        n_ev = int(idx_np.shape[0])
                        q_diag_np = np.zeros((n_ev,), dtype=np.float32)
                        for e in range(n_ev):
                            idx_row = idx_np[e]
                            k_row = k_np[e]
                            m = idx_row >= 0
                            if not bool(m.any()):
                                q = 0.0
                            else:
                                idxv = idx_row[m]
                                kv = k_row[m]
                                g0 = int(idxv[0])
                                bi = int(np.searchsorted(offs_np[1:], g0, side="right"))
                                if bi < 0 or bi >= len(K_np) or K_np[bi] is None:
                                    q = 0.0
                                else:
                                    i0 = int(offs_np[bi])
                                    loc = (idxv - i0).astype(np.int64, copy=False)
                                    Kb = K_np[bi]
                                    if Kb is None or Kb.size == 0:
                                        q = 0.0
                                    else:
                                        try:
                                            Kmm = Kb[np.ix_(loc, loc)]
                                            sol = np.linalg.solve(Kmm, kv)
                                            q = float(kv.dot(sol))
                                        except Exception:
                                            q = 0.0
                            q_diag_np[e] = float(q)
                        q_diag_np = np.clip(q_diag_np, 0.0, 1.0).astype(np.float32, copy=False)
                        resid_np = np.maximum(0.0, 1.0 - q_diag_np).astype(np.float32, copy=False)
                        state.shared_event_latent_inducing_fitc_q_diag = torch.from_numpy(q_diag_np).to(device=state.device, dtype=torch.float32)
                        state.shared_event_latent_inducing_fitc_resid = torch.from_numpy(resid_np).to(device=state.device, dtype=torch.float32)
                        dt_fitc = time.time() - t_fitc0
                        info(
                            "Computed FITC diag residual (shared_event_latent inducing_gp): "
                            f"q_diag[min/mean/max]={float(q_diag_np.min()):.3g}/{float(q_diag_np.mean()):.3g}/{float(q_diag_np.max()):.3g} "
                            f"resid[min/mean/max]={float(resid_np.min()):.3g}/{float(resid_np.mean()):.3g}/{float(resid_np.max()):.3g} "
                            f"(dt={dt_fitc:.2f}s)",
                            section="LIKELIHOOD",
                        )
                except Exception:
                    pass
            return
    else:
        if (
            getattr(state, "shared_event_latent_b", None) is not None
            and getattr(state, "shared_event_latent_u", None) is not None
            and getattr(state, "shared_event_latent_v", None) is not None
            and getattr(state, "shared_event_latent_w", None) is not None
        ):
            return

    n_events = int(state.X_src.shape[0])
    n_stations = int(getattr(state, "n_stations", 0))
    if n_events <= 1 or n_stations <= 0:
        state.params["_shared_event_latent_enabled"] = False
        return

    ell_km = float(state.params.get("_shared_event_latent_ell_km", 0.0))
    q_diag = float(state.params.get("_shared_event_latent_q_diag", 0.0))
    knn = int(state.params.get("_shared_event_latent_knn", 0))
    if not (ell_km > 0.0):
        warn("shared_event_latent enabled but knn/ell_km invalid; disabling.", section="LIKELIHOOD")
        state.params["_shared_event_latent_enabled"] = False
        return
    knn = min(int(knn), max(1, n_events - 1))

    # --- inducing_gp parameterization ---
    if mode == "inducing_gp":
        # This mode uses Stage-2/3 inducing_plan artifacts (selection + interpolation) to build:
        # - per-event sparse neighbor lists into the concatenated inducing list
        # - per-component GP prior blocks K_UU on inducing coefficients
        dev = state.device
        ell_km = float(state.params.get("_shared_event_latent_ell_km", 0.0))
        if not (ell_km > 0.0):
            warn("shared_event_latent inducing_gp requires ell_km > 0; disabling.", section="LIKELIHOOD")
            state.params["_shared_event_latent_enabled"] = False
            return

        # Resolve selection/interpolation file paths (relative to checkpoint_dir)
        def _resolve_path(v):
            if v is None:
                return None
            p = str(v)
            if not p:
                return None
            if os.path.isabs(p):
                return p
            base = str(state.params.get("checkpoint_dir", "."))
            return os.path.join(base, p)

        sel_path = _resolve_path(state.params.get("_shared_event_latent_inducing_plan_selection_outfile", None))
        interp_path = _resolve_path(state.params.get("_shared_event_latent_inducing_plan_interpolation_outfile", None))
        # Prefer runtime-resolved paths from Stage 2/3 if present
        sel_path_rt = state.params.get("_shared_event_latent_inducing_selection_file_runtime", None)
        if sel_path_rt:
            sel_path = str(sel_path_rt)
        interp_path_rt = state.params.get("_shared_event_latent_inducing_interpolation_file_runtime", None)
        if interp_path_rt:
            interp_path = str(interp_path_rt)

        if not sel_path or not os.path.exists(sel_path):
            raise ValueError(
                "shared_event_latent.parameterization='inducing_gp' requires an inducing selection NPZ. "
                "Enable model.likelihood.shared_event_latent.inducing_plan.select=true (Stage 2) or provide the file. "
                f"Expected at: {sel_path}"
            )
        if not interp_path or not os.path.exists(interp_path):
            raise ValueError(
                "shared_event_latent.parameterization='inducing_gp' requires an inducing interpolation NPZ. "
                "Enable model.likelihood.shared_event_latent.inducing_plan.interpolation.enabled=true (Stage 3) or provide the file. "
                f"Expected at: {interp_path}"
            )

        # Load selection
        try:
            sel = np.load(sel_path, allow_pickle=True)
            comp_ids = np.asarray(sel["component_id"], dtype=np.int64)
            comp_off = np.asarray(sel["component_offsets"], dtype=np.int64)
            inducing_event_idx = np.asarray(sel["inducing_event_idx"], dtype=np.int64)
            use_xyz = False
            if "use_xyz" in sel:
                try:
                    use_xyz = bool(int(np.asarray(sel["use_xyz"]).reshape(-1)[0]))
                except Exception:
                    use_xyz = False
        except Exception as e:
            raise ValueError(f"Failed to load inducing selection NPZ '{sel_path}': {e}")

        if comp_ids.size == 0 or comp_off.size != comp_ids.size + 1:
            raise ValueError(f"Invalid inducing selection NPZ '{sel_path}': bad component_offsets/component_id")
        if inducing_event_idx.size == 0:
            raise ValueError(f"Invalid inducing selection NPZ '{sel_path}': empty inducing_event_idx")

        # Load interpolation
        try:
            itp = np.load(interp_path, allow_pickle=True)
            neigh_idx_np = np.asarray(itp["neighbor_inducing_global_idx"], dtype=np.int64)
            neigh_k_np = np.asarray(itp["neighbor_kernel"], dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Failed to load inducing interpolation NPZ '{interp_path}': {e}")

        n_events = int(state.X_src.shape[0])
        if neigh_idx_np.ndim != 2 or neigh_k_np.shape != neigh_idx_np.shape:
            raise ValueError(f"Invalid inducing interpolation NPZ '{interp_path}': neighbor arrays shape mismatch")
        if int(neigh_idx_np.shape[0]) != int(n_events):
            raise ValueError(
                f"Invalid inducing interpolation NPZ '{interp_path}': n_events mismatch "
                f"(file has {int(neigh_idx_np.shape[0])}, run has {n_events})"
            )
        M_total = int(inducing_event_idx.size)
        if int(neigh_idx_np.max(initial=-1)) >= M_total:
            raise ValueError(
                f"Invalid inducing interpolation NPZ '{interp_path}': neighbor indices exceed inducing list "
                f"(max idx {int(neigh_idx_np.max())} vs M_total {M_total})"
            )

        # Store interpolation tensors on device
        state.shared_event_latent_inducing_neighbor_idx = torch.from_numpy(neigh_idx_np).to(device=dev, dtype=torch.int64)
        state.shared_event_latent_inducing_neighbor_k = torch.from_numpy(neigh_k_np).to(device=dev, dtype=torch.float32)

        # Build per-component K_UU blocks on device (RBF kernel with ell_km + small jitter)
        # MAP coordinates in km
        X_map = (state.X_src.detach() + state.dX_src.detach())[:, :3].to(torch.float32).cpu().numpy()
        if use_xyz:
            P = X_map[:, :3].astype(np.float32, copy=False)
            dim_label = "xyz"
        else:
            P = X_map[:, :2].astype(np.float32, copy=False)
            dim_label = "xy"

        jitter = float(state.params.get("_shared_event_latent_inducing_jitter", 1e-6))
        # Require at least tiny numerical jitter so K_UU is strictly PD (proper prior over inducing coeffs).
        jitter = 1e-6 if (not math.isfinite(jitter) or jitter <= 0.0) else jitter
        K_blocks: list[torch.Tensor] = []
        # Optional conditioning diagnostics (cheap for small blocks; skipped for large ones).
        # This helps debug "runaway" behavior caused by nearly-singular K_UU blocks (weakly-regularized coeff modes).
        eig_min_list: list[float] = []
        eig_max_list: list[float] = []
        cond_list: list[float] = []
        skipped_cond_blocks = 0
        max_dim_for_cond = 256
        jitter_used_max = float(jitter)
        jitter_escalated_blocks = 0
        for bi in range(int(comp_ids.size)):
            i0 = int(comp_off[bi]); i1 = int(comp_off[bi + 1])
            if i1 <= i0:
                K_blocks.append(torch.zeros((0, 0), device=dev, dtype=torch.float32))
                continue
            U_ev = inducing_event_idx[i0:i1]
            Uc = torch.from_numpy(P[U_ev]).to(device=dev, dtype=torch.float32)
            # Pairwise distances (M,M)
            D = torch.cdist(Uc, Uc).to(torch.float32)
            K0 = torch.exp(-0.5 * (D / float(ell_km)).square())
            # Enforce symmetry explicitly (GPU cdist can be slightly asymmetric in float32).
            K0 = 0.5 * (K0 + K0.transpose(0, 1))
            mK = int(K0.shape[0])
            if mK <= 0:
                K = K0
            else:
                I = torch.eye(mK, device=dev, dtype=K0.dtype)
                j0 = float(jitter)
                j_used = j0
                K = K0 + (j_used * I)
                # Ensure PD via Cholesky; if it fails, escalate jitter on this block.
                try:
                    L, chol_info = torch.linalg.cholesky_ex(K)
                    if int(chol_info.item()) != 0:
                        max_tries = 6
                        for t in range(1, max_tries + 1):
                            j_used = j0 * (10.0 ** t)
                            K = K0 + (j_used * I)
                            L, chol_info = torch.linalg.cholesky_ex(K)
                            if int(chol_info.item()) == 0:
                                break
                except Exception:
                    pass
                if j_used > j0:
                    jitter_escalated_blocks += 1
                if j_used > jitter_used_max:
                    jitter_used_max = float(j_used)
            K_blocks.append(K)
            try:
                if mK > 0 and mK <= max_dim_for_cond:
                    ev = torch.linalg.eigvalsh(K)
                    lam_min = float(ev.min().item())
                    lam_max = float(ev.max().item())
                    eig_min_list.append(lam_min)
                    eig_max_list.append(lam_max)
                    if lam_min > 0.0:
                        cond_list.append(lam_max / max(lam_min, 1e-30))
                elif mK > max_dim_for_cond:
                    skipped_cond_blocks += 1
            except Exception:
                pass

        # Store in state and params for prior
        offs_t = torch.from_numpy(comp_off).to(device=dev, dtype=torch.int64)
        state.shared_event_latent_inducing_offsets = offs_t
        state.shared_event_latent_inducing_K_blocks = K_blocks
        state.params["_shared_event_latent_inducing_offsets"] = offs_t
        state.params["_shared_event_latent_inducing_K_blocks"] = K_blocks

        # Log K_UU conditioning summary once (helps pick a sensible kernel_jitter).
        try:
            if len(eig_min_list) > 0:
                emn = np.asarray(eig_min_list, dtype=np.float64)
                emx = np.asarray(eig_max_list, dtype=np.float64)
                cnd = np.asarray(cond_list, dtype=np.float64) if len(cond_list) > 0 else None
                state.params["_shared_event_latent_inducing_K_eig_min_min"] = float(np.nanmin(emn))
                state.params["_shared_event_latent_inducing_K_eig_min_med"] = float(np.nanmedian(emn))
                state.params["_shared_event_latent_inducing_K_eig_max_med"] = float(np.nanmedian(emx))
                if cnd is not None and cnd.size > 0:
                    state.params["_shared_event_latent_inducing_K_cond_max"] = float(np.nanmax(cnd))
                info(
                    "Inducing K_UU stats (shared_event_latent inducing_gp): "
                    f"jitter={float(jitter):.3g} jitter_used_max={float(jitter_used_max):.3g} "
                    f"jitter_escalated_blocks={int(jitter_escalated_blocks)} blocks={int(comp_ids.size)} "
                    f"eig_min[min/med]={float(np.nanmin(emn)):.3g}/{float(np.nanmedian(emn)):.3g} "
                    f"eig_max[med]={float(np.nanmedian(emx)):.3g} "
                    f"cond[max]={float(np.nanmax(cnd)):.3g}" if (cnd is not None and cnd.size > 0) else
                    "Inducing K_UU stats (shared_event_latent inducing_gp): "
                    f"jitter={float(jitter):.3g} jitter_used_max={float(jitter_used_max):.3g} "
                    f"jitter_escalated_blocks={int(jitter_escalated_blocks)} blocks={int(comp_ids.size)} "
                    f"eig_min[min/med]={float(np.nanmin(emn)):.3g}/{float(np.nanmedian(emn)):.3g} "
                    f"eig_max[med]={float(np.nanmedian(emx)):.3g} "
                    f"(cond skipped for {int(skipped_cond_blocks)} blocks > {int(max_dim_for_cond)})",
                    section="LIKELIHOOD",
                )
        except Exception:
            pass

        # Stage 5 (FITC): compute diagonal residual Λ_ee = max(0, 1 - Q_ee) where
        # Q_ee ≈ K_eU K_UU^{-1} K_Ue, approximated using the same m-neighbor subset as interpolation.
        fitc_enable = bool(state.params.get("_shared_event_latent_inducing_fitc_enable", False))
        if fitc_enable:
            try:
                t_fitc0 = time.time()
                idx_np = state.shared_event_latent_inducing_neighbor_idx.detach().cpu().numpy().astype(np.int64, copy=False)
                k_np = state.shared_event_latent_inducing_neighbor_k.detach().cpu().numpy().astype(np.float32, copy=False)
                offs_np = comp_off.astype(np.int64, copy=False)
                K_np = [Kb.detach().cpu().numpy().astype(np.float32, copy=False) for Kb in K_blocks]
                n_ev = int(idx_np.shape[0])
                q_diag_np = np.zeros((n_ev,), dtype=np.float32)
                for e in range(n_ev):
                    idx_row = idx_np[e]
                    k_row = k_np[e]
                    m = idx_row >= 0
                    if not bool(m.any()):
                        q = 0.0
                    else:
                        idxv = idx_row[m]
                        kv = k_row[m]
                        g0 = int(idxv[0])
                        bi = int(np.searchsorted(offs_np[1:], g0, side="right"))
                        if bi < 0 or bi >= len(K_np):
                            q = 0.0
                        else:
                            i0 = int(offs_np[bi])
                            loc = (idxv - i0).astype(np.int64, copy=False)
                            try:
                                Kmm = K_np[bi][np.ix_(loc, loc)]
                                sol = np.linalg.solve(Kmm, kv)
                                q = float(kv.dot(sol))
                            except Exception:
                                q = 0.0
                    q_diag_np[e] = float(q)
                q_diag_np = np.clip(q_diag_np, 0.0, 1.0).astype(np.float32, copy=False)
                resid_np = np.maximum(0.0, 1.0 - q_diag_np).astype(np.float32, copy=False)
                state.shared_event_latent_inducing_fitc_q_diag = torch.from_numpy(q_diag_np).to(device=dev, dtype=torch.float32)
                state.shared_event_latent_inducing_fitc_resid = torch.from_numpy(resid_np).to(device=dev, dtype=torch.float32)
                dt_fitc = time.time() - t_fitc0
                info(
                    "FITC diag residual (shared_event_latent inducing_gp): "
                    f"q_diag[min/mean/max]={float(q_diag_np.min()):.3g}/{float(q_diag_np.mean()):.3g}/{float(q_diag_np.max()):.3g} "
                    f"resid[min/mean/max]={float(resid_np.min()):.3g}/{float(resid_np.mean()):.3g}/{float(resid_np.max()):.3g} "
                    f"(dt={dt_fitc:.2f}s)",
                    section="LIKELIHOOD",
                )
            except Exception as e:
                warn(f"FITC diag residual computation failed (shared_event_latent inducing_gp): {e}", section="LIKELIHOOD")

        # Allocate inducing coefficients (sampled). This represents K_UU^{-1} g in a predictive-process GP.
        # Default: per-station coefficients c[station, M_total, 2].
        # Optional: fixed station-geometry basis reduces station DOF: c[rank_R, M_total, 2] with station weights W[station, R].
        n_stations = int(getattr(state, "n_stations", 0))
        use_sta_basis = bool(state.params.get("_shared_event_latent_station_basis_enabled", False))
        r_sta = 0
        if use_sta_basis:
            try:
                r_req = int(state.params.get("_shared_event_latent_station_basis_r", 0))
                ell_sta = float(state.params.get("_shared_event_latent_station_basis_ell_km", 0.0))
                jitter_sta = float(state.params.get("_shared_event_latent_station_basis_jitter", 1e-6))
                method = str(state.params.get("_shared_event_latent_station_basis_method", "eigh_rbf")).strip().lower()
                if r_req < 1 or not (ell_sta > 0.0) or method != "eigh_rbf":
                    raise ValueError("invalid station_basis config")
                r_sta = int(min(int(r_req), int(n_stations)))
                if r_sta < 1:
                    raise ValueError("station_basis.r must be >= 1 and <= n_stations")

                # Build station XY in sta_idx order (aligned with row_station_index mapping).
                dt = state.dtimes
                if "sta_idx" not in dt.columns:
                    raise ValueError("dtimes is missing 'sta_idx' (station index)")
                if "X" not in dt.columns or "Y" not in dt.columns:
                    raise ValueError("dtimes is missing projected station coordinates 'X','Y'")
                sta_xy = (
                    dt.select([pl.col("sta_idx"), pl.col("X"), pl.col("Y")])
                    .unique(subset=["sta_idx"], maintain_order=True)
                    .sort("sta_idx")
                )
                if int(sta_xy.shape[0]) != int(n_stations):
                    raise ValueError(
                        f"station_basis: expected {n_stations} stations (0..{n_stations-1}) but dtimes has {int(sta_xy.shape[0])} unique sta_idx. "
                        "This can happen if sta_idx was not rebuilt after filtering."
                    )
                xy_np = sta_xy.select([pl.col("X"), pl.col("Y")]).to_numpy().astype(np.float32, copy=False)
                XY = torch.from_numpy(xy_np).to(device=dev, dtype=torch.float32)  # (S,2)

                # RBF station kernel + jitter; take top-R eigenpairs to form a fixed basis W = V sqrt(Λ).
                D_sta = torch.cdist(XY, XY).to(torch.float32)
                K_sta = torch.exp(-0.5 * (D_sta / float(ell_sta)).square())
                j = 1e-6 if (not math.isfinite(jitter_sta) or jitter_sta <= 0.0) else float(jitter_sta)
                K_sta = K_sta + (j * torch.eye(int(K_sta.shape[0]), device=dev, dtype=K_sta.dtype))
                evals, evecs = torch.linalg.eigh(K_sta)
                evals = evals.clamp_min(0.0)
                evals_r = evals[-r_sta:]
                evecs_r = evecs[:, -r_sta:]
                W_sta = evecs_r * torch.sqrt(evals_r).unsqueeze(0)  # (S,R)
                state.shared_event_latent_station_basis_W = W_sta.to(device=dev, dtype=torch.float32)
                state.shared_event_latent_station_basis_r = int(r_sta)
                state.params["_shared_event_latent_station_basis_r_runtime"] = int(r_sta)
            except Exception as e:
                warn(f"Failed to build shared_event_latent station_basis; disabling (err={e})", section="LIKELIHOOD")
                use_sta_basis = False
                r_sta = 0
                state.shared_event_latent_station_basis_W = None
                state.shared_event_latent_station_basis_r = 0

        if use_sta_basis and int(r_sta) > 0 and isinstance(getattr(state, "shared_event_latent_station_basis_W", None), torch.Tensor):
            c0 = torch.zeros((int(r_sta), M_total, 2), dtype=torch.float32, device=dev)
        else:
            c0 = torch.zeros((n_stations, M_total, 2), dtype=torch.float32, device=dev)
        state.shared_event_latent_b = torch.nn.Parameter(c0)

        # Clear full-mode graph keys
        state.shared_event_latent_u = None
        state.shared_event_latent_v = None
        state.shared_event_latent_w = None
        state.shared_event_latent_q_diag = 0.0
        state.params["_shared_event_latent_u"] = None
        state.params["_shared_event_latent_v"] = None
        state.params["_shared_event_latent_w"] = None
        state.params["_shared_event_latent_q_diag_runtime"] = 0.0

        try:
            coeff_shape = tuple(state.shared_event_latent_b.shape) if state.shared_event_latent_b is not None else ()
        except Exception:
            coeff_shape = ()
        info(
            f"Initialized shared_event_latent (inducing_gp): coeff shape={coeff_shape} "
            f"neighbor_m={int(neigh_idx_np.shape[1])} comps={int(comp_ids.size)} ell_km={ell_km:g} dims={dim_label} "
            f"jitter={float(jitter):.3g} (FITC={'on' if fitc_enable else 'off'}) "
            f"station_basis_r={int(getattr(state, 'shared_event_latent_station_basis_r', 0))}",
            section="LIKELIHOOD",
        )
        return

    # --- graph_gmrf parameterization (DD-linked event graph) ---
    if mode == "graph_gmrf":
        dev = state.device
        n_events = int(state.X_src.shape[0])
        n_stations = int(getattr(state, "n_stations", 0))
        ell_km = float(state.params.get("_shared_event_latent_ell_km", 0.0))
        if not (ell_km > 0.0) or n_events <= 1 or n_stations <= 0:
            warn("shared_event_latent graph_gmrf requires ell_km > 0 and n_events>1; disabling.", section="LIKELIHOOD")
            state.params["_shared_event_latent_enabled"] = False
            return

        # Optional diagonal term and Laplacian scaling (lambda)
        q_diag = float(state.params.get("_shared_event_latent_q_diag", 1.0))
        try:
            lam = float(state.params.get("_shared_event_latent_graph_lambda", 1.0))
        except Exception:
            lam = 1.0
        if not math.isfinite(q_diag) or q_diag < 0.0:
            q_diag = 1.0
        if not math.isfinite(lam) or lam < 0.0:
            lam = 1.0

        # Build station basis W if enabled (same as inducing_gp path)
        use_sta_basis = bool(state.params.get("_shared_event_latent_station_basis_enabled", False))
        r_sta = 0
        if use_sta_basis:
            try:
                r_req = int(state.params.get("_shared_event_latent_station_basis_r", 0))
                ell_sta = float(state.params.get("_shared_event_latent_station_basis_ell_km", 0.0))
                jitter_sta = float(state.params.get("_shared_event_latent_station_basis_jitter", 1e-6))
                method = str(state.params.get("_shared_event_latent_station_basis_method", "eigh_rbf")).strip().lower()
                if r_req < 1 or not (ell_sta > 0.0) or method != "eigh_rbf":
                    raise ValueError("invalid station_basis config")
                r_sta = int(min(int(r_req), int(n_stations)))
                if r_sta < 1:
                    raise ValueError("station_basis.r must be >= 1 and <= n_stations")

                dt = state.dtimes
                if "sta_idx" not in dt.columns:
                    raise ValueError("dtimes is missing 'sta_idx' (station index)")
                if "X" not in dt.columns or "Y" not in dt.columns:
                    raise ValueError("dtimes is missing projected station coordinates 'X','Y'")
                sta_xy = (
                    dt.select([pl.col("sta_idx"), pl.col("X"), pl.col("Y")])
                    .unique(subset=["sta_idx"], maintain_order=True)
                    .sort("sta_idx")
                )
                if int(sta_xy.shape[0]) != int(n_stations):
                    raise ValueError(
                        f"station_basis: expected {n_stations} stations (0..{n_stations-1}) but dtimes has {int(sta_xy.shape[0])} unique sta_idx."
                    )
                xy_np = sta_xy.select([pl.col("X"), pl.col("Y")]).to_numpy().astype(np.float32, copy=False)
                XY = torch.from_numpy(xy_np).to(device=dev, dtype=torch.float32)  # (S,2)
                D_sta = torch.cdist(XY, XY).to(torch.float32)
                K_sta = torch.exp(-0.5 * (D_sta / float(ell_sta)).square())
                j = 1e-6 if (not math.isfinite(jitter_sta) or jitter_sta <= 0.0) else float(jitter_sta)
                K_sta = K_sta + (j * torch.eye(int(K_sta.shape[0]), device=dev, dtype=K_sta.dtype))
                evals, evecs = torch.linalg.eigh(K_sta)
                evals = evals.clamp_min(0.0)
                evals_r = evals[-r_sta:]
                evecs_r = evecs[:, -r_sta:]
                W_sta = evecs_r * torch.sqrt(evals_r).unsqueeze(0)  # (S,R)
                state.shared_event_latent_station_basis_W = W_sta.to(device=dev, dtype=torch.float32)
                state.shared_event_latent_station_basis_r = int(r_sta)
                state.params["_shared_event_latent_station_basis_r_runtime"] = int(r_sta)
            except Exception as e:
                warn(f"Failed to build shared_event_latent station_basis; disabling (err={e})", section="LIKELIHOOD")
                use_sta_basis = False
                r_sta = 0
                state.shared_event_latent_station_basis_W = None
                state.shared_event_latent_station_basis_r = 0

        # Allocate explicit event latents b (sampled)
        if use_sta_basis and int(r_sta) > 0 and isinstance(getattr(state, "shared_event_latent_station_basis_W", None), torch.Tensor):
            b0 = torch.zeros((int(r_sta), int(n_events), 2), dtype=torch.float32, device=dev)
        else:
            b0 = torch.zeros((int(n_stations), int(n_events), 2), dtype=torch.float32, device=dev)
        state.shared_event_latent_b = torch.nn.Parameter(b0)

        # Build unique undirected DD-linked edges from II (CPU)
        t0 = time.time()
        II_cpu = getattr(state, "_II_cpu", None)
        if II_cpu is None:
            II_cpu = state.II.detach().to("cpu").numpy().astype(np.int64, copy=False)
            try:
                state._II_cpu = II_cpu
            except Exception:
                pass
        if II_cpu.ndim != 2 or II_cpu.shape[1] != 2:
            raise ValueError("shared_event_latent graph_gmrf: II must be (N,2)")
        a = II_cpu[:, 0]
        b = II_cpu[:, 1]
        u0 = np.minimum(a, b)
        v0 = np.maximum(a, b)
        msk = (u0 != v0)
        u0 = u0[msk]
        v0 = v0[msk]
        if u0.size == 0:
            warn("shared_event_latent graph_gmrf: no non-self DD edges; disabling.", section="LIKELIHOOD")
            state.params["_shared_event_latent_enabled"] = False
            return
        # sort + dedup (lexicographic)
        order = np.lexsort((v0, u0))
        u1 = u0[order]
        v1 = v0[order]
        keep = np.ones((u1.size,), dtype=bool)
        keep[1:] = (u1[1:] != u1[:-1]) | (v1[1:] != v1[:-1])
        u = u1[keep].astype(np.int64, copy=False)
        v = v1[keep].astype(np.int64, copy=False)

        # MAP coordinates (km), fixed weights built once
        X_map = (state.X_src.detach() + state.dX_src.detach())[:, :3].to(torch.float32).cpu().numpy()

        # Compute edge weights w_ij = lambda * exp(-||xi-xj||/ell_km)
        E = int(u.size)
        w_np = np.empty((E,), dtype=np.float32)
        chunk = 1_000_000
        for i0 in range(0, E, chunk):
            i1 = min(i0 + chunk, E)
            du = X_map[u[i0:i1]]
            dv = X_map[v[i0:i1]]
            d = np.linalg.norm(du - dv, axis=1).astype(np.float32, copy=False)
            w_np[i0:i1] = (float(lam) * np.exp(-d / float(ell_km))).astype(np.float32, copy=False)

        # Move to device
        u_t = torch.from_numpy(u).to(device=dev, dtype=torch.int64)
        v_t = torch.from_numpy(v).to(device=dev, dtype=torch.int64)
        w_t = torch.from_numpy(w_np).to(device=dev, dtype=torch.float32)

        state.shared_event_latent_u = u_t
        state.shared_event_latent_v = v_t
        state.shared_event_latent_w = w_t
        state.shared_event_latent_q_diag = float(q_diag)
        state.params["_shared_event_latent_u"] = u_t
        state.params["_shared_event_latent_v"] = v_t
        state.params["_shared_event_latent_w"] = w_t
        state.params["_shared_event_latent_q_diag_runtime"] = float(q_diag)

        dt_s = time.time() - t0
        info(
            "Initialized shared_event_latent (graph_gmrf): "
            f"b_shape={tuple(state.shared_event_latent_b.shape)} edges={int(E)} "
            f"ell_km={float(ell_km):g} lambda={float(lam):g} q_diag={float(q_diag):g} "
            f"station_basis_r={int(getattr(state, 'shared_event_latent_station_basis_r', 0))} "
            f"(dt={dt_s:.2f}s)",
            section="LIKELIHOOD",
        )
        return

    # MAP coordinates in km (fix kernel to MAP)
    #
    # IMPORTANT: The original implementation used a full pairwise distance matrix via torch.cdist,
    # which is O(n_events^2) memory/time and becomes a major pause between Phase 1 and Phase 2
    # for large catalogs. We prefer an exact kNN query via SciPy cKDTree when available.
    t0 = time.time()
    X_map = (state.X_src.detach() + state.dX_src.detach())[:, :3].to(torch.float32).cpu()
    X_np = X_map.numpy()
    # Directed kNN edges (i -> j) and distances (km)
    ii_np = None
    jj_np = None
    dd_np = None
    used_backend = "torch_cdist"
    try:
        from scipy.spatial import cKDTree  # type: ignore
        # cKDTree expects finite float coords; use float64 for robustness
        X64 = X_np.astype("float64", copy=False)
        tree = cKDTree(X64)
        kq = int(min(int(knn) + 1, int(n_events)))
        # query returns (d, idx) with shapes (n_events,kq) for kq>1
        try:
            dists, nbrs = tree.query(X64, k=kq, workers=-1)
        except TypeError:
            # Older SciPy versions used `n_jobs` or had no parallelism kwarg.
            try:
                dists, nbrs = tree.query(X64, k=kq, n_jobs=-1)  # type: ignore[call-arg]
            except TypeError:
                dists, nbrs = tree.query(X64, k=kq)
        # Drop the self-neighbor (distance 0). After knn=min(knn,n_events-1), kq>=2 here.
        dists = dists[:, 1:]
        nbrs = nbrs[:, 1:]
        ii_np = np.repeat(np.arange(n_events, dtype=np.int64), int(knn))
        jj_np = nbrs.reshape(-1).astype(np.int64, copy=False)
        dd_np = dists.reshape(-1).astype(np.float32, copy=False)
        used_backend = "scipy_ckdtree"
    except Exception:
        # Fallback: exact but O(n^2) memory. Keep for small problems / minimal environments.
        D = torch.cdist(X_map, X_map).to(torch.float32)
        D.fill_diagonal_(float("inf"))
        vals, nbrs = torch.topk(D, k=knn, largest=False)
        ii = torch.arange(n_events, dtype=torch.int64).unsqueeze(1).expand(-1, knn).reshape(-1)
        jj = nbrs.reshape(-1).to(torch.int64)
        dd = vals.reshape(-1).to(torch.float32)
        ii_np = ii.numpy()
        jj_np = jj.numpy()
        dd_np = dd.numpy()
        used_backend = "torch_cdist"

    # Edge weights
    ww_np = np.exp(-0.5 * (dd_np / float(ell_km)) ** 2).astype(np.float32, copy=False)

    # Symmetrize and coalesce into undirected edges (u < v)
    u = np.minimum(ii_np, jj_np).astype(np.int64, copy=False)
    v = np.maximum(ii_np, jj_np).astype(np.int64, copy=False)
    pairs = np.stack([u, v], axis=1)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    w_sum = np.zeros((uniq.shape[0],), dtype=np.float64)
    w_cnt = np.zeros((uniq.shape[0],), dtype=np.float64)
    np.add.at(w_sum, inv, ww_np.astype(np.float64))
    np.add.at(w_cnt, inv, 1.0)
    w_mean = (w_sum / np.maximum(1.0, w_cnt)).astype(np.float32)

    # Drop self-edges if any slipped in
    mask = (uniq[:, 0] != uniq[:, 1])
    uniq = uniq[mask]
    w_mean = w_mean[mask]

    # Move graph to device
    dev = state.device
    state.shared_event_latent_u = torch.tensor(uniq[:, 0], dtype=torch.int64, device=dev)
    state.shared_event_latent_v = torch.tensor(uniq[:, 1], dtype=torch.int64, device=dev)
    state.shared_event_latent_w = torch.tensor(w_mean, dtype=torch.float32, device=dev)
    state.shared_event_latent_q_diag = float(max(0.0, q_diag))

    # Expose graph to modeling via params (internal keys)
    state.params["_shared_event_latent_u"] = state.shared_event_latent_u
    state.params["_shared_event_latent_v"] = state.shared_event_latent_v
    state.params["_shared_event_latent_w"] = state.shared_event_latent_w
    state.params["_shared_event_latent_q_diag_runtime"] = float(state.shared_event_latent_q_diag)

    # Initialize latent b (station,event,phase) on device
    b0 = torch.zeros((n_stations, n_events, 2), dtype=torch.float32, device=dev)
    state.shared_event_latent_b = torch.nn.Parameter(b0)

    dt_s = time.time() - t0
    info(
        f"Initialized shared_event_latent: b shape=({n_stations},{n_events},2), knn={knn}, "
        f"edges={int(state.shared_event_latent_u.numel())}, ell_km={ell_km:g}, q_diag={state.shared_event_latent_q_diag:g} "
        f"(knn_backend={used_backend}, dt={dt_s:.1f}s)",
        section="LIKELIHOOD",
    )
    return



def locate_all(
    params: dict,
    origins0: pl.DataFrame,
    dtimes: pl.DataFrame,
    model: nn.Module,
    device: torch.device,
    wandb_logger=None,
) -> None:
    """Run the full SPIDER location pipeline (warmup + SGLD sampling)."""
    state = _build_initial_state(params, origins0, dtimes, model, device)
    phase, start_epoch, skip_saving_first_epoch, ckpt = _resume_or_initialize(state)

    # Optional residual-based outlier removal, only at fresh Phase 1 start.
    # Prefer running this during data prep (prepare_input_dfs) immediately after the linearization filter,
    # so we avoid doing an extra expensive residual pass here.
    if (
        ckpt is None
        and phase == "phase1"
        and start_epoch == 0
        and bool(state.params.get("residual_filter_enable", False))
        and not bool(state.params.get("_residual_filter_applied_in_prepare_input_dfs", False))
    ):
        _pre_filter_outlier_residuals(state)
        _print_initial_residual_stats(state)

    # Phase 1: MAP estimation with Adam optimizer
    if phase == "phase1":
        if wandb_logger:
            wandb_logger.start_phase("phase1")
        _phase1_map_warmup(state, start_epoch, wandb_logger)
        phase = "phase2"

    # Build fixed event-kernel graph at MAP + initialize latent shared-event b if enabled.
    _maybe_report_shared_event_latent_inducing_plan(state)
    _maybe_select_shared_event_latent_inducing_points(state)
    _maybe_build_shared_event_latent_inducing_interpolation(state)
    _maybe_init_shared_event_latent(state)

    # Set up sampler and optionally load state if resuming from sampling phases
    sampler = _setup_sampler(state)
    if ckpt is not None and phase in {"phase2", "phase3", "phase4"}:
        # IMPORTANT: Do not load Phase 1 (Adam) optimizer state into the sampler backend.
        # Phase 1 checkpoints store Adam under "optimizer_state_dict"; Phase 2–4 store sampler state there.
        ckpt_phase = str(ckpt.get("phase", "")).strip().lower()
        if ckpt_phase in {"phase2", "phase3", "phase4"}:
            try:
                sampler.load_state_dict(ckpt.get("optimizer_state_dict", {}))
            except Exception as e:
                print(f"Warning: could not load sampler state: {e}")
        else:
            # Phase 1 checkpoint: we will (optionally) transplant preconditioner state from Adam below.
            pass
        # One-time fix: enforce current JSON hyperparameters after resume
        try:
            # Enforce required param-group keys expected by our current samplers.
            lr_user = float(state.params["lr_sampler"])
            lr_mode = str(state.params["sampler_lr_mode"]).strip().lower()
            backend = str(state.params["sampler_backend"]).strip().lower()
            if backend in {"psgld", "sghmc", "adaptive_sghmc"} and lr_mode == "per_obs":
                lr_json = lr_user / float(max(1, int(state.N)))
            else:
                lr_json = lr_user

            beta_json = float(state.params["sampler_beta"])
            eps_json = float(state.params["sampler_eps"])
            temp_json = float(state.params["sampler_temperature"])
            precond_json = bool(state.params["sampler_preconditioning"])

            for g in sampler.param_groups:
                g.setdefault("beta", beta_json)
                g.setdefault("eps", eps_json)
                g.setdefault("temperature", temp_json)
                g.setdefault("preconditioning", precond_json)
                if 'add_noise' not in g:
                    ns = float(g.get('noise_scale', 0.0))
                    g['add_noise'] = bool(ns > 0.0)
                g.setdefault("n_obs", int(state.N))
                g.setdefault("noise_scale", 0.0)
                # If resuming with SGHMC, ensure required friction key exists
                if state.sampler_backend == "sghmc":
                    g.setdefault("alpha", float(state.params.get("sghmc_alpha", 0.01)))

            if hasattr(sampler, "set_lr"):
                sampler.set_lr(lr_json)  # type: ignore[attr-defined]
            else:
                for g in sampler.param_groups:
                    g['lr'] = lr_json
            # Also take temperature from JSON on restart (override)
            for g in sampler.param_groups:
                g["temperature"] = temp_json
            # Re-apply per-group overrides after global JSON overwrite.
            _apply_sampler_group_overrides(state, sampler)
            # For phase2/phase3 restarts, ensure noise is off at entry
            if phase in {"phase2", "phase3"}:
                for g in sampler.param_groups:
                    if 'add_noise' in g:
                        g['add_noise'] = False
                    g['noise_scale'] = 0.0
        except Exception as e:
            print(f"Warning: could not apply JSON params on resume: {e}")

    # Phase 2
    if phase == "phase2":
        if wandb_logger:
            wandb_logger.start_phase("phase2")
        # Transfer preconditioning state from Adam to sampler if coming from phase 1
        if ckpt is None or ckpt.get("phase") == "phase1":
            try:
                from spider.optim.backends import transplant_from_adam_if_supported
                transplant_from_adam_if_supported(state.optimizer, sampler)
                print("Transferred preconditioning state from Adam to sampler (if supported)")
            except Exception as e:
                print(f"Warning: could not transplant preconditioner from Adam (backend): {e}")
        # Receiver-centric sigma_scale computation removed
        
        _phase2_preconditioner(state, start_epoch=start_epoch, skip_saving_first_epoch=skip_saving_first_epoch, wandb_logger=wandb_logger)
        phase = "phase3"

    # Phase 3
    if phase == "phase3":
        if wandb_logger:
            wandb_logger.start_phase("phase3")
        _phase3_noise_ramp(state, start_epoch=start_epoch, skip_saving_first_epoch=skip_saving_first_epoch, wandb_logger=wandb_logger)
        phase = "phase4"
    # Phase 4
    if phase == "phase4":
        if wandb_logger:
            wandb_logger.start_phase("phase4")
    # Restart epoch counting at 0 when entering phase 4 from earlier phases.
    # Only preserve epoch if we are resuming directly from a phase 4 checkpoint.
    ckpt_phase = str(ckpt.get("phase", "")) if ckpt is not None else ""
    ckpt_phase_norm = ckpt_phase.strip().lower()
    resuming_direct_phase4 = (phase == "phase4" and ckpt_phase_norm == "phase4")
    sgld_start_epoch = start_epoch if resuming_direct_phase4 else 0
    _phase4_sampling(state, sgld_start_epoch, skip_saving_first_epoch, wandb_logger)
    return


def locate_map(
    params: dict,
    origins0: pl.DataFrame,
    dtimes: pl.DataFrame,
    model: nn.Module,
    device: torch.device,
    wandb_logger=None,
    *,
    bundle_out: Optional[str] = None,
) -> None:
    """
    Run only Phase 1 (MAP warmup) and optionally dump a Phase-2 bundle.

    This is intended to be run once, then reused by many sampling runs (single or multi-chain),
    avoiding any rerun of Phase 1 and its dataset filtering/rebuilds.
    """
    state = _build_initial_state(params, origins0, dtimes, model, device)
    phase, start_epoch, _, ckpt = _resume_or_initialize(state)
    if ckpt is not None and str(ckpt.get("phase", "")).strip().lower() != "phase1":
        raise ValueError(
            f"locate-map can only resume/run Phase 1, but latest checkpoint is phase={ckpt.get('phase')!r}. "
            "Delete checkpoints or set inference.runtime.reset_batch_numbers=true."
        )
    if phase != "phase1":
        # This can happen if reset logic forced phase2; for locate-map we require phase1.
        phase = "phase1"
        start_epoch = 0

    # Optional residual-based outlier removal, only at fresh Phase 1 start.
    # Prefer running this during data prep (prepare_input_dfs) immediately after the linearization filter.
    if (
        ckpt is None
        and start_epoch == 0
        and bool(state.params.get("residual_filter_enable", False))
        and not bool(state.params.get("_residual_filter_applied_in_prepare_input_dfs", False))
    ):
        _pre_filter_outlier_residuals(state)
        _print_initial_residual_stats(state)

    if wandb_logger:
        wandb_logger.start_phase("phase1")
    _phase1_map_warmup(state, start_epoch=start_epoch, wandb_logger=wandb_logger)

    if bundle_out:
        # Dump "exact input to Phase 2": post-phase1 filtered dtimes/origins, MAP ΔX, and Adam state
        try:
            save_phase2_bundle(
                path=str(bundle_out),
                params=state.params,
                origins0=state.origins0,
                dtimes=state.dtimes,
                dX_src=state.dX_src.detach(),
                noise_log_scale=(state.log_scale_theta.detach() if state.log_scale_theta is not None else None),
                phase1_optimizer_state_dict=(state.optimizer.state_dict() if state.optimizer is not None else {}),
                global_step_count=int(state.global_step_count),
            )
            info(f"Wrote Phase-2 bundle -> {bundle_out}", section="BUNDLE")
        except Exception as e:
            warn(f"Failed to write Phase-2 bundle: {e}", section="BUNDLE")
    return


def locate_sample_from_bundle(
    *,
    params: dict,
    bundle_path: str,
    model: nn.Module,
    device: torch.device,
    wandb_logger=None,
) -> None:
    """
    Start at Phase 2 using a previously dumped Phase-2 bundle.

    This bypasses Phase 1 entirely and uses:
      - the post-phase1 filtered dtimes/origins
      - the MAP ΔX_src
      - the Phase-1 Adam optimizer state (for preconditioner transplant, if supported)
    """
    bun = load_phase2_bundle(path=str(bundle_path))

    # Use bundle dataset; keep current run params (but overwrite with bundled materialized params if desired)
    # For safety, we keep the *current* params dict but allow the bundle to carry the materialized keys
    # that core code expects (e.g., flat materializations).
    run_params = params
    try:
        if isinstance(bun.params, dict):
            # Prefer current runtime flags (device, wandb) but keep bundle materializations.
            run_params = {**bun.params, **params}
    except Exception:
        run_params = params

    state = _build_initial_state(run_params, bun.origins0, bun.dtimes, model, device)

    # Mirror the important "reset/clear samples" behavior from `_resume_or_initialize`,
    # but without ever resuming Phase 2–4 checkpoints (bundle start is always Phase 2 epoch 0).
    try:
        reset_batch_numbers = bool(state.params.get("reset_batch_numbers", False))
        clear_samples_on_reset = bool(state.params.get("clear_samples_on_reset", False))
        if reset_batch_numbers:
            info("Resetting batch numbers to 0 (reset_batch_numbers=True)", section="RUN")
            clear_checkpoint_files(state.params)
            if clear_samples_on_reset:
                clear_samples_file(state.params)
            state.sample_count = 0
        else:
            if clear_samples_on_reset:
                info("Clearing samples file (clear_samples_on_reset=True)", section="SAMPLES")
                clear_samples_file(state.params)
            # Continue batch numbering if samples file exists
            state.sample_count = get_next_sample_count(state.params)
            if state.sample_count > 0:
                info(f"Continuing from batch number {state.sample_count} (existing samples file found)", section="SAMPLES")
    except Exception:
        pass

    # Restore MAP solution
    state.dX_src.data.copy_(bun.dX_src.to(device=state.device, dtype=torch.float32))
    if state.log_scale_theta is not None and bun.noise_log_scale is not None:
        try:
            state.log_scale_theta.data.copy_(bun.noise_log_scale.to(device=state.device, dtype=torch.float32))
        except Exception:
            pass

    # Restore Phase-1 optimizer state so Phase-2 can transplant preconditioner stats if backend supports it.
    try:
        if isinstance(bun.phase1_optimizer_state_dict, dict) and state.optimizer is not None:
            state.optimizer.load_state_dict(bun.phase1_optimizer_state_dict)
    except Exception as e:
        warn(f"Could not load phase1 optimizer state from bundle: {e}", section="BUNDLE")

    state.global_step_count = int(bun.global_step_count)

    # From here, behave as if Phase 1 just finished.
    phase = "phase2"
    start_epoch = 0
    skip_saving_first_epoch = False
    ckpt = None

    _maybe_report_shared_event_latent_inducing_plan(state)
    _maybe_select_shared_event_latent_inducing_points(state)
    _maybe_build_shared_event_latent_inducing_interpolation(state)
    _maybe_init_shared_event_latent(state)

    sampler = _setup_sampler(state)

    if wandb_logger:
        wandb_logger.start_phase("phase2")
    # Transfer preconditioning state from Adam to sampler if coming from phase 1 (bundle always is)
    try:
        from spider.optim.backends import transplant_from_adam_if_supported
        transplant_from_adam_if_supported(state.optimizer, sampler)
        print("Transferred preconditioning state from Adam to sampler (if supported)")
    except Exception as e:
        print(f"Warning: could not transplant preconditioner from Adam (backend): {e}")

    _phase2_preconditioner(
        state, start_epoch=start_epoch, skip_saving_first_epoch=skip_saving_first_epoch, wandb_logger=wandb_logger
    )
    if wandb_logger:
        wandb_logger.start_phase("phase3")
    _phase3_noise_ramp(
        state, start_epoch=0, skip_saving_first_epoch=False, wandb_logger=wandb_logger
    )
    if wandb_logger:
        wandb_logger.start_phase("phase4")
    _phase4_sampling(state, 0, False, wandb_logger)
    return
