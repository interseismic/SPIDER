

# Standardized stdout helper
def _log(*parts, section: str = "RUN", **_kwargs) -> None:
    msg = " ".join(str(p) for p in parts)
    low = msg.strip().lower()
    if low.startswith("warning") or low.startswith("error"):
        warn(msg, section=section)
    else:
        info(msg, section=section)

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
from spider.core.shared_event_re_whitening import build_whitening_cache_entry, _make_cache_key

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


def _ddp_enabled(params: dict) -> bool:
    try:
        return int(params.get("_ddp_world_size", 1) or 1) > 1
    except Exception:
        return False


def _ddp_is_main(params: dict) -> bool:
    try:
        return int(params.get("_ddp_rank", 0) or 0) == 0
    except Exception:
        return True


def _apply_likelihood_group(state: "LocateState", group_key: str) -> None:
    groups = state.params.get("_likelihood_groups", {})
    if not isinstance(groups, dict):
        return
    group = groups.get(group_key, None)
    if not isinstance(group, dict):
        return
    for k, v in group.items():
        state.params[k] = v
    raw_groups = state.params.get("_likelihood_groups_raw", {})
    if isinstance(raw_groups, dict):
        raw = raw_groups.get(group_key, None)
        if isinstance(raw, dict):
            state.params.setdefault("model", {})["likelihood"] = raw
    state.params["_likelihood_group_active"] = str(group_key)


def _lr_for_phase(params: dict, phase: str) -> float:
    phase_key = str(phase).strip().lower()
    lr_vec = params.get("_sampler_lr_per_phase", None)
    if isinstance(lr_vec, list) and len(lr_vec) == 4:
        idx = {"phase1": 0, "phase2": 1, "phase3": 2, "phase4": 3}.get(phase_key, 1)
        try:
            return float(lr_vec[int(idx)])
        except Exception:
            pass
    if phase_key == "phase1":
        return float(params.get("lr_warmup", 1e-3))
    return float(params.get("lr_sampler", 1e-4))


def _sampler_extra_metrics(optimizer: Optional[torch.optim.Optimizer]) -> Dict[str, float]:
    """
    Extra sampler diagnostics for W&B.

    We intentionally do NOT log `drift_ratio_*` metrics anymore (removed).

    We *do* log two SGHMC-specific diagnostics when SGHMC is active and Langevin noise is enabled:
      - grad_noise_to_langevin_*: ratio of minibatch-gradient-induced update variance to injected noise variance
      - t_eff_var_over_target: effective temperature estimate (variance-based) relative to target temperature
    """
    metrics: Dict[str, float] = {}
    if optimizer is None:
        return metrics

    # We log these diagnostics for any sampler backend that exposes the helper methods.
    # (pSGLD/SGHMC/AdaptiveSGHMC all implement grad_vs_noise_stats in spider.optim.*)
    try:
        if not hasattr(optimizer, "grad_vs_noise_stats") and not hasattr(optimizer, "temperature_stats"):
            return metrics
    except Exception:
        return metrics

    # 1) Drift-vs-noise variance ratio (minibatch gradient noise vs injected noise)
    try:
        if hasattr(optimizer, "grad_vs_noise_stats"):
            s = optimizer.grad_vs_noise_stats()  # type: ignore[attr-defined]
            if isinstance(s, dict):
                med = float(s.get("median", float("nan")))
                gm = float(s.get("gm", float("nan")))
                if med == med:
                    metrics["grad_noise_to_langevin_med"] = med
                if gm == gm:
                    metrics["grad_noise_to_langevin_gm"] = gm
                dt_med = float(s.get("dt_median", float("nan")))
                dt_gm = float(s.get("dt_gm", float("nan")))
                if dt_med == dt_med:
                    metrics["grad_noise_to_langevin_med_dt"] = dt_med
                if dt_gm == dt_gm:
                    metrics["grad_noise_to_langevin_gm_dt"] = dt_gm
                vg = float(s.get("var_g_median", float("nan")))
                vn = float(s.get("var_noise_median", float("nan")))
                if vg == vg:
                    metrics["grad_noise_var_med"] = vg
                if vn == vn:
                    metrics["langevin_noise_var_med"] = vn
                pg = s.get("per_group", None)
                if isinstance(pg, list):
                    for i, gs in enumerate(pg):
                        if not isinstance(gs, dict):
                            continue
                        name = str(gs.get("group_name", f"group{i}")).strip().lower()
                        # Map legacy "core" group to a more user-meaningful label.
                        if name in {"core", "main"}:
                            name = "hypocenter"
                        # sanitize
                        name = "".join([c if (c.isalnum() or c in {"_", "-"} ) else "_" for c in name])
                        gmed = float(gs.get("median", float("nan")))
                        ggm = float(gs.get("gm", float("nan")))
                        if gmed == gmed:
                            metrics[f"grad_noise_to_langevin_med_{name}"] = gmed
                        if ggm == ggm:
                            metrics[f"grad_noise_to_langevin_gm_{name}"] = ggm
                        gdt_med = float(gs.get("dt_median", float("nan")))
                        gdt_gm = float(gs.get("dt_gm", float("nan")))
                        if gdt_med == gdt_med:
                            metrics[f"grad_noise_to_langevin_med_{name}_dt"] = gdt_med
                        if gdt_gm == gdt_gm:
                            metrics[f"grad_noise_to_langevin_gm_{name}_dt"] = gdt_gm
    except Exception:
        pass

    # 2) Effective temperature (variance-based), normalized by target temperature (should be ~1)
    try:
        if hasattr(optimizer, "temperature_stats"):
            t = optimizer.temperature_stats()  # type: ignore[attr-defined]
            if isinstance(t, dict):
                vmed = float(t.get("var_median_over_target", float("nan")))
                vgm = float(t.get("var_gm_over_target", float("nan")))
                if vmed == vmed:
                    metrics["t_eff_var_over_target"] = vmed
                if vgm == vgm:
                    metrics["t_eff_var_over_target_gm"] = vgm
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
            _log("Nuisance: alpha coefficients empty.")
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
            _log(
                f"Nuisance amplitude: |alpha| mean/median/p90/max={aa_mean:.3e}/{aa_med:.3e}/{aa_p90:.3e}/{aa_max:.3e} | "
                f"Δb RMS(P/S)={rms_p:.3e}/{rms_s:.3e} s, median|Δb|(P/S)={mad_p:.3e}/{mad_s:.3e} s (sample={B})"
            )
        else:
            _log(
                f"Nuisance amplitude: |alpha| mean/median/p90/max={aa_mean:.3e}/{aa_med:.3e}/{aa_p90:.3e}/{aa_max:.3e} | Δb not computed"
            )
    except Exception as e:
        _log(f"Warning: nuisance amplitude summary failed: {e}")

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
        _log("Shift guard: missing or invalid prior_event_std; skipping guard.")
        return
    if prior_std.numel() != 4:
        _log("Shift guard: prior_event_std must have 4 entries; skipping guard.")
        return
    # Absolute per-dimension shifts
    abs_shift = torch.abs(state.dX_src)  # (Ne,4)
    thr = factor * prior_std[None, :]
    exceed_mask = (abs_shift > thr).any(dim=1).detach().cpu().numpy()
    if not exceed_mask.any():
        return
    idxs = np.nonzero(exceed_mask)[0]
    _log(f"\nShift guard triggered ({context}): {len(idxs)} event(s) exceeded {factor}×prior_event_std. Printing observations and exiting.")
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
            _log(f"\nEvent evid={evid} (row {idx})")
            shift_vec = abs_shift[idx].detach().cpu().numpy().tolist()
            thr_vec = (prior_std * factor).detach().cpu().numpy().tolist()
            _log(f"Shift (dx,dy,dz,dt) = {shift_vec}")
            _log(f"Thresholds (dx,dy,dz,dt) = {thr_vec}")
            dt_sub = dt_with_resid.filter(
                (pl.col("evid1").cast(pl.Utf8) == pl.lit(str(evid))) |
                (pl.col("evid2").cast(pl.Utf8) == pl.lit(str(evid)))
            )
            # Summary: number of unique (station, phase) pairs for this event
            try:
                sp_unique = dt_sub.select(["network", "station", "phase"]).unique(maintain_order=True)
                _log(f"Unique (station, phase) count: {int(sp_unique.shape[0])}")
            except Exception:
                pass
            # Print a limited sample of rows with all columns (avoid dumping all rows)
            try:
                rows_to_show = min(50, int(dt_sub.shape[0]))
                cols_to_show = int(len(dt_sub.columns))
                with pl.Config(tbl_rows=rows_to_show, tbl_cols=cols_to_show):
                    _log(dt_sub)
            except Exception:
                # Fallback to pandas full-column print
                try:
                    import pandas as _pd  # type: ignore
                    _pd.set_option("display.max_columns", None)
                    _pd.set_option("display.width", 0)
                    _log(dt_sub.to_pandas().head(50).to_string(index=False))
                except Exception:
                    _log(dt_sub)
        except Exception as e:
            _log(f"Shift guard: failed to print observations for evid={evid}: {e}")
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
    return _current_noise_scales(state)


def _pre_filter_outlier_residuals(state: LocateState, *, use_current_dX: bool = False) -> None:
    """Optionally drop dtimes with large residuals.

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

    if use_current_dX:
        _log("Residual pre-filter: computing residuals at current ΔX for outlier detection…")
    else:
        _log("Residual pre-filter: computing initial residuals for outlier detection…")
    # Use a generous residual batch size to speed up pass
    bs = max(int(state.batch_size_warmup), 1)
    with torch.no_grad():
        if use_current_dX:
            residuals = compute_residuals_full(
                state.II, state.YY, state.X_src, state.dX_src, state.model, bs, state.N
            )
        else:
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
                _log(
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
            _log("Residual pre-filter: no outliers detected; keeping all rows.")
            return

        # Materialize indices/mask on CPU for polars filtering
        mask_cpu = mask_keep.detach().cpu().numpy()
        if method == "abs":
            thr_msg = f"thr={thr:.3f}s (abs_max)"
        else:
            thr_msg = f"thr={thr:.3f}s (min(abs_max={abs_thr:.3f}s, mad_sigma*mad={mad_thr:.3f}s))"
        _log(
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
            _log(f"Warning: could not filter dtimes DataFrame: {e}")
        _log(f"Residual pre-filter: remaining dtimes = {state.N}")

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
                _log(f"Warning: could not rebuild event->row map after residual pre-filter: {e}")


@torch.no_grad()
def _print_initial_residual_stats(state: LocateState) -> None:
    """Compute and print initial residual statistics (ΔX=0) for P and S separately."""
    if state.N <= 0:
        _log("Initial residual stats: no observations.")
        return
    bs = max(int(state.batch_size_warmup), 1)
    # For very large datasets, computing full residual stats can be expensive.
    # Use a capped random sample by default to keep this a fast pre-Phase1 sanity check.
    try:
        max_rows = int(state.params.get("initial_residual_stats_max_rows", 200_000))
    except Exception:
        max_rows = 200_000
    max_rows = int(max(10_000, max_rows))
    N = int(state.N)
    if N > max_rows:
        try:
            seed = int(state.params.get("runtime_seed", 0) or 0) + 1337
        except Exception:
            seed = 1337
        rng = np.random.default_rng(int(seed))
        idx_np = rng.choice(N, size=int(max_rows), replace=False).astype(np.int64, copy=False)
        idx_np.sort()
        idx_t = torch.as_tensor(idx_np, device=state.device, dtype=torch.int64)
        II = state.II.index_select(0, idx_t)
        YY = state.YY.index_select(0, idx_t)
        N_eval = int(idx_t.numel())
        suffix = f" (sampled n={N_eval}/{N})"
    else:
        II = state.II
        YY = state.YY
        N_eval = int(N)
        suffix = ""

    # Evaluate residuals at ΔX=0 (i.e., current X_src + 0)
    zero_dX = torch.zeros_like(state.dX_src, device=state.dX_src.device)
    residuals = compute_residuals_full(
        II, YY, state.X_src, zero_dX, state.model, bs, N_eval
    )
    idx_p = torch.nonzero(YY[:, 4] < 0.5).squeeze(-1)
    idx_s = torch.nonzero(YY[:, 4] > 0.5).squeeze(-1)

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
    _log(
        f"Initial residual stats (ΔX=0){suffix}: "
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
    lr = float(g.get("lr", float("nan")))
    return f"precond={precond_s} noise={noise_s} lr={lr:g}"

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
        f"dT={metrics.get('dt_med_abs', float('nan')):.3e}",
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
    _apply_likelihood_group(state, "locate_map")
    ddp_main = (not _ddp_enabled(state.params)) or _ddp_is_main(state.params)
    if ddp_main:
        _log(f"Phase 1: MAP (optimizer=adam) | {_format_sampler_status(state.optimizer)}")
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
                _log(f"Warning: cosine scheduler step failed: {e}")

        
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
        if ddp_main:
            σp_now, σs_now = _current_noise_scales(state)
            _log(_format_epoch_line(
                phase="phase1",
                step=epoch + 1,
                total=int(state.params.get("phase1_epochs", 0)),
                metrics=metrics,
                opt=state.optimizer,
                extra=f"sigma_p={float(σp_now):.4f} sigma_s={float(σs_now):.4f}",
            ))
            _shift_guard_check(state, context=f"phase1 epoch {epoch}")

        # periodic checkpoint for MAP phase
        if ddp_main and checkpoint_interval > 0 and epoch > 0 and (epoch % checkpoint_interval == 0):
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
                noise_log_scale=None,
                event_precision_matrix=state.event_precision_matrix,
            )

    # Finalization is mostly file/diagnostic side-effects; only rank0 should run it in torchrun mode.
    if ddp_main:
        _finalize_phase1(state)
    else:
        # Still incrementally clamp/constraints happen in the loop; ensure all ranks exit cleanly.
        pass

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
                _log(f"Warning: could not rebuild event->row map after filtering: {e}")
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
        _log(f"Warning: could not write MAP locations to HDF5: {e}")
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
            _log(f"Applied max_pair_station_ratio={ratio_thr} after Phase 1; kept {after_n}/{before_n} dtimes.")
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
                    _log(f"Warning: could not rebuild event->row map after filtering: {e}")
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
        noise_log_scale=None,
        event_precision_matrix=state.event_precision_matrix,
    )


def _setup_sampler(state: LocateState) -> torch.optim.Optimizer:
    backend_name, sampler = create_sampler_backend(state.params, state)
    info(f"Sampler backend={backend_name}", section="SAMP")
    state.sampler_backend = backend_name
    state.sampler = sampler
    # Enforce JSON preconditioning flags on fresh sampler creation too.
    try:
        precond_json = bool(state.params.get("sampler_preconditioning", False))
        precond_type_json = str(state.params.get("sampler_preconditioner", "none")).strip().lower()
        if precond_json and precond_type_json in {"none", "false", ""}:
            precond_type_json = "rmsprop"
        if (not precond_json) or precond_type_json in {"none", "false", ""}:
            precond_type_json = "none"
        if sampler is not None and hasattr(sampler, "param_groups"):
            for g in sampler.param_groups:
                g["preconditioning"] = precond_json
                g["preconditioner"] = precond_type_json
        # One-time debug log to confirm resolved preconditioning settings.
        try:
            if sampler is not None and hasattr(sampler, "param_groups") and len(sampler.param_groups) > 0:
                g0 = sampler.param_groups[0]
                info(
                    "Sampler preconditioning resolved: "
                    f"enabled={bool(precond_json)} type={str(precond_type_json)} "
                    f"| group_preconditioning={g0.get('preconditioning', None)} "
                    f"group_preconditioner={g0.get('preconditioner', None)}",
                    section="SAMP",
                )
        except Exception:
            pass
    except Exception:
        pass
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
        # Clear one-time debug/runtime flags so logs show on fresh runs.
        if clear_samples_on_reset:
            ok = clear_samples_file(state.params)
            if ok:
                try:
                    sp = str(state.params.get("samples_outfile", state.params.get("io", {}).get("samples_outfile", "samples.h5")))
                except Exception:
                    sp = "samples.h5"
                info(f"Deleted samples file '{sp}' (clear_samples_on_reset=True)", section="SAMPLES")
            if not ok:
                try:
                    sp = str(state.params.get("samples_outfile", state.params.get("io", {}).get("samples_outfile", "samples.h5")))
                except Exception:
                    sp = "samples.h5"
                warn(f"Failed to delete samples file '{sp}' (clear_samples_on_reset=True). New batches may append.", section="SAMPLES")
        state.sample_count = 0
        return "phase1", 0, False, None

    ckpt = load_checkpoint(state.params, state.device)
    if ckpt is not None:
        # no-op: SSST removed entirely
        pass
    
    # Check if we should clear samples even when resuming from checkpoint
    if clear_samples_on_reset and not reset_batch_numbers:
        info("Clearing samples file while resuming from checkpoint (clear_samples_on_reset=True)", section="SAMPLES")
        ok = clear_samples_file(state.params)
        if ok:
            try:
                sp = str(state.params.get("samples_outfile", state.params.get("io", {}).get("samples_outfile", "samples.h5")))
            except Exception:
                sp = "samples.h5"
            info(f"Deleted samples file '{sp}' (clear_samples_on_reset=True)", section="SAMPLES")
        if not ok:
            try:
                sp = str(state.params.get("samples_outfile", state.params.get("io", {}).get("samples_outfile", "samples.h5")))
            except Exception:
                sp = "samples.h5"
            warn(f"Failed to delete samples file '{sp}' (clear_samples_on_reset=True). New batches may append.", section="SAMPLES")
        
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
    # Noise learning removed: ignore any checkpoint-provided noise_log_scale.
    state.stats_tensor = ckpt.get("stats_tensor", state.stats_tensor)
    # SSST removed entirely (no backward compatibility): do not load or compute SSST from checkpoints.
        
    # Adopt Hierarchical Prior P0
    try:
        epm = ckpt.get("event_precision_matrix", None)
        if epm is not None and state.hierarchical_prior_enable:
            state.event_precision_matrix = epm.to(device=state.device, dtype=torch.float32)
            _log("Resumed Hierarchical Event Precision Matrix (P0) from checkpoint.")
    except Exception as e:
        _log(f"Warning: could not adopt Hierarchical Prior P0 from checkpoint: {e}")

    # (Laplacian prior removed: ignore any Laplacian fields that might exist in old checkpoints.)

    state.samples = []
    state.global_step_count = ckpt.get("global_step_count", 0)
    state.sample_count = get_next_sample_count(state.params)
    # Clamp any resumed parameters to respect current config bounds
    try:
        _clamp_dX_inplace(state)
    except Exception as e:
        _log(f"Warning: could not clamp resumed parameters: {e}")

    # If clearing samples but not resetting batch numbers, start from phase 2
    if clear_samples_on_reset and not reset_batch_numbers:
        _log("Starting fresh from phase 2 (as if phase 1 just finished)")
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
            _log(f"Warning: could not load optimizer state: {e}")

    # For SGLD phases we will create sampler later and load then
    skip_saving_first_epoch = True
    _log(f"Resuming from phase '{phase}', next epoch/iter {start_epoch}")
    return phase, start_epoch, skip_saving_first_epoch, ckpt


def _pow2_ceil(v: int) -> int:
    x = int(v)
    if x <= 1:
        return 1
    return 1 << int((x - 1).bit_length())


def _weighted_quantile_from_hist(hist: dict[int, int], q: float) -> int:
    if not isinstance(hist, dict) or len(hist) == 0:
        return 1
    qq = min(max(float(q), 0.0), 1.0)
    items = sorted((int(k), int(v)) for k, v in hist.items() if int(v) > 0 and int(k) > 0)
    if not items:
        return 1
    total = int(sum(v for _, v in items))
    if total <= 0:
        return int(items[-1][0])
    target = int(max(1, math.ceil(qq * total)))
    acc = 0
    for n, c in items:
        acc += int(c)
        if acc >= target:
            return int(n)
    return int(items[-1][0])


def _score_bucket_plan(node_hist: dict[int, int], buckets: list[int], min_bin_groups: int) -> dict[str, float]:
    b = sorted({int(x) for x in buckets if int(x) > 0})
    if not b:
        return {"score": float("inf"), "waste": float("inf"), "leftovers": float("inf"), "underfill": float("inf")}
    largest = int(b[-1])
    bucket_counts = {int(x): 0 for x in b}
    waste = 0.0
    leftovers = 0.0
    for n_raw, c_raw in node_hist.items():
        n = int(n_raw)
        c = int(c_raw)
        if c <= 0 or n <= 0:
            continue
        if n > largest:
            leftovers += float(c)
            continue
        tgt = None
        for bn in b:
            if n <= bn:
                tgt = int(bn)
                break
        if tgt is None:
            leftovers += float(c)
            continue
        bucket_counts[tgt] = int(bucket_counts[tgt] + c)
        waste += float((tgt - n) * c)
    underfill = 0.0
    for bn in b:
        underfill += float(max(0, int(min_bin_groups) - int(bucket_counts.get(int(bn), 0))))
    score = float(waste + (float(largest) * leftovers * 4.0) + (underfill * max(1.0, float(largest) / 8.0)))
    return {
        "score": float(score),
        "waste": float(waste),
        "leftovers": float(leftovers),
        "underfill": float(underfill),
    }


def _propose_bucket_nodes_from_hist(
    *,
    node_hist: dict[int, int],
    current: list[int],
    max_nodes_cap: int,
    max_bins: int,
    min_bin_groups: int,
    min_bucket_node: int,
) -> list[int]:
    min_node = max(1, int(min_bucket_node))
    cur = sorted({int(x) for x in current if int(x) >= min_node})
    if not cur:
        cur = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    obs_max = int(max((int(k) for k, v in node_hist.items() if int(v) > 0), default=cur[-1]))
    hard_cap = int(max(max_nodes_cap, cur[-1]))
    cand = set(cur)
    for q in (0.50, 0.75, 0.90, 0.95, 0.98, 0.995):
        cand.add(_pow2_ceil(_weighted_quantile_from_hist(node_hist, q)))
    cand.add(_pow2_ceil(obs_max))
    if hard_cap > 0:
        cand = {int(min(int(x), hard_cap)) for x in cand if int(x) > 0}
    bins = sorted({int(x) for x in cand if int(x) >= min_node})
    if not bins:
        bins = cur
    while len(bins) > int(max(1, max_bins)):
        sc = _score_bucket_plan(node_hist, bins, min_bin_groups=max(1, int(min_bin_groups)))
        _ = sc
        counts = {int(x): 0 for x in bins}
        for n_raw, c_raw in node_hist.items():
            n = int(n_raw)
            c = int(c_raw)
            if c <= 0 or n <= 0:
                continue
            for bn in bins:
                if n <= int(bn):
                    counts[int(bn)] = int(counts[int(bn)] + c)
                    break
        removable = bins[:-1] if len(bins) > 1 else bins
        drop = min(removable, key=lambda x: int(counts.get(int(x), 0))) if removable else bins[0]
        bins = [int(x) for x in bins if int(x) != int(drop)]
        if not bins:
            bins = cur
            break
    # Merge very sparse bins into coarser neighbors, except the largest "whale" bin.
    while len(bins) > 1:
        counts = {int(x): 0 for x in bins}
        for n_raw, c_raw in node_hist.items():
            n = int(n_raw)
            c = int(c_raw)
            if c <= 0 or n <= 0:
                continue
            for bn in bins:
                if n <= int(bn):
                    counts[int(bn)] = int(counts[int(bn)] + c)
                    break
        sparse = [int(bn) for bn in bins[:-1] if int(counts.get(int(bn), 0)) < int(min_bin_groups)]
        if not sparse:
            break
        drop = int(min(sparse, key=lambda x: int(counts.get(int(x), 0))))
        bins = [int(x) for x in bins if int(x) != int(drop)]
    return sorted({int(x) for x in bins if int(x) >= min_node})


def _maybe_autotune_whitening_bucket_nodes(state: LocateState, epoch: int) -> None:
    p = state.params
    try:
        enabled = bool(p.get("_shared_event_re_enabled", False))
        solver = str(p.get("_shared_event_re_solver_kind", "pcg")).strip().lower()
        batched = bool(p.get("_shared_event_re_solver_batched", True))
    except Exception:
        return
    if (not enabled) or (solver != "pcg") or (not batched):
        return
    if bool(p.get("_shared_event_re_autotune_done", False)):
        return
    if not bool(p.get("_shared_event_re_autotune_enabled", True)):
        p["_shared_event_re_autotune_done"] = True
        return

    observe_epochs = max(1, int(p.get("_shared_event_re_autotune_observe_epochs", 1) or 1))
    latest_epoch = max(observe_epochs, int(p.get("_shared_event_re_autotune_latest_epoch", 2) or 2))
    e1 = int(epoch + 1)
    if e1 < observe_epochs:
        return
    if e1 > latest_epoch:
        p["_shared_event_re_autotune_done"] = True
        return

    node_hist_raw = p.get("_shared_event_re_whitening_epoch_node_hist", None)
    if not isinstance(node_hist_raw, dict) or len(node_hist_raw) == 0:
        return
    node_hist: dict[int, int] = {}
    for k, v in node_hist_raw.items():
        try:
            kk = int(k)
            vv = int(v)
            if kk > 0 and vv > 0:
                node_hist[kk] = int(node_hist.get(kk, 0) + vv)
        except Exception:
            continue
    total_groups = int(sum(int(v) for v in node_hist.values()))
    min_groups = max(1, int(p.get("_shared_event_re_autotune_min_groups", 128) or 128))
    if total_groups < min_groups:
        return

    cur_raw = p.get("_shared_event_re_solver_bucket_nodes", None)
    if isinstance(cur_raw, list) and len(cur_raw) > 0:
        cur = sorted({int(x) for x in cur_raw if int(x) > 0})
    else:
        cur = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    cur = sorted({int(x) for x in cur if int(x) > 0})
    if not cur:
        cur = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    max_nodes_cap = int(p.get("_shared_event_re_max_nodes_per_group", max(cur)) or max(cur))
    if bool(p.get("_shared_event_re_autotune_raise_nodes_cap", True)):
        obs_max = int(max(node_hist.keys()))
        cap_max = int(p.get("_shared_event_re_autotune_nodes_cap_max", 65536) or 65536)
        target_cap = int(min(obs_max, cap_max))
        if target_cap > max_nodes_cap:
            p["_shared_event_re_max_nodes_per_group"] = int(target_cap)
            max_nodes_cap = int(target_cap)
            _log(
                f"shared_event_re whitening auto-tune: max_nodes_per_group -> {int(target_cap)}",
                section="LIKELIHOOD",
            )

    max_bins = max(2, int(p.get("_shared_event_re_autotune_max_bins", 8) or 8))
    min_bin_groups = max(1, int(p.get("_shared_event_re_autotune_min_bin_groups", 24) or 24))
    min_bucket_node = max(1, int(p.get("_shared_event_re_autotune_min_bucket_node", 512) or 512))
    prop = _propose_bucket_nodes_from_hist(
        node_hist=node_hist,
        current=cur,
        max_nodes_cap=max_nodes_cap,
        max_bins=max_bins,
        min_bin_groups=min_bin_groups,
        min_bucket_node=min_bucket_node,
    )
    if not prop:
        return
    cur_score = _score_bucket_plan(node_hist, cur, min_bin_groups=min_bin_groups)
    new_score = _score_bucket_plan(node_hist, prop, min_bin_groups=min_bin_groups)
    gain = float(cur_score["score"] - new_score["score"]) / max(float(cur_score["score"]), 1e-12)
    min_gain = float(p.get("_shared_event_re_autotune_min_gain", 0.08) or 0.08)
    if prop != cur and (gain >= min_gain or float(new_score["leftovers"]) < float(cur_score["leftovers"])):
        p["_shared_event_re_solver_bucket_nodes"] = [int(x) for x in prop]
        p["_shared_event_re_autotune_done"] = True
        p["_shared_event_re_autotune_report"] = {
            "epoch": int(e1),
            "groups": int(total_groups),
            "old_buckets": list(cur),
            "new_buckets": list(prop),
            "score_old": float(cur_score["score"]),
            "score_new": float(new_score["score"]),
            "gain": float(gain),
            "leftovers_old": float(cur_score["leftovers"]),
            "leftovers_new": float(new_score["leftovers"]),
        }
        _log(
            "shared_event_re whitening auto-tune buckets "
            f"epoch={int(e1)} groups={int(total_groups)} "
            f"old={cur} new={prop} gain={100.0*gain:.1f}% "
            f"leftovers={int(cur_score['leftovers'])}->{int(new_score['leftovers'])}",
            section="LIKELIHOOD",
        )
    elif e1 >= latest_epoch:
        p["_shared_event_re_autotune_done"] = True


def _phase2_preconditioner(
    state: LocateState, start_epoch: int = 0, skip_saving_first_epoch: bool = False, wandb_logger=None
) -> None:
    _apply_likelihood_group(state, "sample")
    assert state.sampler is not None
    sampler = state.sampler
    sampler_backend = str(state.params["sampler_backend"]).strip().lower()
    ddp_main = (not _ddp_enabled(state.params)) or _ddp_is_main(state.params)
    
    # ... existing comments ...

    # Respect user intent: only force RMSProp warmup if preconditioning is enabled
    user_preconditioning = bool(state.params.get("sampler_preconditioning", False))
    user_precond_type = str(state.params.get("sampler_preconditioner", "none")).strip().lower()
    if user_preconditioning and user_precond_type in {"none", "false", ""}:
        user_precond_type = "rmsprop"
    
    for g in sampler.param_groups:
        if 'add_noise' in g:
            g['add_noise'] = False
        if 'noise_scale' in g:
            g['noise_scale'] = 0.0

        # include blockdiag_fisher (alias: matrix_ema) and non-diagonal metrics as valid Phase 2 preconditioners
        if user_preconditioning and user_precond_type in {"rmsprop", "blockdiag_fisher", "matrix_ema", "monge", "shampoo"}:
            g['preconditioner'] = user_precond_type
            g['preconditioning'] = True
            g['freeze_preconditioner'] = False
            g['is_burnin'] = True
        else:
            g['preconditioner'] = 'none'
            g['preconditioning'] = False
            g['freeze_preconditioner'] = True

    if ddp_main:
        _log(f"Phase 2: drift-only (noise=off) | {_format_sampler_status(state.sampler)}")
    for epoch in range(start_epoch, state.params["phase2_epochs"]):
        grad_clip_norm = float(state.params.get("sampler_grad_clip_norm", 0.0))
        metrics = _run_epoch(state, epoch, sampler, noise_scale_factor=0.0, grad_clip_norm=grad_clip_norm)
        if ddp_main:
            _maybe_autotune_whitening_bucket_nodes(state, epoch)
        
        # --- per-epoch summary (like phase 3 style) ---
        # Control MAD computation frequency to avoid full-dataset passes
        phase2_interval = int(state.params.get("phase2_mads_interval", 0 if state.event_batch_enable else 10))
        mad_p_val: Optional[float] = None
        mad_s_val: Optional[float] = None
        if phase2_interval > 0 and (epoch % phase2_interval == 0 or epoch == state.params["phase2_epochs"] - 1):
            mp, ms = _compute_phase_mads(state, state.batch_size_sgld)
            mad_p_val, mad_s_val = mp.item(), ms.item()

        # NOTE:
        # - We intentionally do NOT log drift_ratio_* metrics (removed; too noisy/expensive).
        # - We DO log SGHMC grad_noise_to_langevin + t_eff_var_over_target when SGHMC noise is enabled
        #   via `_sampler_extra_metrics()` (sampling diagnostics group).

        sampler_extra = {}
        try:
            sampler_extra = _sampler_extra_metrics(sampler)
        except Exception:
            sampler_extra = {}

        # Log metrics to wandb if enabled (rank0 only under torchrun)
        if ddp_main and wandb_logger and _want_wandb_group(state.params, "core"):
            wandb_metrics = {k:v for k,v in metrics.items()}
            # Noise scales (learned or fixed)
            try:
                if _want_wandb_group(state.params, "noise"):
                    σp_now, σs_now = _current_noise_scales(state)
                    wandb_metrics.update({
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
                if sampler_extra:
                    wandb_metrics.update(sampler_extra)
            wandb_logger.log_phase2_metrics(epoch, wandb_metrics, global_step=state.global_step_count)

        # Report current posterior noise scales instead of MADs (rank0 only under torchrun)
        if ddp_main:
            σp_now, σs_now = _current_noise_scales(state)
            _log(_format_epoch_line(
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

        # periodic checkpointing (no samples written in phase 2) (rank0 only under torchrun)
        checkpoint_interval = int(state.params.get("checkpoint_interval", 50))
        if ddp_main and checkpoint_interval > 0 and epoch > 0 and (epoch % checkpoint_interval == 0) and (not skip_saving_first_epoch):
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
                noise_log_scale=None,
                event_precision_matrix=state.event_precision_matrix,
            )

        if skip_saving_first_epoch:
            skip_saving_first_epoch = False

    # Save checkpoint at end of phase 2 (rank0 only under torchrun)
    phase2_last_epoch = int(state.params.get("phase2_epochs", 0)) - 1
    if phase2_last_epoch < 0:
        phase2_last_epoch = 0
    if ddp_main:
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
            noise_log_scale=None,
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
            _log("\n--- Fisher Information Matrix Diagnostics (End of Phase 2) ---")
            
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
                    _log("Events dropped. Re-computing FIM for preconditioner...")
                    fim_diag, fim_sparse = compute_block_fim(state, batch_size=4096, return_sparse=need_sparse)
                    
                    # Re-initialize the sampler optimizer completely
                    # This ensures no stale state (momentum, RMSprop) from old parameters exists.
                    _log("Re-initializing sampler optimizer due to parameter change...")
                    
                    # _setup_sampler is defined in THIS file, just call it directly.
                    # No need to import.
                    new_sampler = _setup_sampler(state)
                    # We may need to transplant state if we wanted to keep it, but here we explicitly WANT a reset.
                    # However, if Phase 2 had built up useful preconditioner stats (RMSProp), we lose them.
                    # But if we use FIM preconditioner (Matrix), it is injected below anyway.
                    # If using RMSProp, we restart warm-up in Phase 3. This is acceptable.
                    state.sampler = new_sampler
            
            _log("----------------------------------------------------------\n")

            # Install FIM as preconditioner if requested
            if use_fim:
                _log("Installing FIM-based Block-Diagonal Preconditioner for Phase 3/4...")
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
                        _log(f"FIM Preconditioner installed. Damping={damping}")
                    else:
                        _log("Warning: dX_src not found in any sampler param group.")

                except Exception as e:
                    _log(f"Error computing/installing FIM preconditioner (singular?): {e}")

    except Exception as e:
        _log(f"Warning: FIM computation failed: {e}")



def _phase3_noise_ramp(
    state: LocateState, start_epoch: int = 0, skip_saving_first_epoch: bool = False, wandb_logger=None
) -> None:
    _apply_likelihood_group(state, "sample")
    assert state.sampler is not None
    sampler = state.sampler
    ddp_main = (not _ddp_enabled(state.params)) or _ddp_is_main(state.params)
    if ddp_main:
        ddp_main = (not _ddp_enabled(state.params)) or _ddp_is_main(state.params)
        if ddp_main:
            _log(f"Phase 3: noise ramp | {_format_sampler_status(state.sampler)}")
    ramp_len = int(state.params.get("phase3_epochs", 500))

    # Re-verify FIM installation before starting Phase 3
    # If the user enabled FIM, it should be installed after Phase 2.
    # We check if 'matrix_inv' is present in the sampler state.
    has_fim = False
    try:
        p_first = state.dX_src
        if p_first in sampler.state and 'matrix_inv' in sampler.state[p_first]:
            has_fim = True
            _log("Verified: FIM Preconditioner is active for Phase 3.")
        else:
            # Check user intent
            precond_type = str(state.params["sampler_preconditioner"]).lower()
            if precond_type in {"matrix", "fim"}:
                _log("Warning: FIM requested but not found in sampler state! Re-running FIM computation...")
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
                _log("FIM Preconditioner installed (late).")
    except Exception as e:
        _log(f"Warning checking FIM status: {e}")

    # Set constant learning rate (per-observation scaling).
    lr_user = _lr_for_phase(state.params, "phase3")
    sampler_backend = str(state.params["sampler_backend"]).lower()
    if sampler_backend in {"psgld", "sghmc"}:
        base_lr = lr_user / float(max(1, int(state.N)))
    else:
        base_lr = lr_user
    sampler.set_lr(base_lr)
    _apply_sampler_group_overrides(state, sampler)
    
    for g in sampler.param_groups:
        # Phase 3 is burn-in / noise-ramp. Allow the preconditioner to adapt here.
        g["freeze_preconditioner"] = False
        g["is_burnin"] = True
    ddp_main = (not _ddp_enabled(state.params)) or _ddp_is_main(state.params)
    if ddp_main:
        _log(f"Phase 3: noise ramp | {_format_sampler_status(sampler)}")

    for t in range(start_epoch, ramp_len):
        progress = float(min(1.0, (t + 1) / float(ramp_len)))

        # Optional extra noise multiplier (applied in both Phase 3 and Phase 4).
        try:
            noise_mult = float(state.params.get("sampler_noise_scale_mult", 1.0) or 1.0)
            if (not math.isfinite(noise_mult)) or (noise_mult <= 0.0):
                noise_mult = 1.0
        except Exception:
            noise_mult = 1.0
        progress = float(progress) * float(noise_mult)
        
        grad_clip_norm = float(state.params.get("sampler_grad_clip_norm", 0.0))
        metrics = _run_epoch(state, t, sampler, noise_scale_factor=progress, grad_clip_norm=grad_clip_norm)

        # --- per-iteration (ramp step) summary ---
        phase3_interval = int(state.params.get("phase3_mads_interval", 0 if state.event_batch_enable else 10))
        mad_p_val: Optional[float] = None
        mad_s_val: Optional[float] = None
        if phase3_interval > 0 and (t % phase3_interval == 0 or t == ramp_len - 1):
            mp, ms = _compute_phase_mads(state, state.batch_size_sgld)
            mad_p_val, mad_s_val = mp.item(), ms.item()

        # See note in Phase 2: drift_ratio_* is removed; SGHMC teff/noise diagnostics are logged only when noise is on.
        tau_mean = float('nan')
        tau_med = float('nan')
        if wandb_logger and _want_wandb_group(state.params, "sampler"):
            try:
                if hasattr(sampler, "tau_stats"):
                    taustats = sampler.tau_stats()  # type: ignore[attr-defined]
                    tau_mean = float(taustats.get("mean", float("nan")))
                    tau_med = float(taustats.get("median", float("nan")))
            except Exception:
                pass

        # Log metrics to wandb if enabled (rank0 only under torchrun)
        if ddp_main and wandb_logger and _want_wandb_group(state.params, "core"):
            wandb_metrics = {k:v for k,v in metrics.items()}
            # Noise scales (learned or fixed)
            try:
                if _want_wandb_group(state.params, "noise"):
                    σp_now, σs_now = _current_noise_scales(state)
                    wandb_metrics.update({
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
                # tau_* only exists for samplers that expose tau_stats(); skip NaNs.
                _wb_add_if_finite(wandb_metrics, "tau_mean", tau_mean)
                _wb_add_if_finite(wandb_metrics, "tau_med", tau_med)
                wandb_metrics.update(_sampler_extra_metrics(sampler))
            wandb_logger.log_phase3_metrics(t, wandb_metrics, global_step=state.global_step_count)

        # Report current posterior noise scales instead of MADs (rank0 only under torchrun)
        if ddp_main:
            σp_now, σs_now = _current_noise_scales(state)
            _log(_format_epoch_line(
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

        # periodic checkpointing during phase 3 ramp (no samples written) (rank0 only under torchrun)
        checkpoint_interval = int(state.params.get("checkpoint_interval", 50))
        if ddp_main and checkpoint_interval > 0 and t > 0 and (t % checkpoint_interval == 0) and (not skip_saving_first_epoch):
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
                noise_log_scale=None,
                event_precision_matrix=state.event_precision_matrix,
            )

        if skip_saving_first_epoch:
            skip_saving_first_epoch = False

    # Save checkpoint at end of phase 3 (rank0 only under torchrun)
    phase3_last_iter = int(ramp_len) - 1
    if phase3_last_iter < 0:
        phase3_last_iter = 0
    if ddp_main:
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
            noise_log_scale=None,
            event_precision_matrix=state.event_precision_matrix,
        )


def _phase4_sampling(
    state: LocateState, start_epoch: int, skip_saving_first_epoch: bool, wandb_logger=None
) -> None:
    _apply_likelihood_group(state, "sample")
    assert state.sampler is not None
    sampler = state.sampler
    ddp_main = (not _ddp_enabled(state.params)) or _ddp_is_main(state.params)
    # Ensure LR is set from config (per-observation scaling).
    try:
        lr_user = _lr_for_phase(state.params, "phase4")
        backend = str(state.params["sampler_backend"]).strip().lower()
        if backend in {"psgld", "sghmc"}:
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
        ddp_main = (not _ddp_enabled(state.params)) or _ddp_is_main(state.params)
        if ddp_main:
            _log("Phase 4: sampling skipped (phase4_epochs=0)")
        return
    if int(start_epoch) >= int(n_epochs):
        ddp_main = (not _ddp_enabled(state.params)) or _ddp_is_main(state.params)
        if ddp_main:
            _log(f"Phase 4: sampling skipped (start_epoch={start_epoch} >= phase4_epochs={n_epochs})")
        return

    freeze_precond = bool(state.params.get("freeze_preconditioner_sampling", True))
    for g in sampler.param_groups:
        g["freeze_preconditioner"] = freeze_precond
        g["is_burnin"] = False
    _apply_sampler_group_overrides(state, sampler)
    ddp_main = (not _ddp_enabled(state.params)) or _ddp_is_main(state.params)
    if ddp_main:
        # Production-sampling guardrails:
        # - Deterministic (unshuffled) batching in Phase 4 can create long-term trends because the
        #   gradient-noise process becomes periodic rather than i.i.d.
        # - Leaving the preconditioner unfrozen in Phase 4 makes the Markov kernel time-inhomogeneous.
        try:
            if not bool(state.params.get("batch_shuffle", True)):
                warn(
                    "Phase 4: batching.shuffle=false (deterministic minibatch order). "
                    "For production posterior sampling, enable inference.batching.standard.shuffle=true "
                    "to avoid periodic gradient-noise artifacts / long-term trends.",
                    section="SAMPLER",
                )
        except Exception:
            pass
        try:
            precond_on = bool(state.params.get("sampler_preconditioning", True))
            if precond_on and (not bool(freeze_precond)):
                warn(
                    "Phase 4: freeze_preconditioner_sampling=false while preconditioning is enabled. "
                    "This makes the sampling kernel time-inhomogeneous and can look like continued optimization. "
                    "For production posterior samples, set inference.sampler.freeze_preconditioner_sampling=true.",
                    section="SAMPLER",
                )
        except Exception:
            pass
        try:
            if bool(state.params.get("sampler_preconditioning", True)) and int(state.params.get("phase2_epochs", 0) or 0) <= 0:
                warn(
                    "Phase 4: phase2_epochs=0 while preconditioning is enabled. "
                    "Consider running a nonzero Phase 2 to stabilize RMSProp statistics before sampling "
                    "(then freeze in Phase 4).",
                    section="SAMPLER",
                )
        except Exception:
            pass
        _log(f"Phase 4: sampling | {_format_sampler_status(sampler)}")

    # Track relative parameter changes over the last N and N2 epochs
    rel_window = int(state.params.get("rel_change_window", 10))
    rel_window2 = int(state.params.get("rel_change_window2", 50))
    param_snapshots = collections.deque(maxlen=max(rel_window, rel_window2) + 1)

    # --- lightweight drift diagnostics (subset + epoch-level batch-means t-test) ---
    # Goal: detect systematic long-term drift in internal modes (e.g., coherent warps) during Phase-4 sampling.
    # We track per-epoch increments of mean-centered dX_z on a fixed subset of events and compute a rolling
    # t-statistic across the last W epochs. This is cheap (O(W*subset_n)) and GPU-friendly.
    drift_subset_n_default = 2048
    drift_window_default = 50
    drift_min_blocks_default = 10
    try:
        drift_subset_n = int(state.params.get("_runtime_drift_subset_n", drift_subset_n_default) or drift_subset_n_default)
        drift_window = int(state.params.get("_runtime_drift_window_epochs", drift_window_default) or drift_window_default)
        drift_min_blocks = int(state.params.get("_runtime_drift_min_blocks", drift_min_blocks_default) or drift_min_blocks_default)
        drift_subset_n = max(64, drift_subset_n)
        drift_window = max(5, drift_window)
        drift_min_blocks = max(5, min(drift_window, drift_min_blocks))
    except Exception:
        drift_subset_n, drift_window, drift_min_blocks = drift_subset_n_default, drift_window_default, drift_min_blocks_default

    if not hasattr(state, "_runtime_drift_tracker"):
        try:
            Ne = int(state.dX_src.shape[0])
            n_sub = min(int(drift_subset_n), int(Ne))
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(state.params.get("runtime_seed", 0) or 0) + 1337)
            idx_cpu = torch.randperm(Ne, generator=gen, device="cpu")[:n_sub].to(torch.int64)
            idx = idx_cpu.to(device=state.dX_src.device, non_blocking=True)
            setattr(state, "_runtime_drift_tracker", {
                "idx": idx,
                "prev": None,
                "deltas": collections.deque(maxlen=int(drift_window)),
                "n_sub": int(n_sub),
                "window": int(drift_window),
                "min_blocks": int(drift_min_blocks),
            })
        except Exception:
            setattr(state, "_runtime_drift_tracker", None)

    @torch.no_grad()
    def _drift_centered_dz_subset() -> Optional[torch.Tensor]:
        tr = getattr(state, "_runtime_drift_tracker", None)
        if not isinstance(tr, dict):
            return None
        idx = tr.get("idx", None)
        if not isinstance(idx, torch.Tensor) or idx.numel() == 0:
            return None
        try:
            dz = state.dX_src.detach()[:, 2].to(dtype=torch.float32)  # (Ne,)
            zsub = dz.index_select(0, idx)  # (n_sub,)
            zsub = zsub - zsub.mean()
            return zsub
        except Exception:
            return None

    for epoch in range(int(start_epoch), int(n_epochs)):
        # Drift tracker: capture state at epoch start (for delta over this epoch)
        try:
            tr = getattr(state, "_runtime_drift_tracker", None)
            if isinstance(tr, dict) and tr.get("prev", None) is None:
                tr["prev"] = _drift_centered_dz_subset()
        except Exception:
            pass

        metrics = _run_epoch(
            state,
            epoch,
            sampler,
            is_sampling=True,
            noise_scale_factor=float(state.params.get("sampler_noise_scale_mult", 1.0) or 1.0),
            grad_clip_norm=float(state.params.get("sampler_grad_clip_norm", 0.0)),
        )

        # Drift tracker: update deltas and compute rolling t-stats on a fixed subset (rank0 logs below).
        drift_metrics: Dict[str, float] = {}
        try:
            tr = getattr(state, "_runtime_drift_tracker", None)
            if isinstance(tr, dict):
                prev = tr.get("prev", None)
                cur = _drift_centered_dz_subset()
                if isinstance(prev, torch.Tensor) and isinstance(cur, torch.Tensor) and prev.shape == cur.shape:
                    d = (cur - prev).detach().to(device="cpu", dtype=torch.float32)
                    deltas = tr.get("deltas", None)
                    if isinstance(deltas, collections.deque):
                        deltas.append(d)
                    tr["prev"] = cur
                # Compute rolling stats (only when we have enough blocks)
                deltas = tr.get("deltas", None)
                if isinstance(deltas, collections.deque) and len(deltas) >= int(tr.get("min_blocks", 10)):
                    D = torch.stack(list(deltas), dim=0)  # (B, n_sub)
                    B = int(D.shape[0])
                    mean = D.mean(dim=0)
                    std = D.std(dim=0, unbiased=True).clamp_min(1e-30)
                    tstat = mean / (std / float(max(1, B)) ** 0.5)
                    abs_t = tstat.abs()
                    drift_metrics = {
                        "drift_z/frac_abs_t_gt3": float((abs_t > 3.0).float().mean().item()),
                        "drift_z/frac_abs_t_gt10": float((abs_t > 10.0).float().mean().item()),
                        "drift_z/abs_t_p99": float(torch.quantile(abs_t, 0.99).item()),
                        "drift_z/abs_t_p999": float(torch.quantile(abs_t, 0.999).item()),
                        "drift_z/window_epochs": float(B),
                        "drift_z/subset_n": float(int(tr.get("n_sub", int(D.shape[1])))),
                    }
        except Exception:
            drift_metrics = {}

        # --- per-epoch summary ---
        phase4_interval = int(state.params.get("phase4_mads_interval", 0 if state.event_batch_enable else 10))
        mad_p_val: Optional[float] = None
        mad_s_val: Optional[float] = None
        if phase4_interval > 0 and (epoch % phase4_interval == 0 or epoch == n_epochs - 1):
            mp, ms = _compute_phase_mads(state, state.batch_size_sgld)
            mad_p_val, mad_s_val = mp.item(), ms.item()

        # Sampler diagnostics (noise variance ratios, etc.)
        sampler_extra: Dict[str, float] = {}
        try:
            sampler_extra = _sampler_extra_metrics(sampler)
        except Exception:
            sampler_extra = {}

        # See note in Phase 2: drift_ratio_* is removed; SGHMC teff/noise diagnostics are logged only when noise is on.
        tau_mean = float('nan')
        tau_med = float('nan')
        if wandb_logger and _want_wandb_group(state.params, "sampler"):
            try:
                if hasattr(sampler, "tau_stats"):
                    taustats = sampler.tau_stats()  # type: ignore[attr-defined]
                    tau_mean = float(taustats.get("mean", float("nan")))
                    tau_med = float(taustats.get("median", float("nan")))
            except Exception:
                pass

        # Log metrics to wandb if enabled (rank0 only under torchrun)
        if ddp_main and wandb_logger and _want_wandb_group(state.params, "core"):
            wandb_metrics = {k:v for k,v in metrics.items()}
            # Noise scales (learned or fixed)
            try:
                if _want_wandb_group(state.params, "noise"):
                    σp_now, σs_now = _current_noise_scales(state)
                    wandb_metrics.update({
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
                # tau_* only exists for samplers that expose tau_stats(); skip NaNs.
                _wb_add_if_finite(wandb_metrics, "tau_mean", tau_mean)
                _wb_add_if_finite(wandb_metrics, "tau_med", tau_med)
                if sampler_extra:
                    wandb_metrics.update(sampler_extra)
                # Drift diagnostics (subset + rolling epoch-level batch-means test)
                if drift_metrics:
                    wandb_metrics.update(drift_metrics)
            wandb_logger.log_phase4_metrics(epoch, wandb_metrics, global_step=state.global_step_count)

        # Compact console line (rank0 only under torchrun)
        if ddp_main:
            σp_now, σs_now = _current_noise_scales(state)
            _log(_format_epoch_line(
                phase="phase4",
                step=epoch + 1,
                total=int(n_epochs),
                metrics=metrics,
                opt=sampler,
                extra=f"sigma_p={float(σp_now):.4f} sigma_s={float(σs_now):.4f}",
            ))
            _shift_guard_check(state, context=f"phase4 epoch {epoch}")

        # Periodic checkpointing (rank0 only under torchrun)
        checkpoint_interval = int(state.params.get("checkpoint_interval", 50))
        if ddp_main and checkpoint_interval > 0 and epoch > 0 and (epoch % checkpoint_interval == 0) and (not skip_saving_first_epoch):
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
                noise_log_scale=None,
                event_precision_matrix=state.event_precision_matrix,
            )

        # Periodic sample flush to HDF5 (rank0 only under torchrun)
        # Historically, we flushed samples at checkpoint cadence. `sample_write_interval`
        # allows these to be decoupled (defaults to checkpoint_interval when omitted).
        sample_write_interval = int(state.params.get("sample_write_interval", checkpoint_interval))
        write_samples = bool(state.params.get("write_samples", True))
        if (
            ddp_main
            and write_samples
            and sample_write_interval > 0
            and epoch > 0
            and (epoch % sample_write_interval == 0)
            and (not skip_saving_first_epoch)
        ):
            state.sample_count = save_samples_periodic(
                state.params,
                state.origins0,
                state.X_src,
                state.samples,
                state.projector,
                state.sample_count,
                noise_log_scales=None,
                global_step_count=int(state.global_step_count),
                epoch=int(epoch),
                phase="phase4",
            )
            state.samples = []

        if skip_saving_first_epoch:
            skip_saving_first_epoch = False

    # Final flush + checkpoint (rank0 only under torchrun)
    if ddp_main:
        last_epoch = int(n_epochs) - 1
        if last_epoch < 0:
            last_epoch = 0
        if bool(state.params.get("write_samples", True)):
            state.sample_count = save_samples_periodic(
                state.params,
                state.origins0,
                state.X_src,
                state.samples,
                state.projector,
                state.sample_count,
                noise_log_scales=None,
                global_step_count=int(state.global_step_count),
                epoch=int(last_epoch),
                phase="phase4",
            )
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
            noise_log_scale=None,
            event_precision_matrix=state.event_precision_matrix,
        )  # type: ignore[arg-type]
    return


def _apply_sampler_group_overrides(state: "LocateState", sampler: Optional[torch.optim.Optimizer]) -> None:
    """
    This is necessary because some call sites overwrite lr/temperature for *all* groups.
    """
    if sampler is None:
        return
    # If the user did not explicitly provide any overrides, do NOT touch group hyperparams.
    # This avoids surprising behavior and ensures global flags like sampler.freeze_preconditioner_sampling
    # apply uniformly to all parameter groups.
    if not bool(state.params.get("_sampler_group_overrides_active", False)):
        return
    group_overrides = state.params.get("_sampler_group_overrides", {})
    if not isinstance(group_overrides, dict) or not group_overrides:
        return

    for g in sampler.param_groups:
        gname = str(g.get("group_name", "")).strip().lower()
        if not gname:
            continue
        ov = group_overrides.get(gname, None)
        if not isinstance(ov, dict) or not ov:
            continue
        try:
            # Keep base lr/temperature as whatever caller set, then apply multipliers.
            # We store base values on the group to avoid compounding.
            if "lr_mult" in ov:
                if "base_lr" not in g:
                    g["base_lr"] = float(g.get("lr", 0.0))
                g["lr"] = float(g["base_lr"]) * float(ov["lr_mult"])
            if "temperature_mult" in ov:
                if "base_temperature" not in g:
                    g["base_temperature"] = float(g.get("temperature", 1.0))
                g["temperature"] = float(g["base_temperature"]) * float(ov["temperature_mult"])
            # Stabilize RMSProp preconditioner/noise amplification
            if "eps" in ov:
                g["eps"] = float(ov["eps"])
            # Do not clobber the global freeze flag; if the run is in a frozen phase (Phase 4),
            # keep it frozen even if this group override isn't requesting freezing.
            if "freeze_preconditioner_sampling" in ov:
                g["freeze_preconditioner"] = bool(g.get("freeze_preconditioner", False)) or bool(ov["freeze_preconditioner_sampling"])
        except Exception:
            pass
    return


@torch.no_grad()
#     """

#     Given a per-component inducing selection (Stage 2), build a sparse interpolation structure:
#       - for each event e, choose its nearest m inducing points within its component
#       - store (global inducing index, kernel value) pairs

#     This does NOT change inference. It writes an .npz for inspection and later integration.
#     """
#     try:
#             return
#             return
#             return
#         if not (ell > 0.0):
#             return
#         m = max(1, m)
#         if out_path is None or str(out_path).strip() == "":
#     except Exception:
#         return

#     # Load selection arrays from memory or from file
#     if comp_ids is None or comp_off is None or inducing_event_idx is None:
#         # Try to load from NPZ file
#         if sel_path is None:
#         if sel_path is None:
#             warn("Inducing interpolation requested but no inducing selection is available; enable inducing_plan.select first.", section="LIKELIHOOD")
#             return
#         try:
#             sel_path_s = str(sel_path)
#             if not os.path.isabs(sel_path_s):
#                 base = str(state.params.get("checkpoint_dir", "."))
#                 sel_path_s = os.path.join(base, sel_path_s)
#             data = np.load(sel_path_s, allow_pickle=True)
#             comp_ids = data["component_id"]
#             comp_off = data["component_offsets"]
#             inducing_event_idx = data["inducing_event_idx"]
#             if "inducing_xyz_km" in data:
#                 inducing_xyz_km = data["inducing_xyz_km"]
#             # prefer runtime use_xyz from file if present
#             if "use_xyz" in data:
#                 try:
#                     use_xyz = bool(int(np.asarray(data["use_xyz"]).reshape(-1)[0]))
#                 except Exception:
#                     pass
#         except Exception as e:
#             warn(f"Inducing interpolation requested but could not load selection NPZ: {e}", section="LIKELIHOOD")
#             return

#     try:
#         comp_ids = np.asarray(comp_ids, dtype=np.int64)
#         comp_off = np.asarray(comp_off, dtype=np.int64)
#         inducing_event_idx = np.asarray(inducing_event_idx, dtype=np.int64)
#         if inducing_xyz_km is not None:
#             inducing_xyz_km = np.asarray(inducing_xyz_km, dtype=np.float32)
#     except Exception:
#         return
#     if comp_ids.size == 0 or comp_off.size != comp_ids.size + 1 or inducing_event_idx.size == 0:
#         return

#     # Event -> component mapping
#     if getattr(state, "cluster_ids", None) is None:
#         return
#     try:
#         ev_comp = state.cluster_ids.detach().cpu().numpy().astype(np.int64, copy=False)
#     except Exception:
#         return
#     n_events = int(ev_comp.shape[0])

#     # MAP coordinates in km (XY or XYZ)
#     try:
#         X_map = (state.X_src.detach() + state.dX_src.detach())[:, :3].to(torch.float32).cpu().numpy()
#     except Exception:
#         return
#     if X_map.shape[0] != n_events:
#         return
#     if use_xyz:
#         P = X_map[:, :3].astype(np.float32, copy=False)
#         dim_label = "xyz"
#     else:
#         P = X_map[:, :2].astype(np.float32, copy=False)
#         dim_label = "xy"

#     # Build a mapping from component id -> block index in comp_ids
#     n_comp_total = int(max(ev_comp.max() + 1, comp_ids.max() + 1)) if n_events > 0 else int(comp_ids.max() + 1)
#     comp_to_block = np.full((n_comp_total,), -1, dtype=np.int64)
#     for bi, cid in enumerate(comp_ids.tolist()):
#         if cid >= 0 and cid < comp_to_block.shape[0]:
#             comp_to_block[int(cid)] = int(bi)

#     # Output arrays (fixed m per event; padded with -1/0)
#     neigh_idx = np.full((n_events, m), -1, dtype=np.int64)
#     neigh_k = np.zeros((n_events, m), dtype=np.float32)
#     # Precompute Matérn(3/2) kernel values at MAP for each event->neighbor inducing pair.
#     neigh_k_matern32 = np.zeros((n_events, m), dtype=np.float32)
#     neigh_d = np.zeros((n_events, m), dtype=np.float32) if store_dist else None

#     t0 = time.time()
#     use_scipy = True

#     for bi, cid in enumerate(comp_ids.tolist()):
#         i0 = int(comp_off[bi]); i1 = int(comp_off[bi + 1])
#         if i1 <= i0:
#             continue
#         # Inducing locations for this component. Prefer fixed inducing XYZ if available.
#         Uc = None
#         try:
#             if inducing_xyz_km is not None and getattr(inducing_xyz_km, "ndim", 0) == 2 and int(inducing_xyz_km.shape[1]) >= int(P.shape[1]):
#                 Uc = np.asarray(inducing_xyz_km[i0:i1, : int(P.shape[1])]).astype(np.float32, copy=False)
#         except Exception:
#             Uc = None
#         if Uc is None:
#             U_ev = inducing_event_idx[i0:i1]
#             Uc = P[U_ev]
#         M = int(Uc.shape[0])
#         if M <= 0:
#             continue
#         # All events in this component
#         ev_idx = np.flatnonzero(ev_comp == int(cid)).astype(np.int64, copy=False)
#         if ev_idx.size == 0:
#             continue
#         Pc = P[ev_idx]
#         kq = int(min(m, M))
#         if kq <= 0:
#             continue
#         # Query nearest inducing points
#         try:
#             from scipy.spatial import cKDTree  # type: ignore
#             tree = cKDTree(Uc.astype("float64", copy=False))
#             try:
#                 dists, nbrs = tree.query(Pc.astype("float64", copy=False), k=kq, workers=-1)
#             except TypeError:
#                 try:
#                     dists, nbrs = tree.query(Pc.astype("float64", copy=False), k=kq, n_jobs=-1)  # type: ignore[call-arg]
#                 except TypeError:
#                     dists, nbrs = tree.query(Pc.astype("float64", copy=False), k=kq)
#             # Normalize shapes to (n,kq)
#             if kq == 1:
#                 dists = np.asarray(dists).reshape(-1, 1)
#                 nbrs = np.asarray(nbrs).reshape(-1, 1)
#         except Exception:
#             use_scipy = False
#             # Fallback: brute force with chunking (can be slow for large components)
#             Uc_f = Uc.astype(np.float32, copy=False)
#             Pc_f = Pc.astype(np.float32, copy=False)
#             nbrs = np.empty((Pc_f.shape[0], kq), dtype=np.int64)
#             dists = np.empty((Pc_f.shape[0], kq), dtype=np.float32)
#             chunk = 8192
#             for s0 in range(0, Pc_f.shape[0], chunk):
#                 s1 = min(s0 + chunk, Pc_f.shape[0])
#                 Q = Pc_f[s0:s1]  # (B,d)
#                 # (B,M) squared distances
#                 # Use (x-y)^2 = x^2 + y^2 - 2 x y for speed
#                 q2 = (Q * Q).sum(axis=1, keepdims=True)
#                 u2 = (Uc_f * Uc_f).sum(axis=1, keepdims=True).T
#                 d2 = q2 + u2 - 2.0 * (Q @ Uc_f.T)
#                 d2 = np.maximum(d2, 0.0)
#                 # partial sort
#                 part = np.argpartition(d2, kth=kq - 1, axis=1)[:, :kq]
#                 d2_part = np.take_along_axis(d2, part, axis=1)
#                 ord2 = np.argsort(d2_part, axis=1)
#                 part_sorted = np.take_along_axis(part, ord2, axis=1)
#                 d2_sorted = np.take_along_axis(d2_part, ord2, axis=1)
#                 nbrs[s0:s1, :] = part_sorted
#                 dists[s0:s1, :] = np.sqrt(d2_sorted).astype(np.float32, copy=False)

#         # Convert to global inducing indices [0..sum_M)
#         gidx = (np.asarray(nbrs, dtype=np.int64) + int(i0)).astype(np.int64, copy=False)
#         # Kernel values (RBF)
#         d_f = np.asarray(dists, dtype=np.float32)
#         k_val = np.exp(-0.5 * (d_f / float(ell)) ** 2).astype(np.float32, copy=False)

#         neigh_idx[ev_idx, :kq] = gidx
#         neigh_k[ev_idx, :kq] = k_val
#         if store_dist and neigh_d is not None:
#             neigh_d[ev_idx, :kq] = d_f

#     dt_s = time.time() - t0
#     info(
#         f"ell_km={ell:g} backend={'scipy_ckdtree' if use_scipy else 'bruteforce'} dt={dt_s:.1f}s",
#         section="LIKELIHOOD",
#     )

#     # Keep interpolation in memory for inducing modes so inference does not need NPZ round-trips.
#     # (We may still write NPZ below for reproducibility / restarts.)
#     try:
#         if mode in {"inducing_gp", "slowness_inducing_gp"}:
#             dev = state.device
#             try:
#             except Exception:
#             try:
#                 if inducing_xyz_km is not None:
#                 else:
#             except Exception:
#     except Exception:
#         pass

#     # Write NPZ (relative -> checkpoint_dir)
#     try:
#         out_path_s = str(out_path)
#         if not os.path.isabs(out_path_s):
#             base = str(state.params.get("checkpoint_dir", "."))
#             out_path_s = os.path.join(base, out_path_s)
#         os.makedirs(os.path.dirname(out_path_s) or ".", exist_ok=True)
#         payload = dict(
#             component_id=comp_ids,
#             component_offsets=comp_off,
#             inducing_event_idx=inducing_event_idx,
#             inducing_xyz_km=(inducing_xyz_km if inducing_xyz_km is not None else np.zeros((0, 3), dtype=np.float32)),
#             event_component_id=ev_comp.astype(np.int64, copy=False),
#             neighbor_inducing_global_idx=neigh_idx,
#             neighbor_kernel=neigh_k,
#             ell_km=np.asarray([float(ell)], dtype=np.float32),
#             use_xyz=np.asarray([int(bool(use_xyz))], dtype=np.int8),
#         )
#         if store_dist and neigh_d is not None:
#             payload["neighbor_dist_km"] = neigh_d
#         np.savez_compressed(out_path_s, **payload)
#         info(f"Wrote inducing interpolation NPZ: {out_path_s}", section="LIKELIHOOD")
#         try:
#         except Exception:
#             pass
#     except Exception as e:
#         warn(f"Could not write inducing interpolation NPZ: {e}", section="LIKELIHOOD")
#     return


# @torch.no_grad()
#     """
#     Stage-2: Select inducing points per connected component for the collapsed slowness covariance likelihood.

#     """
#     try:
#             return
#             return
#         if not (ell > 0.0):
#             return
#         if not (cover_frac > 0.0) or (not math.isfinite(cover_frac)):
#             cover_frac = 1.0
#         r = float(cover_frac) * float(ell)
#         top_k = max(1, top_k)
#         min_m = max(1, min_m)
#         max_m = max(min_m, max_m)
#         if seed_strategy not in {"max_degree", "random"}:
#             seed_strategy = "max_degree"
#     except Exception:
#         return

#     # If selection already present (resume), do nothing.
#     try:
#             return
#     except Exception:
#         pass

#     if getattr(state, "cluster_ids", None) is None or getattr(state, "cluster_counts", None) is None:
#         return
#     try:
#         comp = state.cluster_ids.detach().cpu().numpy().astype(np.int64, copy=False)
#         counts = state.cluster_counts.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
#     except Exception:
#         return
#     if comp.size == 0 or counts.size == 0:
#         return
#     n_comp = int(counts.size)

#     # MAP event XYZ in km
#     try:
#         X_map = (state.X_src.detach() + state.dX_src.detach())[:, :3].to(torch.float32).cpu().numpy()
#     except Exception:
#         return
#     if X_map.shape[0] != comp.shape[0]:
#         return
#     P = X_map[:, :3].astype(np.float32, copy=False)

#     # Optional degree for linkage-aware seeding
#     deg = None
#     try:
#         if getattr(state, "dd_event_degree", None) is not None:
#             deg = state.dd_event_degree.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
#             if deg.shape[0] != comp.shape[0]:
#                 deg = None
#     except Exception:
#         deg = None

#     rng = np.random.default_rng(int(state.params.get("runtime_seed", 0)))
#     t0 = time.time()

#     comp_ids = np.arange(n_comp, dtype=np.int64)
#     valid = counts > 0
#     comp_ids = comp_ids[valid]
#     comp_ids = comp_ids[np.argsort(-counts[comp_ids])]

#     inducing_idx_all: list[np.ndarray] = []
#     inducing_off = [0]
#     comp_out = []
#     m_out = []
#     cov_out = []
#     for ci in comp_ids.tolist():
#         idxs = np.flatnonzero(comp == int(ci)).astype(np.int64, copy=False)
#         if idxs.size == 0:
#             continue
#         m_cap = int(min(max_m, int(idxs.size)))
#         if seed_strategy == "max_degree" and deg is not None:
#             seed_local = int(np.argmax(deg[idxs]))
#         else:
#             seed_local = int(rng.integers(0, int(idxs.size)))
#         Pc = P[idxs]  # (n,3)
#         sel_local = [seed_local]
#         d0 = Pc - Pc[seed_local]
#         min_d2 = (d0 * d0).sum(axis=1).astype(np.float32, copy=False)
#         while len(sel_local) < m_cap:
#             max_d2 = float(min_d2.max()) if min_d2.size > 0 else 0.0
#             if math.sqrt(max_d2) <= r:
#                 break
#             j = int(np.argmax(min_d2))
#             if j in sel_local:
#                 break
#             sel_local.append(j)
#             dj = Pc - Pc[j]
#             d2 = (dj * dj).sum(axis=1).astype(np.float32, copy=False)
#             min_d2 = np.minimum(min_d2, d2)
#         max_dist = math.sqrt(float(min_d2.max())) if min_d2.size > 0 else 0.0
#         sel_global = idxs[np.asarray(sel_local, dtype=np.int64)]
#         inducing_idx_all.append(sel_global.astype(np.int64, copy=False))
#         inducing_off.append(int(inducing_off[-1] + int(sel_global.size)))
#         comp_out.append(int(ci))
#         m_out.append(int(sel_global.size))
#         cov_out.append(float(max_dist))

#     if not inducing_idx_all:
#         return

#     inducing_idx = np.concatenate(inducing_idx_all, axis=0).astype(np.int64, copy=False)
#     inducing_xyz_km = X_map[inducing_idx, :3].astype(np.float32, copy=False) if inducing_idx.size > 0 else np.zeros((0, 3), dtype=np.float32)
#     offsets = np.asarray(inducing_off, dtype=np.int64)
#     comp_out_a = np.asarray(comp_out, dtype=np.int64)
#     m_out_a = np.asarray(m_out, dtype=np.int64)
#     cov_out_a = np.asarray(cov_out, dtype=np.float32)

#     dt_s = time.time() - t0
#     info(
#         f"ell_km={ell:g} cover_r={r:g} dims=xyz fixed_xyz={int(bool(fixed_xyz))} seed={seed_strategy} dt={dt_s:.1f}s",
#         section="LIKELIHOOD",
#     )
#     order = np.argsort(-counts[comp_out_a])
#     for j in range(int(min(int(top_k), int(order.size)))):
#         ci = int(comp_out_a[order[j]])
#         info(
#             f"  comp[{j}] id={ci} n={int(counts[ci]):,} M={int(m_out_a[order[j]])} "
#             f"max_dist_to_inducing={float(cov_out_a[order[j]]):.3g}km (target_r={r:.3g}km)",
#             section="LIKELIHOOD",
#         )

#     # Always expose selection in-memory for runtime use; optionally persist to NPZ if requested.
#     try:
#     except Exception:
#         pass

#     try:
#         want_write = bool(out_path is not None and str(out_path).strip() != "")
#     except Exception:
#         want_write = False
#     if want_write:
#         try:
#             out_path_s = str(out_path)
#             if not os.path.isabs(out_path_s):
#                 base = str(state.params.get("checkpoint_dir", "."))
#                 out_path_s = os.path.join(base, out_path_s)
#             os.makedirs(os.path.dirname(out_path_s) or ".", exist_ok=True)
#             np.savez_compressed(
#                 out_path_s,
#                 component_id=comp_out_a,
#                 component_n_events=counts[comp_out_a].astype(np.int64, copy=False),
#                 component_offsets=offsets,
#                 inducing_event_idx=inducing_idx,
#                 inducing_xyz_km=inducing_xyz_km,
#                 ell_km=np.asarray([float(ell)], dtype=np.float32),
#                 cover_r_km=np.asarray([float(r)], dtype=np.float32),
#                 fixed_xyz=np.asarray([int(bool(fixed_xyz))], dtype=np.int8),
#                 seed_strategy=np.asarray([seed_strategy], dtype=object),
#                 max_dist_to_inducing_km=cov_out_a,
#             )
#         except Exception as e:


# @torch.no_grad()
#     """

#     We store neighbor indices on-device for runtime use; kernel values in the NPZ (if written)
#     """
#     try:
#             return
#             return
#             return
#         if not (ell > 0.0):
#             return
#         m = max(1, m)
#     except Exception:
#         return

#     # If neighbor idx already present, do nothing (but only if Matérn weights are also available).
#         return
#         try:
#             return
#         except Exception:
#             pass

#     # Load selection arrays from memory or file
#     if comp_ids is None or comp_off is None or inducing_event_idx is None:
#         if sel_path is None:
#         if sel_path is None:
#             return
#         try:
#             sel_path_s = str(sel_path)
#             if not os.path.isabs(sel_path_s):
#                 base = str(state.params.get("checkpoint_dir", "."))
#                 sel_path_s = os.path.join(base, sel_path_s)
#             data = np.load(sel_path_s, allow_pickle=True)
#             comp_ids = data["component_id"]
#             comp_off = data["component_offsets"]
#             inducing_event_idx = data["inducing_event_idx"]
#             if "inducing_xyz_km" in data:
#                 inducing_xyz_km = data["inducing_xyz_km"]
#         except Exception as e:
#             return

#     try:
#         comp_ids = np.asarray(comp_ids, dtype=np.int64)
#         comp_off = np.asarray(comp_off, dtype=np.int64)
#         inducing_event_idx = np.asarray(inducing_event_idx, dtype=np.int64)
#         if inducing_xyz_km is not None:
#             inducing_xyz_km = np.asarray(inducing_xyz_km, dtype=np.float32)
#     except Exception:
#         return
#     if comp_ids.size == 0 or comp_off.size != comp_ids.size + 1 or inducing_event_idx.size == 0:
#         return

#     if getattr(state, "cluster_ids", None) is None:
#         return
#     try:
#         ev_comp = state.cluster_ids.detach().cpu().numpy().astype(np.int64, copy=False)
#     except Exception:
#         return
#     n_events = int(ev_comp.shape[0])

#     # MAP event coordinates (XYZ)
#     try:
#         X_map = (state.X_src.detach() + state.dX_src.detach())[:, :3].to(torch.float32).cpu().numpy()
#     except Exception:
#         return
#     if X_map.shape[0] != n_events:
#         return
#     P = X_map[:, :3].astype(np.float32, copy=False)

#     neigh_idx = np.full((n_events, m), -1, dtype=np.int64)
#     neigh_k = np.zeros((n_events, m), dtype=np.float32)
#     neigh_k_matern32 = np.zeros((n_events, m), dtype=np.float32)
#     t0 = time.time()
#     use_scipy = True
#     for bi, cid in enumerate(comp_ids.tolist()):
#         i0 = int(comp_off[bi]); i1 = int(comp_off[bi + 1])
#         if i1 <= i0:
#             continue
#         # Inducing locations for this component
#         Uc = None
#         try:
#             if fixed_xyz and inducing_xyz_km is not None and getattr(inducing_xyz_km, "ndim", 0) == 2 and int(inducing_xyz_km.shape[1]) >= 3:
#                 Uc = np.asarray(inducing_xyz_km[i0:i1, :3]).astype(np.float32, copy=False)
#         except Exception:
#             Uc = None
#         if Uc is None:
#             U_ev = inducing_event_idx[i0:i1]
#             Uc = P[U_ev]
#         M = int(Uc.shape[0])
#         if M <= 0:
#             continue
#         ev_idx = np.flatnonzero(ev_comp == int(cid)).astype(np.int64, copy=False)
#         if ev_idx.size == 0:
#             continue
#         Pc = P[ev_idx]
#         kq = int(min(m, M))
#         if kq <= 0:
#             continue
#         try:
#             from scipy.spatial import cKDTree  # type: ignore
#             tree = cKDTree(Uc.astype("float64", copy=False))
#             try:
#                 dists, nbrs = tree.query(Pc.astype("float64", copy=False), k=kq, workers=-1)
#             except TypeError:
#                 try:
#                     dists, nbrs = tree.query(Pc.astype("float64", copy=False), k=kq, n_jobs=-1)  # type: ignore[call-arg]
#                 except TypeError:
#                     dists, nbrs = tree.query(Pc.astype("float64", copy=False), k=kq)
#             if kq == 1:
#                 dists = np.asarray(dists).reshape(-1, 1)
#                 nbrs = np.asarray(nbrs).reshape(-1, 1)
#         except Exception:
#             use_scipy = False
#             Uc_f = Uc.astype(np.float32, copy=False)
#             Pc_f = Pc.astype(np.float32, copy=False)
#             nbrs = np.empty((Pc_f.shape[0], kq), dtype=np.int64)
#             dists = np.empty((Pc_f.shape[0], kq), dtype=np.float32)
#             chunk = 8192
#             for s0 in range(0, Pc_f.shape[0], chunk):
#                 s1 = min(s0 + chunk, Pc_f.shape[0])
#                 Q = Pc_f[s0:s1]
#                 q2 = (Q * Q).sum(axis=1, keepdims=True)
#                 u2 = (Uc_f * Uc_f).sum(axis=1, keepdims=True).T
#                 d2 = q2 + u2 - 2.0 * (Q @ Uc_f.T)
#                 d2 = np.maximum(d2, 0.0)
#                 part = np.argpartition(d2, kth=kq - 1, axis=1)[:, :kq]
#                 d2_part = np.take_along_axis(d2, part, axis=1)
#                 ord2 = np.argsort(d2_part, axis=1)
#                 part_sorted = np.take_along_axis(part, ord2, axis=1)
#                 d2_sorted = np.take_along_axis(d2_part, ord2, axis=1)
#                 nbrs[s0:s1, :] = part_sorted
#                 dists[s0:s1, :] = np.sqrt(d2_sorted).astype(np.float32, copy=False)

#         gidx = (np.asarray(nbrs, dtype=np.int64) + int(i0)).astype(np.int64, copy=False)
#         d_f = np.asarray(dists, dtype=np.float32)
#         k_val = np.exp(-0.5 * (d_f / float(ell)) ** 2).astype(np.float32, copy=False)
#         a = np.float32(np.sqrt(3.0) / float(ell))
#         x = (a * d_f).astype(np.float32, copy=False)
#         k_m32 = ((1.0 + x) * np.exp(-x)).astype(np.float32, copy=False)
#         neigh_idx[ev_idx, :kq] = gidx
#         neigh_k[ev_idx, :kq] = k_val
#         neigh_k_matern32[ev_idx, :kq] = k_m32

#     dt_s = time.time() - t0
#     info(
#         f"backend={'scipy_ckdtree' if use_scipy else 'bruteforce'} dt={dt_s:.1f}s",
#         section="LIKELIHOOD",
#     )

#     # Keep in memory for runtime
#     try:
#         dev = state.device
#         try:
#         except Exception:
#         try:
#             if inducing_xyz_km is not None:
#             else:
#         except Exception:
#     except Exception:
#         pass

#     # Optional NPZ write
#     try:
#         out_path_s = str(out_path) if (out_path is not None and str(out_path).strip() != "") else str("")
#         if out_path_s != "":
#             if not os.path.isabs(out_path_s):
#                 base = str(state.params.get("checkpoint_dir", "."))
#                 out_path_s = os.path.join(base, out_path_s)
#             os.makedirs(os.path.dirname(out_path_s) or ".", exist_ok=True)
#             np.savez_compressed(
#                 out_path_s,
#                 component_id=comp_ids,
#                 component_offsets=comp_off,
#                 inducing_event_idx=inducing_event_idx,
#                 inducing_xyz_km=(inducing_xyz_km if inducing_xyz_km is not None else np.zeros((0, 3), dtype=np.float32)),
#                 event_component_id=ev_comp.astype(np.int64, copy=False),
#                 neighbor_inducing_global_idx=neigh_idx,
#                 neighbor_kernel=neigh_k,
#                 neighbor_kernel_matern32=neigh_k_matern32,
#                 ell_km=np.asarray([float(ell)], dtype=np.float32),
#                 fixed_xyz=np.asarray([int(bool(fixed_xyz))], dtype=np.int8),
#             )
#     except Exception as e:


# @torch.no_grad()
#     """

#     This prepares small per-component matrices that the collapsed likelihood can use at runtime.
#     """
#     try:
#             return
#     except Exception:
#         return

#     # If already initialized (resume), keep existing.
#         return

#     try:
#         if not (ell_km > 0.0):
#             return
#         if not math.isfinite(jitter) or jitter < 0.0:
#             jitter = 1e-6
#     except Exception:
#         return

#     # Need interpolation (neighbor idx) and selection (offsets + inducing idx/xyz)
#     if nei_idx is None:
#     if not isinstance(nei_idx, torch.Tensor):
#         # Try loading from NPZ if present
#         if ip is not None:
#             try:
#                 ip_s = str(ip)
#                 if not os.path.isabs(ip_s):
#                     base = str(state.params.get("checkpoint_dir", "."))
#                     ip_s = os.path.join(base, ip_s)
#                 data = np.load(ip_s, allow_pickle=True)
#                 neigh_idx_np = np.asarray(data["neighbor_inducing_global_idx"], dtype=np.int64)
#                 dev = state.device
#                 nei_idx = torch.from_numpy(neigh_idx_np).to(device=dev, dtype=torch.int64)
#             except Exception:
#                 nei_idx = None
#     if not isinstance(nei_idx, torch.Tensor):
#         return

#     if comp_ids is None or comp_off is None or inducing_event_idx is None:
#         if sel_path is None:
#             return
#         try:
#             sel_path_s = str(sel_path)
#             if not os.path.isabs(sel_path_s):
#                 base = str(state.params.get("checkpoint_dir", "."))
#                 sel_path_s = os.path.join(base, sel_path_s)
#             data = np.load(sel_path_s, allow_pickle=True)
#             comp_ids = data["component_id"]
#             comp_off = data["component_offsets"]
#             inducing_event_idx = data["inducing_event_idx"]
#             if "inducing_xyz_km" in data:
#                 inducing_xyz_km = data["inducing_xyz_km"]
#         except Exception:
#             return

#     try:
#         comp_ids = np.asarray(comp_ids, dtype=np.int64)
#         comp_off = np.asarray(comp_off, dtype=np.int64)
#         inducing_event_idx = np.asarray(inducing_event_idx, dtype=np.int64)
#         if inducing_xyz_km is not None:
#             inducing_xyz_km = np.asarray(inducing_xyz_km, dtype=np.float32)
#     except Exception:
#         return
#     if comp_ids.size == 0 or comp_off.size != comp_ids.size + 1:
#         return

#     # MAP event XYZ in km (for non-fixed inducing geometry)
#     try:
#         X_map = (state.X_src.detach() + state.dX_src.detach())[:, :3].to(torch.float32).cpu().numpy()
#         P = X_map.astype(np.float32, copy=False)
#     except Exception:
#         return

#     dev = state.device

#     def _matern32_from_dist(D: torch.Tensor) -> torch.Tensor:
#         a = float(np.sqrt(3.0) / float(ell_km))
#         x = (a * D).to(torch.float32)
#         return (1.0 + x) * torch.exp(-x)

#     # Build K_UU blocks (per component)
#     t0 = time.time()
#     K_blocks: list[torch.Tensor] = []
#     jitter_escalated_blocks = 0
#     jitter_used_max = float(jitter)
#     for bi, cid in enumerate(comp_ids.tolist()):
#         i0 = int(comp_off[bi]); i1 = int(comp_off[bi + 1])
#         if i1 <= i0:
#             K_blocks.append(torch.zeros((0, 0), device=dev, dtype=torch.float32))
#             continue
#         Uc_np = None
#         try:
#             if fixed_xyz and inducing_xyz_km is not None and getattr(inducing_xyz_km, "ndim", 0) == 2 and int(inducing_xyz_km.shape[1]) >= 3:
#                 Uc_np = np.asarray(inducing_xyz_km[i0:i1, :3]).astype(np.float32, copy=False)
#         except Exception:
#             Uc_np = None
#         if Uc_np is None:
#             U_ev = inducing_event_idx[i0:i1]
#             Uc_np = np.asarray(P[U_ev, :3]).astype(np.float32, copy=False)
#         Uc = torch.from_numpy(Uc_np).to(device=dev, dtype=torch.float32)
#         D = torch.cdist(Uc, Uc).to(torch.float32)
#         K0 = _matern32_from_dist(D)
#         K0 = 0.5 * (K0 + K0.transpose(0, 1))
#         mK = int(K0.shape[0])
#         if mK <= 0:
#             K = K0
#         else:
#             I = torch.eye(mK, device=dev, dtype=K0.dtype)
#             j0 = float(jitter)
#             j_used = j0
#             K = K0 + (j_used * I)
#             try:
#                 L, chol_info = torch.linalg.cholesky_ex(K)
#                 if int(chol_info.item()) != 0:
#                     max_tries = 6
#                     for t in range(1, max_tries + 1):
#                         j_used = j0 * (10.0 ** t)
#                         K = K0 + (j_used * I)
#                         L, chol_info = torch.linalg.cholesky_ex(K)
#                         if int(chol_info.item()) == 0:
#                             break
#             except Exception:
#                 pass
#             if j_used > j0:
#                 jitter_escalated_blocks += 1
#             if j_used > jitter_used_max:
#                 jitter_used_max = float(j_used)
#         K_blocks.append(K)

#     offs_t = torch.from_numpy(comp_off).to(device=dev, dtype=torch.int64)

#     # Also build a single block-diagonal K_UU over all inducing points. This lets modeling.py solve
#     # per station-phase group without additionally splitting by connected component (still no coupling
#     # across components because K_UU is block diagonal and neighbor supports are disjoint).
#     try:
#         if K_blocks:
#             K_full = torch.block_diag(*[Kb.to(device=dev, dtype=torch.float32) for Kb in K_blocks])
#         else:
#             K_full = torch.zeros((0, 0), device=dev, dtype=torch.float32)
#         # Also precompute (K_full ⊗ I3) as a block-diagonal matrix to avoid rebuilding it per-group.
#         try:
#             K_full3 = torch.block_diag(K_full, K_full, K_full)
#         except Exception:
#             K_full3 = None
#     except Exception:
#         pass

#     # Build comp_id -> block index mapping for O(1) lookup at runtime.
#     try:
#         if getattr(state, "cluster_counts", None) is not None:
#             n_comp_total = int(state.cluster_counts.numel())
#         else:
#             n_comp_total = int(comp_ids.max() + 1)
#         comp_to_block = np.full((n_comp_total,), -1, dtype=np.int64)
#         for bi, cid in enumerate(comp_ids.tolist()):
#             if 0 <= int(cid) < int(n_comp_total):
#                 comp_to_block[int(cid)] = int(bi)
#         comp_to_block_t = torch.from_numpy(comp_to_block).to(device=dev, dtype=torch.int64)
#     except Exception:
#         pass

#     dt_init = time.time() - t0
#     info(
#         f"ell_km={ell_km:g} jitter={float(jitter):.3g} jitter_used_max={float(jitter_used_max):.3g} "
#         f"jitter_escalated_blocks={int(jitter_escalated_blocks)} dt={dt_init:.1f}s",
#         section="LIKELIHOOD",
#     )

#     # Optional FITC diagonal residual (computed at MAP; broadcast in DDP)
#     if fitc_enable:
#         ddp_on = _ddp_enabled(state.params)
#         ddp_main = (not ddp_on) or _ddp_is_main(state.params)
#         dist_ok = False
#         dist = None
#         if ddp_on:
#             try:
#                 import torch.distributed as dist  # type: ignore
#                 dist_ok = bool(dist.is_available()) and bool(dist.is_initialized())
#             except Exception:
#                 dist_ok = False
#                 dist = None
#         if ddp_on and dist_ok and (not ddp_main):
#             try:
#                 n_ev = int(nei_idx.shape[0])
#                 q_t = torch.empty((n_ev,), device=dev, dtype=torch.float32)
#                 r_t = torch.empty((n_ev,), device=dev, dtype=torch.float32)
#                 dist.broadcast(q_t, src=0)  # type: ignore[union-attr]
#                 dist.broadcast(r_t, src=0)  # type: ignore[union-attr]
#             except Exception as e:
#                 ddp_main = True
#                 dist_ok = False

#         if (not ddp_on) or ddp_main:
#             try:
#                 t_fitc0 = time.time()
#                 idx_np = nei_idx.detach().cpu().numpy().astype(np.int64, copy=False)
#                 offs_np = comp_off.astype(np.int64, copy=False)
#                 K_np = [Kb.detach().cpu().numpy().astype(np.float32, copy=False) for Kb in K_blocks]
#                 n_ev = int(idx_np.shape[0])
#                 q_diag_np = np.zeros((n_ev,), dtype=np.float32)

#                 # Inducing point coordinates (XYZ) at MAP
#                 if fixed_xyz and inducing_xyz_km is not None and getattr(inducing_xyz_km, "ndim", 0) == 2 and int(inducing_xyz_km.shape[1]) >= 3:
#                     P_ind = inducing_xyz_km.astype(np.float32, copy=False)
#                 else:
#                     P_ind = P[inducing_event_idx, :3].astype(np.float32, copy=False)

#                 a = np.float32(np.sqrt(3.0) / float(ell_km))
#                 for e in range(n_ev):
#                     idx_row = idx_np[e]
#                     m = idx_row >= 0
#                     if not bool(m.any()):
#                         q = 0.0
#                     else:
#                         idxv = idx_row[m]
#                         xe = P[int(e), :3]
#                         xu = P_ind[idxv].astype(np.float32, copy=False)
#                         d = np.linalg.norm(xu - xe[None, :], axis=1).astype(np.float32, copy=False)
#                         x = a * d
#                         kv = ((1.0 + x) * np.exp(-x)).astype(np.float32, copy=False)
#                         g0 = int(idxv[0])
#                         bi = int(np.searchsorted(offs_np[1:], g0, side="right"))
#                         if bi < 0 or bi >= len(K_np):
#                             q = 0.0
#                         else:
#                             i0 = int(offs_np[bi])
#                             loc = (idxv - i0).astype(np.int64, copy=False)
#                             try:
#                                 Kmm = K_np[bi][np.ix_(loc, loc)]
#                                 sol = np.linalg.solve(Kmm, kv)
#                                 q = float(kv.dot(sol))
#                             except Exception:
#                                 q = 0.0
#                     q_diag_np[e] = float(q)

#                 q_diag_np = np.clip(q_diag_np, 0.0, 1.0).astype(np.float32, copy=False)
#                 resid_np = np.maximum(0.0, 1.0 - q_diag_np).astype(np.float32, copy=False)
#                 q_t = torch.from_numpy(q_diag_np).to(device=dev, dtype=torch.float32)
#                 r_t = torch.from_numpy(resid_np).to(device=dev, dtype=torch.float32)

#                 if ddp_on and dist_ok:
#                     try:
#                         dist.broadcast(q_t, src=0)  # type: ignore[union-attr]
#                         dist.broadcast(r_t, src=0)  # type: ignore[union-attr]
#                     except Exception:
#                         pass

#                 dt_fitc = time.time() - t_fitc0
#                 info(
#                     f"q_diag[min/mean/max]={float(q_diag_np.min()):.3g}/{float(q_diag_np.mean()):.3g}/{float(q_diag_np.max()):.3g} "
#                     f"resid[min/mean/max]={float(resid_np.min()):.3g}/{float(resid_np.mean()):.3g}/{float(resid_np.max()):.3g} "
#                     f"(dt={dt_fitc:.2f}s)",
#                     section="LIKELIHOOD",
#                 )
#             except Exception as e:


# @torch.no_grad()
#     """
#     Initialize uncollapsed shared-event latent random effects b[s,event,phase] and a fixed event-space
#     kernel graph (kNN Laplacian) built from the current MAP (post Phase-1).

#     This is intended for small/medium problems where the full latent b is feasible to sample.
#     The event graph is fixed after Phase 1 under the assumption event locations won't move much.
#     """
#     try:
#             return
#     except Exception:
#         return

#     # Require stable station indices
#     if getattr(state, "row_station_index", None) is None or int(getattr(state, "n_stations", 0)) <= 0:
#         return

#     if mode not in {"full", "inducing_gp", "slowness_inducing_gp", "graph_gmrf"}:
#         mode = "full"

#     # If already initialized (e.g., resume), keep existing
#     if mode in {"inducing_gp", "slowness_inducing_gp"}:
#         already = (
#         )
#         if already:
#             # Note: W is not a Parameter and is not stored in checkpoints, so we must rebuild it.
#             try:
#             except Exception:
#                 use_sta_basis = False
#             if use_sta_basis:
#                 try:
#                     n_stations = int(getattr(state, "n_stations", 0))
#                     if (
#                         isinstance(b_lat, torch.Tensor)
#                         and b_lat.ndim == 3
#                         and int(b_lat.shape[2]) == 2
#                         and int(n_stations) > 0
#                     ):
#                         # We expect basis-rank coefficients: b_lat.shape[0] == R (not n_stations).
#                         r_sta = int(b_lat.shape[0])
#                         if r_sta >= 1 and r_sta <= n_stations and method == "eigh_rbf" and (ell_sta > 0.0):
#                             dt = state.dtimes
#                             if "sta_idx" in dt.columns and "X" in dt.columns and "Y" in dt.columns:
#                                 sta_xy = (
#                                     dt.select([pl.col("sta_idx"), pl.col("X"), pl.col("Y")])
#                                     .unique(subset=["sta_idx"], maintain_order=True)
#                                     .sort("sta_idx")
#                                 )
#                                 xy_np = sta_xy.select([pl.col("X"), pl.col("Y")]).to_numpy().astype(np.float32, copy=False)
#                                 XY = torch.from_numpy(xy_np).to(device=state.device, dtype=torch.float32)
#                                 D_sta = torch.cdist(XY, XY).to(torch.float32)
#                                 K_sta = torch.exp(-0.5 * (D_sta / float(ell_sta)).square())
#                                 j = 1e-6 if (not math.isfinite(jitter_sta) or jitter_sta <= 0.0) else float(jitter_sta)
#                                 K_sta = K_sta + (j * torch.eye(int(K_sta.shape[0]), device=state.device, dtype=K_sta.dtype))
#                                 evals, evecs = torch.linalg.eigh(K_sta)
#                                 evals = evals.clamp_min(0.0)
#                                 evals_r = evals[-r_sta:]
#                                 evecs_r = evecs[:, -r_sta:]
#                                 W_sta = evecs_r * torch.sqrt(evals_r).unsqueeze(0)  # (S,R)
#                 except Exception as e:

#             # If FITC is enabled but residual diag wasn't computed (e.g., resume from older checkpoint),
#             # compute it now and keep the existing parameter tensor.
#                 try:
#                     if (
#                         isinstance(nei_idx, torch.Tensor)
#                         and isinstance(nei_k, torch.Tensor)
#                         and isinstance(offs_t, torch.Tensor)
#                         and isinstance(K_blocks, list)
#                         and K_blocks
#                     ):
#                         t_fitc0 = time.time()
#                         idx_np = nei_idx.detach().cpu().numpy().astype(np.int64, copy=False)
#                         k_np = nei_k.detach().cpu().numpy().astype(np.float32, copy=False)
#                         offs_np = offs_t.detach().cpu().numpy().astype(np.int64, copy=False)
#                         K_np = [
#                             (Kb.detach().cpu().numpy().astype(np.float32, copy=False) if isinstance(Kb, torch.Tensor) else None)
#                             for Kb in K_blocks
#                         ]
#                         n_ev = int(idx_np.shape[0])
#                         q_diag_np = np.zeros((n_ev,), dtype=np.float32)
#                         for e in range(n_ev):
#                             idx_row = idx_np[e]
#                             k_row = k_np[e]
#                             m = idx_row >= 0
#                             if not bool(m.any()):
#                                 q = 0.0
#                             else:
#                                 idxv = idx_row[m]
#                                 kv = k_row[m]
#                                 g0 = int(idxv[0])
#                                 bi = int(np.searchsorted(offs_np[1:], g0, side="right"))
#                                 if bi < 0 or bi >= len(K_np) or K_np[bi] is None:
#                                     q = 0.0
#                                 else:
#                                     i0 = int(offs_np[bi])
#                                     loc = (idxv - i0).astype(np.int64, copy=False)
#                                     Kb = K_np[bi]
#                                     if Kb is None or Kb.size == 0:
#                                         q = 0.0
#                                     else:
#                                         try:
#                                             Kmm = Kb[np.ix_(loc, loc)]
#                                             sol = np.linalg.solve(Kmm, kv)
#                                             q = float(kv.dot(sol))
#                                         except Exception:
#                                             q = 0.0
#                             q_diag_np[e] = float(q)
#                         q_diag_np = np.clip(q_diag_np, 0.0, 1.0).astype(np.float32, copy=False)
#                         resid_np = np.maximum(0.0, 1.0 - q_diag_np).astype(np.float32, copy=False)
#                         dt_fitc = time.time() - t_fitc0
#                         info(
#                             f"q_diag[min/mean/max]={float(q_diag_np.min()):.3g}/{float(q_diag_np.mean()):.3g}/{float(q_diag_np.max()):.3g} "
#                             f"resid[min/mean/max]={float(resid_np.min()):.3g}/{float(resid_np.mean()):.3g}/{float(resid_np.max()):.3g} "
#                             f"(dt={dt_fitc:.2f}s)",
#                             section="LIKELIHOOD",
#                         )
#                 except Exception:
#                     pass
#             return
#     else:
#         if (
#         ):
#             return

#     n_events = int(state.X_src.shape[0])
#     n_stations = int(getattr(state, "n_stations", 0))
#     if n_events <= 1 or n_stations <= 0:
#         return

#     if not (ell_km > 0.0):
#         return
#     knn = min(int(knn), max(1, n_events - 1))

#     # --- inducing_gp parameterization ---
#     # --- slowness_inducing_gp parameterization ---
#     if mode in {"inducing_gp", "slowness_inducing_gp"}:
#         # This mode uses Stage-2/3 inducing_plan artifacts (selection + interpolation) to build:
#         # - per-event sparse neighbor lists into the concatenated inducing list
#         # - per-component GP prior blocks K_UU on inducing coefficients
#         dev = state.device
#         if not (ell_km > 0.0):
#             return

#         # Prefer in-memory artifacts (generated earlier in this run) to avoid NPZ disk round-trips.
#         comp_ids = None
#         comp_off = None
#         inducing_event_idx = None
#         inducing_xyz_km = None
#         have_interp_mem = False
#         try:
#             if comp_ids is not None and comp_off is not None and inducing_event_idx is not None:
#                 comp_ids = np.asarray(comp_ids, dtype=np.int64)
#                 comp_off = np.asarray(comp_off, dtype=np.int64)
#                 inducing_event_idx = np.asarray(inducing_event_idx, dtype=np.int64)
#                 if inducing_xyz_km is not None:
#                     inducing_xyz_km = np.asarray(inducing_xyz_km, dtype=np.float32)
#         except Exception:
#             comp_ids = None
#             comp_off = None
#             inducing_event_idx = None
#             inducing_xyz_km = None
#         try:
#             have_interp_mem = bool(
#                 and (
#                     (mode == "slowness_inducing_gp")
#                 )
#             )
#         except Exception:
#             have_interp_mem = False

#         # Resolve selection/interpolation file paths (relative to checkpoint_dir)
#         def _resolve_path(v):
#             if v is None:
#                 return None
#             p = str(v)
#             if not p:
#                 return None
#             if os.path.isabs(p):
#                 return p
#             base = str(state.params.get("checkpoint_dir", "."))
#             return os.path.join(base, p)

#         # Prefer runtime-resolved paths from Stage 2/3 if present
#         if sel_path_rt:
#             sel_path = str(sel_path_rt)
#         if interp_path_rt:
#             interp_path = str(interp_path_rt)

#         # Load selection if not already available in memory
#         if comp_ids is None or comp_off is None or inducing_event_idx is None:
#             if not sel_path or not os.path.exists(sel_path):
#                 raise ValueError(
#                     "Either run Stage 2 (inducing_plan.select=true) in this run or provide the NPZ. "
#                     f"Expected at: {sel_path}"
#                 )
#             try:
#                 sel = np.load(sel_path, allow_pickle=True)
#                 comp_ids = np.asarray(sel["component_id"], dtype=np.int64)
#                 comp_off = np.asarray(sel["component_offsets"], dtype=np.int64)
#                 inducing_event_idx = np.asarray(sel["inducing_event_idx"], dtype=np.int64)
#                 if "inducing_xyz_km" in sel:
#                     try:
#                         inducing_xyz_km = np.asarray(sel["inducing_xyz_km"], dtype=np.float32)
#                     except Exception:
#                         inducing_xyz_km = None
#                 use_xyz = False
#                 if "use_xyz" in sel:
#                     try:
#                         use_xyz = bool(int(np.asarray(sel["use_xyz"]).reshape(-1)[0]))
#                     except Exception:
#                         use_xyz = False
#             except Exception as e:
#                 raise ValueError(f"Failed to load inducing selection NPZ '{sel_path}': {e}")

#         if mode == "slowness_inducing_gp":
#             # Force XYZ geometry for slowness-vector GP.
#             use_xyz = True

#         if comp_ids is None or comp_off is None or inducing_event_idx is None:
#             raise ValueError("Missing inducing selection arrays (component_id/component_offsets/inducing_event_idx).")
#         if int(comp_ids.size) == 0 or int(comp_off.size) != int(comp_ids.size) + 1:
#             raise ValueError("Invalid inducing selection arrays: bad component_offsets/component_id.")
#         if int(inducing_event_idx.size) == 0:
#             raise ValueError("Invalid inducing selection arrays: empty inducing_event_idx.")

#         # Interpolation: if in-memory neighbor tensors exist, keep them; otherwise load from NPZ
#         if not have_interp_mem:
#             if not interp_path or not os.path.exists(interp_path):
#                 raise ValueError(
#                     "Either run Stage 3 (inducing_plan.interpolation.enabled=true) in this run or provide the NPZ. "
#                     f"Expected at: {interp_path}"
#                 )
#             try:
#                 itp = np.load(interp_path, allow_pickle=True)
#                 neigh_idx_np = np.asarray(itp["neighbor_inducing_global_idx"], dtype=np.int64)
#                 neigh_k_np = np.asarray(itp["neighbor_kernel"], dtype=np.float32)
#             except Exception as e:
#                 raise ValueError(f"Failed to load inducing interpolation NPZ '{interp_path}': {e}")

#         n_events = int(state.X_src.shape[0])
#         M_total = int(inducing_event_idx.size)
#         if not have_interp_mem:
#             if neigh_idx_np.ndim != 2 or neigh_k_np.shape != neigh_idx_np.shape:
#                 raise ValueError(f"Invalid inducing interpolation NPZ '{interp_path}': neighbor arrays shape mismatch")
#             if int(neigh_idx_np.shape[0]) != int(n_events):
#                 raise ValueError(
#                     f"Invalid inducing interpolation NPZ '{interp_path}': n_events mismatch "
#                     f"(file has {int(neigh_idx_np.shape[0])}, run has {n_events})"
#                 )
#             if int(neigh_idx_np.max(initial=-1)) >= M_total:
#                 raise ValueError(
#                     f"Invalid inducing interpolation NPZ '{interp_path}': neighbor indices exceed inducing list "
#                     f"(max idx {int(neigh_idx_np.max())} vs M_total {M_total})"
#                 )
#             # Store interpolation tensors on device
#         # Store inducing event indices (needed for slowness_inducing_gp runtime kernel weights).
#         try:
#         except Exception:

#         # Build per-component K_UU blocks on device (kernel + small jitter)
#         # - inducing_gp: RBF
#         # - slowness_inducing_gp: Matérn(3/2)
#         # MAP coordinates in km
#         X_map = (state.X_src.detach() + state.dX_src.detach())[:, :3].to(torch.float32).cpu().numpy()
#         if use_xyz:
#             P = X_map[:, :3].astype(np.float32, copy=False)
#             dim_label = "xyz"
#         else:
#             P = X_map[:, :2].astype(np.float32, copy=False)
#             dim_label = "xy"

#         # Resolve fixed inducing XYZ locations (km) from selection, if available.
#         # For Option-B (fixed_xyz), we treat these as fixed inducing points U in space.
#         inducing_xyz_np = None
#         try:
#             if inducing_xyz_km is not None and getattr(inducing_xyz_km, "ndim", 0) == 2 and int(inducing_xyz_km.shape[0]) == int(inducing_event_idx.size) and int(inducing_xyz_km.shape[1]) >= 3:
#                 inducing_xyz_np = np.asarray(inducing_xyz_km, dtype=np.float32).astype(np.float32, copy=False)
#         except Exception:
#             inducing_xyz_np = None
#         if inducing_xyz_np is None:
#             # Backward-compatible: derive inducing XYZ from the MAP positions of inducing event indices.
#             try:
#                 inducing_xyz_np = X_map[inducing_event_idx, :3].astype(np.float32, copy=False)
#             except Exception:
#                 inducing_xyz_np = None
#         # Keep inducing XYZ on device for runtime kernels (especially slowness_inducing_gp).
#         try:
#             if inducing_xyz_np is not None:
#             else:
#         except Exception:

#         def _kernel_from_dist(D: torch.Tensor) -> torch.Tensor:
#             if mode == "slowness_inducing_gp":
#                 # Matérn ν=3/2: (1 + √3 r/ℓ) exp(-√3 r/ℓ)
#                 a = (math.sqrt(3.0) / float(ell_km))
#                 x = (a * D).to(torch.float32)
#                 return (1.0 + x) * torch.exp(-x)
#             # RBF: exp(-0.5 (r/ℓ)^2)
#             return torch.exp(-0.5 * (D / float(ell_km)).square())

#         # Require at least tiny numerical jitter so K_UU is strictly PD (proper prior over inducing coeffs).
#         jitter = 1e-6 if (not math.isfinite(jitter) or jitter <= 0.0) else jitter
#         K_blocks: list[torch.Tensor] = []
#         # Optional conditioning diagnostics (cheap for small blocks; skipped for large ones).
#         # This helps debug "runaway" behavior caused by nearly-singular K_UU blocks (weakly-regularized coeff modes).
#         eig_min_list: list[float] = []
#         eig_max_list: list[float] = []
#         cond_list: list[float] = []
#         skipped_cond_blocks = 0
#         max_dim_for_cond = 256
#         jitter_used_max = float(jitter)
#         jitter_escalated_blocks = 0
#         for bi in range(int(comp_ids.size)):
#             i0 = int(comp_off[bi]); i1 = int(comp_off[bi + 1])
#             if i1 <= i0:
#                 K_blocks.append(torch.zeros((0, 0), device=dev, dtype=torch.float32))
#                 continue
#             if fixed_xyz and inducing_xyz_np is not None:
#                 Uc_np = inducing_xyz_np[i0:i1, : int(P.shape[1])]
#                 Uc = torch.from_numpy(Uc_np).to(device=dev, dtype=torch.float32)
#             else:
#                 U_ev = inducing_event_idx[i0:i1]
#                 Uc = torch.from_numpy(P[U_ev]).to(device=dev, dtype=torch.float32)
#             # Pairwise distances (M,M)
#             D = torch.cdist(Uc, Uc).to(torch.float32)
#             K0 = _kernel_from_dist(D)
#             # Enforce symmetry explicitly (GPU cdist can be slightly asymmetric in float32).
#             K0 = 0.5 * (K0 + K0.transpose(0, 1))
#             mK = int(K0.shape[0])
#             if mK <= 0:
#                 K = K0
#             else:
#                 I = torch.eye(mK, device=dev, dtype=K0.dtype)
#                 j0 = float(jitter)
#                 j_used = j0
#                 K = K0 + (j_used * I)
#                 # Ensure PD via Cholesky; if it fails, escalate jitter on this block.
#                 try:
#                     L, chol_info = torch.linalg.cholesky_ex(K)
#                     if int(chol_info.item()) != 0:
#                         max_tries = 6
#                         for t in range(1, max_tries + 1):
#                             j_used = j0 * (10.0 ** t)
#                             K = K0 + (j_used * I)
#                             L, chol_info = torch.linalg.cholesky_ex(K)
#                             if int(chol_info.item()) == 0:
#                                 break
#                 except Exception:
#                     pass
#                 if j_used > j0:
#                     jitter_escalated_blocks += 1
#                 if j_used > jitter_used_max:
#                     jitter_used_max = float(j_used)
#             K_blocks.append(K)
#             try:
#                 if mK > 0 and mK <= max_dim_for_cond:
#                     ev = torch.linalg.eigvalsh(K)
#                     lam_min = float(ev.min().item())
#                     lam_max = float(ev.max().item())
#                     eig_min_list.append(lam_min)
#                     eig_max_list.append(lam_max)
#                     if lam_min > 0.0:
#                         cond_list.append(lam_max / max(lam_min, 1e-30))
#                 elif mK > max_dim_for_cond:
#                     skipped_cond_blocks += 1
#             except Exception:
#                 pass

#         # Store in state and params for prior
#         offs_t = torch.from_numpy(comp_off).to(device=dev, dtype=torch.int64)

#         # Log K_UU conditioning summary once (helps pick a sensible kernel_jitter).
#         try:
#             if len(eig_min_list) > 0:
#                 emn = np.asarray(eig_min_list, dtype=np.float64)
#                 emx = np.asarray(eig_max_list, dtype=np.float64)
#                 cnd = np.asarray(cond_list, dtype=np.float64) if len(cond_list) > 0 else None
#                 if cnd is not None and cnd.size > 0:
#                 info(
#                     f"jitter={float(jitter):.3g} jitter_used_max={float(jitter_used_max):.3g} "
#                     f"jitter_escalated_blocks={int(jitter_escalated_blocks)} blocks={int(comp_ids.size)} "
#                     f"eig_min[min/med]={float(np.nanmin(emn)):.3g}/{float(np.nanmedian(emn)):.3g} "
#                     f"eig_max[med]={float(np.nanmedian(emx)):.3g} "
#                     f"cond[max]={float(np.nanmax(cnd)):.3g}" if (cnd is not None and cnd.size > 0) else
#                     f"jitter={float(jitter):.3g} jitter_used_max={float(jitter_used_max):.3g} "
#                     f"jitter_escalated_blocks={int(jitter_escalated_blocks)} blocks={int(comp_ids.size)} "
#                     f"eig_min[min/med]={float(np.nanmin(emn)):.3g}/{float(np.nanmedian(emn)):.3g} "
#                     f"eig_max[med]={float(np.nanmedian(emx)):.3g} "
#                     f"(cond skipped for {int(skipped_cond_blocks)} blocks > {int(max_dim_for_cond)})",
#                     section="LIKELIHOOD",
#                 )
#         except Exception:
#             pass

#         # Stage 5 (FITC): compute diagonal residual Λ_ee = max(0, 1 - Q_ee) where
#         # Q_ee ≈ K_eU K_UU^{-1} K_Ue, approximated using the same m-neighbor subset as interpolation.
#         #
#         # Notes:
#         # - For inducing_gp, interpolation NPZ stores RBF kernel values; we reuse them here.
#         # - For slowness_inducing_gp, we use Matérn(3/2) in XYZ and recompute k(e,U) at MAP because
#         #   the interpolation NPZ kernel values are RBF (and slowness mode uses a different kernel).
#         if fitc_enable:
#             # DDP optimization: compute FITC diag once on rank0, then broadcast to all ranks.
#             ddp_on = _ddp_enabled(state.params)
#             ddp_main = (not ddp_on) or _ddp_is_main(state.params)
#             dist_ok = False
#             dist = None
#             if ddp_on:
#                 try:
#                     import torch.distributed as dist  # type: ignore
#                     dist_ok = bool(dist.is_available()) and bool(dist.is_initialized())
#                 except Exception:
#                     dist_ok = False
#                     dist = None

#             if ddp_on and dist_ok and (not ddp_main):
#                 # Non-rank0: receive and skip the CPU loop entirely.
#                 try:
#                     q_t = torch.empty((n_ev,), device=dev, dtype=torch.float32)
#                     r_t = torch.empty((n_ev,), device=dev, dtype=torch.float32)
#                     dist.broadcast(q_t, src=0)  # type: ignore[union-attr]
#                     dist.broadcast(r_t, src=0)  # type: ignore[union-attr]
#                 except Exception as e:
#                     # If broadcast fails for some reason, fall back to local compute on this rank.
#                     ddp_main = True
#                     dist_ok = False

#             if (not ddp_on) or ddp_main:
#                 try:
#                     t_fitc0 = time.time()
#                     offs_np = comp_off.astype(np.int64, copy=False)
#                     K_np = [Kb.detach().cpu().numpy().astype(np.float32, copy=False) for Kb in K_blocks]
#                     n_ev = int(idx_np.shape[0])
#                     q_diag_np = np.zeros((n_ev,), dtype=np.float32)

#                     # MAP coordinates for events (XYZ) used for slowness_inducing_gp kernel eval
#                     P_ev = None
#                     P_ind = None
#                     if mode == "slowness_inducing_gp":
#                         try:
#                             X_map = (state.X_src.detach() + state.dX_src.detach())[:, :3].to(torch.float32).cpu().numpy()
#                             P_ev = X_map.astype(np.float32, copy=False)
#                             # Prefer fixed inducing XYZ locations (Option-B) if available; otherwise fall back to inducing_event_idx at MAP.
#                                 P_ind = Ux.detach().cpu().numpy().astype(np.float32, copy=False)
#                             else:
#                                 P_ind = P_ev[inducing_event_idx].astype(np.float32, copy=False)
#                         except Exception:
#                             P_ev = None
#                             P_ind = None
#                     else:

#                     for e in range(n_ev):
#                         idx_row = idx_np[e]
#                         m = idx_row >= 0
#                         if not bool(m.any()):
#                             q = 0.0
#                         else:
#                             idxv = idx_row[m]
#                             if mode == "slowness_inducing_gp":
#                                 if P_ev is None or P_ind is None:
#                                     kv = np.ones((int(idxv.size),), dtype=np.float32)
#                                 else:
#                                     # Matérn ν=3/2 kernel values k(||x_e - x_u||) at MAP
#                                     xe = P_ev[int(e)]
#                                     xu = P_ind[idxv].astype(np.float32, copy=False)
#                                     d = np.linalg.norm(xu - xe[None, :], axis=1).astype(np.float32, copy=False)
#                                     a = np.float32(np.sqrt(3.0) / float(ell_km))
#                                     x = a * d
#                                     kv = ((1.0 + x) * np.exp(-x)).astype(np.float32, copy=False)
#                             else:
#                                 k_row = k_np[e]
#                                 kv = k_row[m]
#                             g0 = int(idxv[0])
#                             bi = int(np.searchsorted(offs_np[1:], g0, side="right"))
#                             if bi < 0 or bi >= len(K_np):
#                                 q = 0.0
#                             else:
#                                 i0 = int(offs_np[bi])
#                                 loc = (idxv - i0).astype(np.int64, copy=False)
#                                 try:
#                                     Kmm = K_np[bi][np.ix_(loc, loc)]
#                                     sol = np.linalg.solve(Kmm, kv)
#                                     q = float(kv.dot(sol))
#                                 except Exception:
#                                     q = 0.0
#                         q_diag_np[e] = float(q)

#                     q_diag_np = np.clip(q_diag_np, 0.0, 1.0).astype(np.float32, copy=False)
#                     resid_np = np.maximum(0.0, 1.0 - q_diag_np).astype(np.float32, copy=False)
#                     q_t = torch.from_numpy(q_diag_np).to(device=dev, dtype=torch.float32)
#                     r_t = torch.from_numpy(resid_np).to(device=dev, dtype=torch.float32)

#                     # Broadcast results to the other ranks (so they can skip the loop).
#                     if ddp_on and dist_ok:
#                         try:
#                             dist.broadcast(q_t, src=0)  # type: ignore[union-attr]
#                             dist.broadcast(r_t, src=0)  # type: ignore[union-attr]
#                         except Exception:
#                             pass

#                     dt_fitc = time.time() - t_fitc0
#                     info(
#                         f"q_diag[min/mean/max]={float(q_diag_np.min()):.3g}/{float(q_diag_np.mean()):.3g}/{float(q_diag_np.max()):.3g} "
#                         f"resid[min/mean/max]={float(resid_np.min()):.3g}/{float(resid_np.mean()):.3g}/{float(resid_np.max()):.3g} "
#                         f"(dt={dt_fitc:.2f}s)",
#                         section="LIKELIHOOD",
#                     )
#                 except Exception as e:

#         # Allocate inducing coefficients (sampled). This represents K_UU^{-1} g in a predictive-process GP.
#         # Default: per-station coefficients c[station, M_total, 2].
#         # Optional: fixed station-geometry basis reduces station DOF: c[rank_R, M_total, 2] with station weights W[station, R].
#         n_stations = int(getattr(state, "n_stations", 0))
#         r_sta = 0
#         if use_sta_basis:
#             try:
#                 if r_req < 1 or not (ell_sta > 0.0) or method != "eigh_rbf":
#                 r_sta = int(min(int(r_req), int(n_stations)))
#                 if r_sta < 1:

#                 # Build station XY in sta_idx order (aligned with row_station_index mapping).
#                 dt = state.dtimes
#                 if "sta_idx" not in dt.columns:
#                     raise ValueError("dtimes is missing 'sta_idx' (station index)")
#                 if "X" not in dt.columns or "Y" not in dt.columns:
#                     raise ValueError("dtimes is missing projected station coordinates 'X','Y'")
#                 sta_xy = (
#                     dt.select([pl.col("sta_idx"), pl.col("X"), pl.col("Y")])
#                     .unique(subset=["sta_idx"], maintain_order=True)
#                     .sort("sta_idx")
#                 )
#                 if int(sta_xy.shape[0]) != int(n_stations):
#                     raise ValueError(
#                         "This can happen if sta_idx was not rebuilt after filtering."
#                     )
#                 xy_np = sta_xy.select([pl.col("X"), pl.col("Y")]).to_numpy().astype(np.float32, copy=False)
#                 XY = torch.from_numpy(xy_np).to(device=dev, dtype=torch.float32)  # (S,2)

#                 # RBF station kernel + jitter; take top-R eigenpairs to form a fixed basis W = V sqrt(Λ).
#                 D_sta = torch.cdist(XY, XY).to(torch.float32)
#                 K_sta = torch.exp(-0.5 * (D_sta / float(ell_sta)).square())
#                 j = 1e-6 if (not math.isfinite(jitter_sta) or jitter_sta <= 0.0) else float(jitter_sta)
#                 K_sta = K_sta + (j * torch.eye(int(K_sta.shape[0]), device=dev, dtype=K_sta.dtype))
#                 evals, evecs = torch.linalg.eigh(K_sta)
#                 evals = evals.clamp_min(0.0)
#                 evals_r = evals[-r_sta:]
#                 evecs_r = evecs[:, -r_sta:]
#                 W_sta = evecs_r * torch.sqrt(evals_r).unsqueeze(0)  # (S,R)
#             except Exception as e:
#                 use_sta_basis = False
#                 r_sta = 0

#         if mode == "slowness_inducing_gp":
#             # Slowness-vector coefficients: c[..., inducing, phase, xyz]
#                 c0 = torch.zeros((int(r_sta), M_total, 2, 3), dtype=torch.float32, device=dev)
#             else:
#                 c0 = torch.zeros((n_stations, M_total, 2, 3), dtype=torch.float32, device=dev)
#         else:
#                 c0 = torch.zeros((int(r_sta), M_total, 2), dtype=torch.float32, device=dev)
#             else:
#                 c0 = torch.zeros((n_stations, M_total, 2), dtype=torch.float32, device=dev)

#         # Clear full-mode graph keys

#         try:
#         except Exception:
#             coeff_shape = ()
#         # neighbor_m can come from either NPZ load path (neigh_idx_np) or in-memory tensors.
#         try:
#             neighbor_m = int(neigh_idx_t.shape[1]) if isinstance(neigh_idx_t, torch.Tensor) and neigh_idx_t.ndim == 2 else -1
#         except Exception:
#             neighbor_m = -1
#         info(
#             f"neighbor_m={int(neighbor_m)} comps={int(comp_ids.size)} ell_km={ell_km:g} dims={dim_label} "
#             f"jitter={float(jitter):.3g} (FITC={'on' if fitc_enable else 'off'}) "
#             section="LIKELIHOOD",
#         )
#         return

#     # --- graph_gmrf parameterization (DD-linked event graph) ---
#     if mode == "graph_gmrf":
#         dev = state.device
#         n_events = int(state.X_src.shape[0])
#         n_stations = int(getattr(state, "n_stations", 0))
#         if not (ell_km > 0.0) or n_events <= 1 or n_stations <= 0:
#             return

#         # Optional diagonal term and Laplacian scaling (lambda)
#         try:
#         except Exception:
#             lam = 1.0
#         # Optional DD-edge pruning: keep only top-k DD neighbors per node (and/or within max_edge_km).
#         try:
#             max_edge_km_f = float(max_edge_km) if max_edge_km is not None else None
#             if max_edge_km_f is not None and (not math.isfinite(max_edge_km_f) or max_edge_km_f <= 0.0):
#                 max_edge_km_f = None
#         except Exception:
#             max_edge_km_f = None
#         if max_deg < 0:
#             max_deg = 0
#         if not math.isfinite(q_diag) or q_diag < 0.0:
#             q_diag = 1.0
#         if not math.isfinite(lam) or lam < 0.0:
#             lam = 1.0

#         # Build station basis W if enabled (same as inducing_gp path)
#         r_sta = 0
#         if use_sta_basis:
#             try:
#                 if r_req < 1 or not (ell_sta > 0.0) or method != "eigh_rbf":
#                 r_sta = int(min(int(r_req), int(n_stations)))
#                 if r_sta < 1:

#                 dt = state.dtimes
#                 if "sta_idx" not in dt.columns:
#                     raise ValueError("dtimes is missing 'sta_idx' (station index)")
#                 if "X" not in dt.columns or "Y" not in dt.columns:
#                     raise ValueError("dtimes is missing projected station coordinates 'X','Y'")
#                 sta_xy = (
#                     dt.select([pl.col("sta_idx"), pl.col("X"), pl.col("Y")])
#                     .unique(subset=["sta_idx"], maintain_order=True)
#                     .sort("sta_idx")
#                 )
#                 if int(sta_xy.shape[0]) != int(n_stations):
#                     raise ValueError(
#                     )
#                 xy_np = sta_xy.select([pl.col("X"), pl.col("Y")]).to_numpy().astype(np.float32, copy=False)
#                 XY = torch.from_numpy(xy_np).to(device=dev, dtype=torch.float32)  # (S,2)
#                 D_sta = torch.cdist(XY, XY).to(torch.float32)
#                 K_sta = torch.exp(-0.5 * (D_sta / float(ell_sta)).square())
#                 j = 1e-6 if (not math.isfinite(jitter_sta) or jitter_sta <= 0.0) else float(jitter_sta)
#                 K_sta = K_sta + (j * torch.eye(int(K_sta.shape[0]), device=dev, dtype=K_sta.dtype))
#                 evals, evecs = torch.linalg.eigh(K_sta)
#                 evals = evals.clamp_min(0.0)
#                 evals_r = evals[-r_sta:]
#                 evecs_r = evecs[:, -r_sta:]
#                 W_sta = evecs_r * torch.sqrt(evals_r).unsqueeze(0)  # (S,R)
#             except Exception as e:
#                 use_sta_basis = False
#                 r_sta = 0

#         # Allocate explicit event latents b (sampled)
#             b0 = torch.zeros((int(r_sta), int(n_events), 2), dtype=torch.float32, device=dev)
#         else:
#             b0 = torch.zeros((int(n_stations), int(n_events), 2), dtype=torch.float32, device=dev)

#         # Build unique undirected DD-linked edges from II (CPU)
#         t0 = time.time()
#         II_cpu = getattr(state, "_II_cpu", None)
#         if II_cpu is None:
#             II_cpu = state.II.detach().to("cpu").numpy().astype(np.int64, copy=False)
#             try:
#                 state._II_cpu = II_cpu
#             except Exception:
#                 pass
#         if II_cpu.ndim != 2 or II_cpu.shape[1] != 2:
#         a = II_cpu[:, 0]
#         b = II_cpu[:, 1]
#         u0 = np.minimum(a, b)
#         v0 = np.maximum(a, b)
#         msk = (u0 != v0)
#         u0 = u0[msk]
#         v0 = v0[msk]
#         if u0.size == 0:
#             return
#         # sort + dedup (lexicographic)
#         order = np.lexsort((v0, u0))
#         u1 = u0[order]
#         v1 = v0[order]
#         keep = np.ones((u1.size,), dtype=bool)
#         keep[1:] = (u1[1:] != u1[:-1]) | (v1[1:] != v1[:-1])
#         u = u1[keep].astype(np.int64, copy=False)
#         v = v1[keep].astype(np.int64, copy=False)

#         # MAP coordinates (km), fixed weights built once
#         X_map = (state.X_src.detach() + state.dX_src.detach())[:, :3].to(torch.float32).cpu().numpy()

#         # Compute edge weights w_ij = lambda * exp(-||xi-xj||/ell_km)
#         E = int(u.size)
#         w_np = np.empty((E,), dtype=np.float32)
#         d_np = np.empty((E,), dtype=np.float32) if (max_deg > 0 or max_edge_km_f is not None) else None
#         chunk = 1_000_000
#         for i0 in range(0, E, chunk):
#             i1 = min(i0 + chunk, E)
#             du = X_map[u[i0:i1]]
#             dv = X_map[v[i0:i1]]
#             d = np.linalg.norm(du - dv, axis=1).astype(np.float32, copy=False)
#             w_np[i0:i1] = (float(lam) * np.exp(-d / float(ell_km))).astype(np.float32, copy=False)
#             if d_np is not None:
#                 d_np[i0:i1] = d

#         # Optional sparsification on the DD-edge set:
#         # - If max_edge_km is set, drop long edges first.
#         # - If max_deg>0, cap degree by keeping the closest max_deg DD-neighbors per node (kNN on DD graph).
#         # We do this on the undirected DD graph by selecting directed top-k per source, then symmetrizing (union).
#         if d_np is not None:
#             if max_edge_km_f is not None:
#                 m = d_np <= float(max_edge_km_f)
#                 if not bool(np.all(m)):
#                     u = u[m]
#                     v = v[m]
#                     w_np = w_np[m]
#                     d_np = d_np[m]
#                     E = int(u.size)

#             if max_deg > 0 and E > 0:
#                 # Build directed edge list
#                 src = np.concatenate([u, v]).astype(np.int64, copy=False)
#                 dst = np.concatenate([v, u]).astype(np.int64, copy=False)
#                 dd = np.concatenate([d_np, d_np]).astype(np.float32, copy=False)
#                 ww = np.concatenate([w_np, w_np]).astype(np.float32, copy=False)

#                 # Sort by (src, dd) so closest neighbors come first per node
#                 order = np.lexsort((dd, src))
#                 src = src[order]
#                 dst = dst[order]
#                 ww = ww[order]

#                 # Keep first max_deg per src
#                 # Compute start indices of each src group
#                 starts = np.empty((src.size,), dtype=bool)
#                 starts[0] = True
#                 starts[1:] = (src[1:] != src[:-1])
#                 group_ids = np.cumsum(starts) - 1  # 0..n_groups-1
#                 # position within each group (0,1,2,...)
#                 idx_in_group = np.arange(src.size, dtype=np.int64) - np.maximum.accumulate(np.where(starts, np.arange(src.size, dtype=np.int64), 0))
#                 keep_dir = idx_in_group < int(max_deg)
#                 src_k = src[keep_dir]
#                 dst_k = dst[keep_dir]
#                 ww_k = ww[keep_dir]

#                 # Symmetrize by union and coalesce to unique undirected edges
#                 uu = np.minimum(src_k, dst_k)
#                 vv = np.maximum(src_k, dst_k)
#                 pairs = np.stack([uu, vv], axis=1).astype(np.int64, copy=False)
#                 uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
#                 # combine weights: take max (preserve strongest link if both directions kept)
#                 w_out = np.zeros((uniq.shape[0],), dtype=np.float32)
#                 np.maximum.at(w_out, inv, ww_k.astype(np.float32, copy=False))
#                 m2 = (uniq[:, 0] != uniq[:, 1])
#                 uniq = uniq[m2]
#                 w_out = w_out[m2]
#                 u = uniq[:, 0].astype(np.int64, copy=False)
#                 v = uniq[:, 1].astype(np.int64, copy=False)
#                 w_np = w_out.astype(np.float32, copy=False)
#                 E = int(u.size)

#         # Move to device
#         u_t = torch.from_numpy(u).to(device=dev, dtype=torch.int64)
#         v_t = torch.from_numpy(v).to(device=dev, dtype=torch.int64)
#         w_t = torch.from_numpy(w_np).to(device=dev, dtype=torch.float32)


#         dt_s = time.time() - t0
#         info(
#             f"ell_km={float(ell_km):g} lambda={float(lam):g} q_diag={float(q_diag):g} "
#             f"max_degree={int(max_deg)} max_edge_km={(float(max_edge_km_f) if max_edge_km_f is not None else 'None')} "
#             f"(dt={dt_s:.2f}s)",
#             section="LIKELIHOOD",
#         )
#         return

#     # MAP coordinates in km (fix kernel to MAP)
#     #
#     # IMPORTANT: The original implementation used a full pairwise distance matrix via torch.cdist,
#     # which is O(n_events^2) memory/time and becomes a major pause between Phase 1 and Phase 2
#     # for large catalogs. We prefer an exact kNN query via SciPy cKDTree when available.
#     t0 = time.time()
#     X_map = (state.X_src.detach() + state.dX_src.detach())[:, :3].to(torch.float32).cpu()
#     X_np = X_map.numpy()
#     # Directed kNN edges (i -> j) and distances (km)
#     ii_np = None
#     jj_np = None
#     dd_np = None
#     used_backend = "torch_cdist"
#     try:
#         from scipy.spatial import cKDTree  # type: ignore
#         # cKDTree expects finite float coords; use float64 for robustness
#         X64 = X_np.astype("float64", copy=False)
#         tree = cKDTree(X64)
#         kq = int(min(int(knn) + 1, int(n_events)))
#         # query returns (d, idx) with shapes (n_events,kq) for kq>1
#         try:
#             dists, nbrs = tree.query(X64, k=kq, workers=-1)
#         except TypeError:
#             # Older SciPy versions used `n_jobs` or had no parallelism kwarg.
#             try:
#                 dists, nbrs = tree.query(X64, k=kq, n_jobs=-1)  # type: ignore[call-arg]
#             except TypeError:
#                 dists, nbrs = tree.query(X64, k=kq)
#         # Drop the self-neighbor (distance 0). After knn=min(knn,n_events-1), kq>=2 here.
#         dists = dists[:, 1:]
#         nbrs = nbrs[:, 1:]
#         ii_np = np.repeat(np.arange(n_events, dtype=np.int64), int(knn))
#         jj_np = nbrs.reshape(-1).astype(np.int64, copy=False)
#         dd_np = dists.reshape(-1).astype(np.float32, copy=False)
#         used_backend = "scipy_ckdtree"
#     except Exception:
#         # Fallback: exact but O(n^2) memory. Keep for small problems / minimal environments.
#         D = torch.cdist(X_map, X_map).to(torch.float32)
#         D.fill_diagonal_(float("inf"))
#         vals, nbrs = torch.topk(D, k=knn, largest=False)
#         ii = torch.arange(n_events, dtype=torch.int64).unsqueeze(1).expand(-1, knn).reshape(-1)
#         jj = nbrs.reshape(-1).to(torch.int64)
#         dd = vals.reshape(-1).to(torch.float32)
#         ii_np = ii.numpy()
#         jj_np = jj.numpy()
#         dd_np = dd.numpy()
#         used_backend = "torch_cdist"

#     # Edge weights
#     ww_np = np.exp(-0.5 * (dd_np / float(ell_km)) ** 2).astype(np.float32, copy=False)

#     # Symmetrize and coalesce into undirected edges (u < v)
#     u = np.minimum(ii_np, jj_np).astype(np.int64, copy=False)
#     v = np.maximum(ii_np, jj_np).astype(np.int64, copy=False)
#     pairs = np.stack([u, v], axis=1)
#     uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
#     w_sum = np.zeros((uniq.shape[0],), dtype=np.float64)
#     w_cnt = np.zeros((uniq.shape[0],), dtype=np.float64)
#     np.add.at(w_sum, inv, ww_np.astype(np.float64))
#     np.add.at(w_cnt, inv, 1.0)
#     w_mean = (w_sum / np.maximum(1.0, w_cnt)).astype(np.float32)

#     # Drop self-edges if any slipped in
#     mask = (uniq[:, 0] != uniq[:, 1])
#     uniq = uniq[mask]
#     w_mean = w_mean[mask]

#     # Move graph to device
#     dev = state.device

#     # Expose graph to modeling via params (internal keys)

#     # Initialize latent b (station,event,phase) on device
#     b0 = torch.zeros((n_stations, n_events, 2), dtype=torch.float32, device=dev)

#     dt_s = time.time() - t0
#     info(
#         f"(knn_backend={used_backend}, dt={dt_s:.1f}s)",
#         section="LIKELIHOOD",
#     )
#     return


@torch.no_grad()
def _maybe_estimate_eikonet_v1d_speed(state: "LocateState") -> None:
    """
    Optional diagnostic: estimate an effective 1D reference speed curve v(z) from EikoNet.

    Idea:
      For isotropic media, ||∂T/∂x_src|| has units s/km and is approximately the slowness magnitude 1/v.
      EikoNet is a learned surrogate in a 3D model; averaging ||∂T/∂x_src|| over many source/receiver
      directions at the same depth yields a useful *effective* v(z) reference curve.

    This is primarily useful to map fractional velocity perturbations (dimensionless, e.g. 1–2%)
    into slowness amplitudes (s/km) via tau_u(z) ≈ vel_frac / v(z).

    Config (optional; new schema allows extra keys under inference.diagnostics):
      inference:
        diagnostics:
          eikonet_v1d:
            enabled: bool
            n_depth_bins: int
            n_events_per_bin: int
            n_stations_per_event: int
            batch_size: int
            seed: int
            outfile: str|null   # relative to checkpoint_dir if not abs
    """
    # We run this automatically when the slowness_inducing_gp model is configured in "fraction" mode,
    # because we need v(z) to map a dimensionless vel_frac field into slowness units (s/km).
    # Timer (helps users diagnose "hangs" in torchrun mode where non-rank0 appears silent).
    try:
        import time as _time
        _t0 = float(_time.time())
    except Exception:
        _t0 = None

    cfg = None
    try:
        want_auto = False
        want_auto = bool(mode == "slowness_inducing_gp" and units == "vel_frac")
        # Also enable auto v(z) estimation for the collapsed slowness covariance likelihood when it uses vel_frac.
        want_auto = bool(want_auto or (units2 == "vel_frac"))
        # Optional user override (legacy): inference.diagnostics.eikonet_v1d.enabled=true
        inf = state.params.get("inference", None)
        dg = inf.get("diagnostics", None) if isinstance(inf, dict) else None
        cfg0 = dg.get("eikonet_v1d", None) if isinstance(dg, dict) else None
        # If the user explicitly sets enabled=false, treat it as a hard override even in auto mode.
        if isinstance(cfg0, dict) and ("enabled" in cfg0) and (cfg0.get("enabled", None) is False):
            return
        want_cfg = bool(isinstance(cfg0, dict) and bool(cfg0.get("enabled", False)))
        if not (want_auto or want_cfg):
            return
        cfg = cfg0 if isinstance(cfg0, dict) else {}
    except Exception:
        return

    # DDP support: compute on rank0 then broadcast arrays to all ranks so vel_frac scaling is consistent.
    ddp_on = _ddp_enabled(state.params)
    ddp_main = (not ddp_on) or _ddp_is_main(state.params)
    dist_ok = False
    dist = None
    if ddp_on:
        try:
            import torch.distributed as dist  # type: ignore
            dist_ok = bool(dist.is_available()) and bool(dist.is_initialized())
        except Exception:
            dist_ok = False
            dist = None

    # Run once (per process)
    if bool(state.params.get("_eikonet_v1d_done", False)):
        return

    if ddp_on and dist_ok and (not ddp_main):
        # Non-main ranks: receive the arrays from rank0.
        try:
            # 1) sizes
            sz = torch.zeros((1,), device=state.device, dtype=torch.int64)
            dist.broadcast(sz, src=0)  # type: ignore[union-attr]
            K = int(sz.item())
            # 2) tensors (depth centers + vp + vs)
            zc = torch.empty((K,), device=state.device, dtype=torch.float32)
            vp = torch.empty((K,), device=state.device, dtype=torch.float32)
            vs = torch.empty((K,), device=state.device, dtype=torch.float32)
            dist.broadcast(zc, src=0)  # type: ignore[union-attr]
            dist.broadcast(vp, src=0)  # type: ignore[union-attr]
            dist.broadcast(vs, src=0)  # type: ignore[union-attr]
            # Store to params (as Python lists, matching the main-rank storage convention)
            state.params["_eikonet_v1d_depth_centers_km"] = zc.detach().cpu().tolist()
            state.params["_eikonet_v1d_vp_km_s"] = vp.detach().cpu().tolist()
            state.params["_eikonet_v1d_vs_km_s"] = vs.detach().cpu().tolist()
            state.params["_eikonet_v1d_done"] = True
        except Exception:
            # If broadcast fails, just continue without v(z); epoch_runner will fall back to constants.
            return
        return

    try:
        import os
        import math
        import numpy as np
        import torch
        import polars as pl
    except Exception:
        return

    # Parse config (auto mode uses defaults; cfg mode can override)
    try:
        n_depth_bins = int((cfg or {}).get("n_depth_bins", 20))
        n_events_per_bin = int((cfg or {}).get("n_events_per_bin", 64))
        n_stations_per_event = int((cfg or {}).get("n_stations_per_event", 8))
        batch_size = int((cfg or {}).get("batch_size", 2048))
        seed = int((cfg or {}).get("seed", 0))
        # For huge dtimes tables, building a unique station table can be expensive.
        # We only need a representative station pool for v(z); cap the number of rows we scan.
        max_station_rows = int((cfg or {}).get("max_station_rows", 5_000_000))
        # Default: do NOT write a file unless explicitly requested (avoid clutter).
        out_path = (cfg or {}).get("outfile", None)
    except Exception:
        n_depth_bins = 20
        n_events_per_bin = 64
        n_stations_per_event = 8
        batch_size = 2048
        seed = 0
        max_station_rows = 5_000_000
        out_path = None

    n_depth_bins = max(2, n_depth_bins)
    n_events_per_bin = max(1, n_events_per_bin)
    n_stations_per_event = max(1, n_stations_per_event)
    batch_size = max(32, batch_size)
    if max_station_rows < 0:
        max_station_rows = 0

    # Resolve output path
    try:
        out_path_s = str(out_path) if out_path is not None else ""
    except Exception:
        out_path_s = ""
    if not out_path_s:
        out_path_s = ""
    try:
        if out_path_s:
            if not os.path.isabs(out_path_s):
                base = str(state.params.get("checkpoint_dir", "."))
                out_path_s = os.path.join(base, out_path_s)
            os.makedirs(os.path.dirname(out_path_s) or ".", exist_ok=True)
    except Exception:
        pass

    # Build station XYZ table (km) in sta_idx order
    try:
        dt = state.dtimes
        if not isinstance(dt, pl.DataFrame):
            raise ValueError("missing dtimes DF")
        if "sta_idx" not in dt.columns:
            raise ValueError("dtimes missing sta_idx")
        # Optional cap for performance on huge tables
        if isinstance(max_station_rows, int) and max_station_rows > 0 and int(dt.shape[0]) > int(max_station_rows):
            dt = dt.head(int(max_station_rows))
        # Prefer XYZ columns if present; fallback to YY receiver columns (less robust).
        if ("X" in dt.columns) and ("Y" in dt.columns) and ("Z" in dt.columns):
            sta_xyz = (
                dt.select([pl.col("sta_idx"), pl.col("X"), pl.col("Y"), pl.col("Z")])
                .unique(subset=["sta_idx"], maintain_order=True)
                .sort("sta_idx")
            )
            xyz_np = sta_xyz.select([pl.col("X"), pl.col("Y"), pl.col("Z")]).to_numpy().astype(np.float32, copy=False)
        else:
            # YY is (dt, x_rec, y_rec, z_rec, phase); need station index mapping
            raise ValueError("dtimes missing station XYZ columns X,Y,Z")
        sta_xyz_t = torch.from_numpy(xyz_np).to(device=state.device, dtype=torch.float32)  # (S,3)
        n_st = int(sta_xyz_t.shape[0])
        if n_st <= 0:
            return
    except Exception as e:
        warn(f"EikoNet v1d: could not build station XYZ table: {e}", section="LIKELIHOOD")
        return

    # Current event locations (km)
    try:
        Xcur = (state.X_src + state.dX_src)[:, :3].detach().to(torch.float32)
        z = Xcur[:, 2].detach().cpu().numpy().astype(np.float32, copy=False)
        if z.size < 2:
            return
        zmin = float(np.nanmin(z))
        zmax = float(np.nanmax(z))
        if not (math.isfinite(zmin) and math.isfinite(zmax)) or zmax <= zmin:
            return
    except Exception:
        return

    # Depth bins (equal-width)
    edges = np.linspace(zmin, zmax, num=int(n_depth_bins) + 1, dtype=np.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])

    rng = np.random.default_rng(seed)
    # Pre-choose a station pool for sampling
    sta_all = np.arange(n_st, dtype=np.int64)

    # Model in eval mode for stability
    model = state.model
    try:
        was_training = bool(model.training)
    except Exception:
        was_training = False
    try:
        model.eval()
    except Exception:
        pass

    # Accumulators
    stats = {
        "P": {"sum": np.zeros((n_depth_bins,), dtype=np.float64), "sum2": np.zeros((n_depth_bins,), dtype=np.float64), "n": np.zeros((n_depth_bins,), dtype=np.int64)},
        "S": {"sum": np.zeros((n_depth_bins,), dtype=np.float64), "sum2": np.zeros((n_depth_bins,), dtype=np.float64), "n": np.zeros((n_depth_bins,), dtype=np.int64)},
    }

    def _accum(bin_idx: int, phase_key: str, gnorm: torch.Tensor) -> None:
        if gnorm.numel() <= 0:
            return
        g = gnorm.detach().cpu().to(torch.float64).numpy()
        stats[phase_key]["sum"][bin_idx] += float(g.sum())
        stats[phase_key]["sum2"][bin_idx] += float((g * g).sum())
        stats[phase_key]["n"][bin_idx] += int(g.size)

    # Iterate depth bins
    with torch.enable_grad():
        for bi in range(int(n_depth_bins)):
            lo = float(edges[bi]); hi = float(edges[bi + 1])
            mask = (z >= lo) & (z < hi if bi < (n_depth_bins - 1) else z <= hi)
            ev_idx = np.flatnonzero(mask).astype(np.int64, copy=False)
            if ev_idx.size == 0:
                continue
            k_ev = int(min(int(n_events_per_bin), int(ev_idx.size)))
            chosen_ev = rng.choice(ev_idx, size=k_ev, replace=False)

            # For each chosen event, choose stations and build batches for P and S
            # Build endpoint lists
            ev_rep = np.repeat(chosen_ev, repeats=int(n_stations_per_event)).astype(np.int64, copy=False)
            sta_rep = rng.choice(sta_all, size=int(ev_rep.size), replace=True).astype(np.int64, copy=False)
            if ev_rep.size == 0:
                continue

            ev_t = torch.from_numpy(ev_rep).to(device=state.device, dtype=torch.int64)
            sta_t = torch.from_numpy(sta_rep).to(device=state.device, dtype=torch.int64)
            src0 = Xcur.index_select(0, ev_t).detach()
            rec0 = sta_xyz_t.index_select(0, sta_t).detach()

            for phase_key, ph_val in (("P", 0.0), ("S", 1.0)):
                # Process in minibatches
                for i0 in range(0, int(src0.shape[0]), int(batch_size)):
                    i1 = min(i0 + int(batch_size), int(src0.shape[0]))
                    src = src0[i0:i1].clone().detach().requires_grad_(True)
                    rec = rec0[i0:i1]
                    ph = torch.full((int(i1 - i0), 1), float(ph_val), device=state.device, dtype=torch.float32)
                    coords = torch.cat([src, rec, ph], dim=1)
                    T = model(coords).reshape(-1)
                    g = torch.autograd.grad(T.sum(), src, retain_graph=False, create_graph=False, allow_unused=False)[0]
                    gnorm = torch.linalg.norm(g.to(torch.float32), dim=1).clamp_min(1e-12)
                    _accum(bi, phase_key, gnorm)

    # Restore training state
    try:
        if was_training:
            model.train()
    except Exception:
        pass

    # Compute means and v(z)
    vP = np.full((n_depth_bins,), np.nan, dtype=np.float32)
    vS = np.full((n_depth_bins,), np.nan, dtype=np.float32)
    sP = np.full((n_depth_bins,), np.nan, dtype=np.float32)
    sS = np.full((n_depth_bins,), np.nan, dtype=np.float32)
    for bi in range(int(n_depth_bins)):
        nP = int(stats["P"]["n"][bi])
        nS = int(stats["S"]["n"][bi])
        if nP > 0:
            m = float(stats["P"]["sum"][bi]) / float(nP)
            sP[bi] = float(m)
            vP[bi] = float(1.0 / max(m, 1e-12))
        if nS > 0:
            m = float(stats["S"]["sum"][bi]) / float(nS)
            sS[bi] = float(m)
            vS[bi] = float(1.0 / max(m, 1e-12))

    # Store to params for downstream use / notebooks
    try:
        state.params["_eikonet_v1d_depth_edges_km"] = edges.tolist()
        state.params["_eikonet_v1d_depth_centers_km"] = centers.tolist()
        state.params["_eikonet_v1d_slowness_p_s_per_km"] = sP.tolist()
        state.params["_eikonet_v1d_slowness_s_s_per_km"] = sS.tolist()
        state.params["_eikonet_v1d_vp_km_s"] = vP.tolist()
        state.params["_eikonet_v1d_vs_km_s"] = vS.tolist()
        state.params["_eikonet_v1d_outfile"] = str(out_path_s)
        state.params["_eikonet_v1d_done"] = True
    except Exception:
        pass

    # If DDP is active, broadcast the key arrays to other ranks (so vel_frac scaling is identical everywhere).
    if ddp_on and dist_ok and ddp_main:
        try:
            zc_t = torch.tensor(np.asarray(centers, dtype=np.float32).reshape(-1), device=state.device, dtype=torch.float32)
            vp_t = torch.tensor(np.asarray(vP, dtype=np.float32).reshape(-1), device=state.device, dtype=torch.float32)
            vs_t = torch.tensor(np.asarray(vS, dtype=np.float32).reshape(-1), device=state.device, dtype=torch.float32)
            K = int(zc_t.numel())
            sz = torch.tensor([K], device=state.device, dtype=torch.int64)
            dist.broadcast(sz, src=0)  # type: ignore[union-attr]
            dist.broadcast(zc_t, src=0)  # type: ignore[union-attr]
            dist.broadcast(vp_t, src=0)  # type: ignore[union-attr]
            dist.broadcast(vs_t, src=0)  # type: ignore[union-attr]
        except Exception:
            pass

    # Always log a brief summary once (even if we don't write an NPZ),
    # because these values directly set the scaling for vel_frac slowness models.
    try:
        vp_med = float(np.nanmedian(vP)) if np.isfinite(np.nanmedian(vP)) else float("nan")
        vs_med = float(np.nanmedian(vS)) if np.isfinite(np.nanmedian(vS)) else float("nan")
        sp_med = float(np.nanmedian(sP)) if np.isfinite(np.nanmedian(sP)) else float("nan")
        ss_med = float(np.nanmedian(sS)) if np.isfinite(np.nanmedian(sS)) else float("nan")
        dt_s = None
        try:
            if _t0 is not None:
                dt_s = float(_time.time() - float(_t0))
        except Exception:
            dt_s = None
        info(
            "EikoNet v1d speed estimated: "
            f"z[km]=[{zmin:.3g},{zmax:.3g}] bins={int(n_depth_bins)} "
            f"vp_med≈{vp_med:.3g}km/s vs_med≈{vs_med:.3g}km/s "
            f"(slowness_med P≈{sp_med:.3g}s/km S≈{ss_med:.3g}s/km) "
            + (f"outfile={out_path_s}" if out_path_s else "outfile=<none>")
            + (f" dt={dt_s:.1f}s" if (dt_s is not None and np.isfinite(dt_s)) else ""),
            section="LIKELIHOOD",
        )
        # Heuristic sanity check: flag wildly implausible speeds (often indicates unit mismatch).
        if (np.isfinite(vp_med) and (vp_med > 100.0 or vp_med < 0.1)) or (np.isfinite(vs_med) and (vs_med > 100.0 or vs_med < 0.05)):
            warn(
                "EikoNet v1d: estimated speeds look implausible. "
                "This can cause vel_frac models to rescale latents incorrectly. "
                "Double-check coordinate units (km) and the EikoNet checkpoint/domain scale.",
                section="LIKELIHOOD",
            )
    except Exception:
        pass

    # Optional: save NPZ (best-effort) only if outfile was provided
    if out_path_s:
        try:
            np.savez_compressed(
                out_path_s,
                depth_edges_km=edges,
                depth_centers_km=centers,
                slowness_p_s_per_km=sP,
                slowness_s_s_per_km=sS,
                vp_km_s=vP,
                vs_km_s=vS,
                n_samples_p=stats["P"]["n"],
                n_samples_s=stats["S"]["n"],
            )
            info(
                "EikoNet v1d speed estimated: "
                f"z[km]=[{zmin:.3g},{zmax:.3g}] bins={int(n_depth_bins)} "
                f"vp_med~{float(np.nanmedian(vP)):.3g} vs_med~{float(np.nanmedian(vS)):.3g} "
                f"outfile={out_path_s}",
                section="LIKELIHOOD",
            )
        except Exception as e:
            warn(f"EikoNet v1d save failed: {e}", section="LIKELIHOOD")


@torch.no_grad()
def _maybe_precompute_shared_event_re_whitening(state: LocateState) -> None:
    try:
        if not bool(state.params.get("_shared_event_re_enabled", False)):
            return
        if not bool(state.params.get("_shared_event_re_enabled", False)):
            return
        if not bool(state.params.get("_shared_event_re_solver_precompute_enabled", False)):
            return
        event_batches = bool(state.params.get("event_batch_enable", False))
        if (not event_batches) and bool(state.params.get("batch_shuffle", True)):
            return
    except Exception:
        return

    device_pref = str(state.params.get("_shared_event_re_solver_precompute_device", "gpu")).strip().lower()
    if device_pref not in {"gpu", "cpu"}:
        device_pref = "gpu"
    if device_pref == "cpu":
        # CPU precompute is not supported for reuse on GPU without extra transfers.
        return

    batch_size = int(getattr(state, "batch_size_sgld", 0) or state.params.get("batch_size_sgld", 0) or 0)
    N = int(state.N)
    if N <= 0:
        return

    cache = state.params.get("_shared_event_re_solver_cache", None)
    if not isinstance(cache, dict):
        cache = {}

    grouping = str(state.params.get("_shared_event_re_grouping", "station_phase")).strip().lower()
    if grouping in {"stationphase", "station-phase"}:
        grouping = "station_phase"
    if grouping != "station_phase":
        return
    if not isinstance(state.row_station_index, torch.Tensor):
        return

    tau_ps = state.params.get("_shared_event_re_tau_s", [0.0, 0.0])
    tau_p = float(tau_ps[0]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
    tau_s = float(tau_ps[1]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
    jitter0 = float(state.params.get("_shared_event_re_jitter0", 1e-8))
    max_rows_per_group = int(state.params.get("_shared_event_re_max_rows_per_group", 200000))
    max_nodes_per_group = int(state.params.get("_shared_event_re_max_nodes_per_group", 512))
    solver = str(state.params.get("_shared_event_re_solver_kind", "pcg")).strip().lower()

    edge_weighting = str(state.params.get("_shared_event_re_edge_weight_mode", "uniform")).strip().lower()
    edge_weight_ell_km = float(state.params.get("_shared_event_re_edge_weight_ell_km", 1.0))
    edge_weight_eps_km = float(state.params.get("_shared_event_re_edge_weight_eps_km", 1e-3))
    edge_weight_power = float(state.params.get("_shared_event_re_edge_weight_power", 1.0))
    edge_weight_scale_km = float(state.params.get("_shared_event_re_edge_weight_scale_km", 1.0))
    edge_weight_global_scale = float(state.params.get("_shared_event_re_edge_weight_global_scale", 1.0))
    edge_weight_normalize = bool(state.params.get("_shared_event_re_edge_weight_normalize", False))

    X_event = None
    if edge_weighting in {"distance_rbf", "distance_linear", "distance_power"}:
        X_event = state.params.get("_shared_event_re_edge_weight_X_event", None)
        if not isinstance(X_event, torch.Tensor) or int(X_event.shape[0]) != int(state.X_src.shape[0]):
            X_event = (state.X_src + state.dX_src)[:, :3].detach().to(device=state.device, dtype=torch.float32)
            state.params["_shared_event_re_edge_weight_X_event"] = X_event

    σp, σs = _current_noise_scales(state)
    σp = σp.to(device=state.device)
    σs = σs.to(device=state.device)

    batch_specs: list[tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    if bool(event_batches):
        try:
            from spider.core.batching import _ensure_owner_buckets

            seed0 = int(state.params.get("runtime_seed", 0))
            epoch_seed = int(seed0 + int(state.params.get("_runtime_epoch_index", 0)))
            max_edges = int(state.params.get("event_batch_max_edges_sgld", state.params.get("event_batch_max_edges", 0)))
            events_per_batch = int(state.params.get("event_batch_size", 256))
            reorder_all = bool(state.params.get("event_bucket_reorder_all", False))
            reuse_epochs = int(state.params.get("event_bucket_reuse_epochs", 1))
            _ensure_owner_buckets(
                state,
                epoch_index=epoch_seed,
                events_per_batch=events_per_batch,
                max_edges_per_batch=max_edges,
                reorder_all=reorder_all,
                reuse_epochs=reuse_epochs,
            )
            if state._bucket_offsets is None or state._bucket_II is None or state._bucket_YY is None:
                return
            if not isinstance(getattr(state, "_bucket_station_index", None), torch.Tensor):
                return
            nb = int(state._bucket_offsets.numel() - 1)
            for b in range(nb):
                i_start = int(state._bucket_offsets[b].item())
                i_end = int(state._bucket_offsets[b + 1].item())
                if i_end <= i_start:
                    continue
                idx_b = state._bucket_II[i_start:i_end].to(device=state.device, dtype=torch.int64)
                YY_b = state._bucket_YY[i_start:i_end].to(device=state.device)
                sta_b = state._bucket_station_index[i_start:i_end].to(device=state.device, dtype=torch.int64)
                batch_specs.append(
                    (
                        {
                            "mode": "event_bucket",
                            "batch_id": -1,
                            "batch_i0": -1,
                            "batch_i1": -1,
                            "bucket_id": int(b),
                            "bucket_gen": int(getattr(state, "_bucket_last_epoch", -1) or -1),
                            "is_shuffled": False,
                            "epoch_index": int(state.params.get("_runtime_epoch_index", -1)),
                            "batch_seq": int(b),
                        },
                        idx_b,
                        YY_b,
                        sta_b,
                    )
                )
        except Exception:
            return
    else:
        if batch_size <= 0:
            return
        n_batches = int(math.ceil(float(N) / float(batch_size)))
        for b in range(n_batches):
            i_start = b * batch_size
            i_end = min(i_start + batch_size, N)
            idx_b = state.II[i_start:i_end].to(device=state.device, dtype=torch.int64)
            YY_b = state.YY[i_start:i_end].to(device=state.device)
            sta_b = state.row_station_index[i_start:i_end].to(device=state.device, dtype=torch.int64)
            batch_specs.append(
                (
                    {
                        "mode": "standard",
                        "batch_id": int(b),
                        "batch_i0": int(i_start),
                        "batch_i1": int(i_end),
                        "bucket_id": -1,
                        "bucket_gen": -1,
                        "is_shuffled": False,
                        "epoch_index": int(state.params.get("_runtime_epoch_index", -1)),
                        "batch_seq": int(b),
                    },
                    idx_b,
                    YY_b,
                    sta_b,
                )
            )

    for b, (batch_context, idx_b, YY_b, sta_b) in enumerate(batch_specs):
        ph_id = torch.where(
            YY_b[:, 4] < 0.5,
            torch.zeros_like(YY_b[:, 4], dtype=torch.int64),
            torch.ones_like(YY_b[:, 4], dtype=torch.int64),
        )
        keys = (sta_b * 2) + ph_id

        cache_key_extra = None
        cache_key = _make_cache_key(
            edge_weighting=edge_weighting,
            edge_weight_ell_km=edge_weight_ell_km,
            edge_weight_eps_km=edge_weight_eps_km,
            edge_weight_power=edge_weight_power,
            edge_weight_scale_km=edge_weight_scale_km,
            edge_weight_global_scale=edge_weight_global_scale,
            edge_weight_normalize=edge_weight_normalize,
            tau_p=float(tau_p),
            tau_s=float(tau_s),
            max_rows_per_group=int(max_rows_per_group),
            max_nodes_per_group=int(max_nodes_per_group),
            solver=str(solver),
            batch_context=batch_context,
            cache_key_extra=cache_key_extra,
        )
        if cache_key in cache:
            continue
        cache_entry = build_whitening_cache_entry(
            idx=idx_b,
            keys=keys,
            ph_id=ph_id,
            sigma_p=σp,
            sigma_s=σs,
            tau_p=float(tau_p),
            tau_s=float(tau_s),
            jitter0=float(jitter0),
            max_rows_per_group=int(max_rows_per_group),
            max_nodes_per_group=int(max_nodes_per_group),
            solver=str(solver),
            edge_weighting=edge_weighting,
            edge_weight_ell_km=float(edge_weight_ell_km),
            edge_weight_eps_km=float(edge_weight_eps_km),
            edge_weight_power=float(edge_weight_power),
            edge_weight_scale_km=float(edge_weight_scale_km),
            edge_weight_global_scale=float(edge_weight_global_scale),
            edge_weight_normalize=bool(edge_weight_normalize),
            X_event=X_event,
        )
        cache[cache_key] = cache_entry
        try:
            cache_max = int(state.params.get("_shared_event_re_solver_cache_max_entries", 0) or 0)
            if cache_max > 0 and len(cache) > cache_max:
                cache.pop(next(iter(cache)))
        except Exception:
            pass

    state.params["_shared_event_re_solver_cache"] = cache


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

    # Phase 1: MAP estimation with Adam optimizer
    if phase == "phase1":
        if wandb_logger:
            wandb_logger.start_phase("phase1")
        _phase1_map_warmup(state, start_epoch, wandb_logger)
        phase = "phase2"

    # Optional residual-based outlier removal at the start of Phase 2 (post-MAP).
    # Prefer running this during data prep (prepare_input_dfs) immediately after the linearization filter,
    # so we avoid doing an extra expensive residual pass here.
    if (
        ckpt is None
        and phase == "phase2"
        and bool(state.params.get("residual_filter_enable", False))
        and not bool(state.params.get("_residual_filter_applied_in_prepare_input_dfs", False))
        and not bool(state.params.get("_residual_filter_applied_in_phase2", False))
    ):
        _pre_filter_outlier_residuals(state, use_current_dX=True)
        state.params["_residual_filter_applied_in_phase2"] = True
        _print_initial_residual_stats(state)


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
                _log(f"Warning: could not load sampler state: {e}")
        else:
            # Phase 1 checkpoint: we will (optionally) transplant preconditioner state from Adam below.
            pass
        # One-time fix: enforce current JSON hyperparameters after resume
        try:
            # Enforce required param-group keys expected by our current samplers.
            lr_user = _lr_for_phase(state.params, phase)
            backend = str(state.params["sampler_backend"]).strip().lower()
            if backend in {"psgld", "sghmc"}:
                lr_json = lr_user / float(max(1, int(state.N)))
            else:
                lr_json = lr_user

            beta_json = float(state.params["sampler_beta"])
            eps_json = float(state.params["sampler_eps"])
            temp_json = float(state.params["sampler_temperature"])
            precond_json = bool(state.params["sampler_preconditioning"])
            precond_type_json = str(state.params.get("sampler_preconditioner", "none")).strip().lower()
            if precond_json and precond_type_json in {"none", "false", ""}:
                precond_type_json = "rmsprop"
            if (not precond_json) or precond_type_json in {"none", "false", ""}:
                precond_type_json = "none"

            for g in sampler.param_groups:
                g.setdefault("beta", beta_json)
                g.setdefault("eps", eps_json)
                g.setdefault("temperature", temp_json)
                # Always honor current JSON preconditioning flags on resume.
                g["preconditioning"] = precond_json
                g["preconditioner"] = precond_type_json
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
            _log(f"Warning: could not apply JSON params on resume: {e}")

    # Phase 2
    if phase == "phase2":
        if wandb_logger:
            wandb_logger.start_phase("phase2")
        try:
            _maybe_precompute_shared_event_re_whitening(state)
        except Exception as e:
            warn(f"shared_event_re whitening precompute failed: {e}", section="LIKELIHOOD")
        # Transfer preconditioning state from Adam to sampler if coming from phase 1
        if ckpt is None or ckpt.get("phase") == "phase1":
            try:
                from spider.optim.backends import transplant_from_adam_if_supported
                transplant_from_adam_if_supported(state.optimizer, sampler)
                if (not _ddp_enabled(state.params)) or _ddp_is_main(state.params):
                    _log("Transferred preconditioning state from Adam to sampler (if supported)")
            except Exception as e:
                _log(f"Warning: could not transplant preconditioner from Adam (backend): {e}")
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

    # Always print a pre-optimization residual sanity check at the *initial* catalog locations
    # (ΔX=0), even when the residual filter was already applied during data prep.
    # This is useful for quickly comparing datasets/configs and spotting busted inputs.
    try:
        if ckpt is None and start_epoch == 0:
            ddp_main = (not _ddp_enabled(state.params)) or _ddp_is_main(state.params)
            if ddp_main:
                _print_initial_residual_stats(state)
    except Exception:
        pass

    if wandb_logger:
        wandb_logger.start_phase("phase1")

    _phase1_map_warmup(state, start_epoch=start_epoch, wandb_logger=wandb_logger)

    # In torchrun/DDP mode, only rank0 writes bundle outputs.
    ddp_main = (not _ddp_enabled(state.params)) or _ddp_is_main(state.params)
    if bundle_out and ddp_main:
        # Dump "exact input to Phase 2": post-phase1 filtered dtimes/origins, MAP ΔX, and Adam state
        try:
            save_phase2_bundle(
                path=str(bundle_out),
                origins0=state.origins0,
                dtimes=state.dtimes,
                dX_src=state.dX_src.detach(),
                    # Bundle should contain only locations + picks; everything else is rebuilt cleanly at sampling start.
                    params=None,
                    noise_log_scale=None,
                    phase1_optimizer_state_dict=None,
                    global_step_count=0,
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

    # IMPORTANT: Phases 2–4 must ALWAYS use the current run config (JSON → schema materialization),
    # not whatever params happened to be stored inside the Phase-2 bundle.
    #
    # The bundle is used only for its dataset payload (origins0/dtimes) and initial MAP state.
    run_params = params
    state = _build_initial_state(run_params, bun.origins0, bun.dtimes, model, device)
    try:
        _maybe_precompute_shared_event_re_whitening(state)
    except Exception as e:
        warn(f"shared_event_re whitening precompute failed: {e}", section="LIKELIHOOD")

    # Mirror the important "reset/clear samples" behavior from `_resume_or_initialize`,
    # but without ever resuming Phase 2–4 checkpoints (bundle start is always Phase 2 epoch 0).
    try:
        reset_batch_numbers = bool(state.params.get("reset_batch_numbers", False))
        clear_samples_on_reset = bool(state.params.get("clear_samples_on_reset", False))
        if reset_batch_numbers:
            info("Resetting batch numbers to 0 (reset_batch_numbers=True)", section="RUN")
            clear_checkpoint_files(state.params)
            if clear_samples_on_reset:
                ok = clear_samples_file(state.params)
                if ok:
                    try:
                        sp = str(state.params.get("samples_outfile", state.params.get("io", {}).get("samples_outfile", "samples.h5")))
                    except Exception:
                        sp = "samples.h5"
                    info(f"Deleted samples file '{sp}' (clear_samples_on_reset=True)", section="SAMPLES")
                if not ok:
                    try:
                        sp = str(state.params.get("samples_outfile", state.params.get("io", {}).get("samples_outfile", "samples.h5")))
                    except Exception:
                        sp = "samples.h5"
                    warn(f"Failed to delete samples file '{sp}' (clear_samples_on_reset=True). New batches may append.", section="SAMPLES")
            state.sample_count = 0
        else:
            if clear_samples_on_reset:
                info("Clearing samples file (clear_samples_on_reset=True)", section="SAMPLES")
                ok = clear_samples_file(state.params)
                if ok:
                    try:
                        sp = str(state.params.get("samples_outfile", state.params.get("io", {}).get("samples_outfile", "samples.h5")))
                    except Exception:
                        sp = "samples.h5"
                    info(f"Deleted samples file '{sp}' (clear_samples_on_reset=True)", section="SAMPLES")
                if not ok:
                    try:
                        sp = str(state.params.get("samples_outfile", state.params.get("io", {}).get("samples_outfile", "samples.h5")))
                    except Exception:
                        sp = "samples.h5"
                    warn(f"Failed to delete samples file '{sp}' (clear_samples_on_reset=True). New batches may append.", section="SAMPLES")
            # Continue batch numbering if samples file exists
            state.sample_count = get_next_sample_count(state.params)
            if state.sample_count > 0:
                info(f"Continuing from batch number {state.sample_count} (existing samples file found)", section="SAMPLES")
    except Exception:
        pass

    # Restore MAP solution
    state.dX_src.data.copy_(bun.dX_src.to(device=state.device, dtype=torch.float32))
    state.global_step_count = 0

    # From here, behave as if Phase 1 just finished.
    phase = "phase2"
    start_epoch = 0
    skip_saving_first_epoch = False
    ckpt = None

    # Optional residual-based outlier removal at the start of Phase 2 (post-MAP).
    if (
        bool(state.params.get("residual_filter_enable", False))
        and not bool(state.params.get("_residual_filter_applied_in_prepare_input_dfs", False))
        and not bool(state.params.get("_residual_filter_applied_in_phase2", False))
    ):
        _pre_filter_outlier_residuals(state, use_current_dX=True)
        state.params["_residual_filter_applied_in_phase2"] = True
        _print_initial_residual_stats(state)


    sampler = _setup_sampler(state)

    if wandb_logger:
        wandb_logger.start_phase("phase2")
    # Transfer preconditioning state from Adam to sampler if coming from phase 1 (bundle always is)
    try:
        from spider.optim.backends import transplant_from_adam_if_supported
        transplant_from_adam_if_supported(state.optimizer, sampler)
        if (not _ddp_enabled(state.params)) or _ddp_is_main(state.params):
            _log("Transferred preconditioning state from Adam to sampler (if supported)")
    except Exception as e:
        _log(f"Warning: could not transplant preconditioner from Adam (backend): {e}")

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
