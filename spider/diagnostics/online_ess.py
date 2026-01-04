"""
Online ESS / integrated autocorrelation time (IACT) diagnostics for a subset of events.

Designed to run during sampling using the in-memory `state.samples` buffer (list of dX snapshots).
Avoids pandas/matplotlib; uses PyTorch FFT-based autocorrelation similar to analysis/results.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class OnlineESSResult:
    n_samples: int
    n_events: int
    n_dims: int
    dims: Tuple[int, ...]
    max_lag: int
    ess_median: float
    ess_p10: float
    ess_p90: float
    tau_median: float
    tau_p10: float
    tau_p90: float
    ess_median_by_dim: Tuple[float, ...]
    ess_p10_by_dim: Tuple[float, ...]
    ess_p90_by_dim: Tuple[float, ...]
    tau_median_by_dim: Tuple[float, ...]
    tau_p10_by_dim: Tuple[float, ...]
    tau_p90_by_dim: Tuple[float, ...]

    def to_metrics(self, prefix: str = "ess_online") -> Dict[str, float]:
        out = {
            f"{prefix}/n_samples": float(self.n_samples),
            f"{prefix}/n_events": float(self.n_events),
            f"{prefix}/n_dims": float(self.n_dims),
            f"{prefix}/max_lag": float(self.max_lag),
            f"{prefix}/ess_median": float(self.ess_median),
            f"{prefix}/ess_p10": float(self.ess_p10),
            f"{prefix}/ess_p90": float(self.ess_p90),
            f"{prefix}/tau_median": float(self.tau_median),
            f"{prefix}/tau_p10": float(self.tau_p10),
            f"{prefix}/tau_p90": float(self.tau_p90),
        }
        # Per-dimension summaries (median/p10/p90 across events), keyed by the actual dim id.
        for j, dim_id in enumerate(self.dims):
            out[f"{prefix}/ess_median_dim{int(dim_id)}"] = float(self.ess_median_by_dim[j])
            out[f"{prefix}/ess_p10_dim{int(dim_id)}"] = float(self.ess_p10_by_dim[j])
            out[f"{prefix}/ess_p90_dim{int(dim_id)}"] = float(self.ess_p90_by_dim[j])
            out[f"{prefix}/tau_median_dim{int(dim_id)}"] = float(self.tau_median_by_dim[j])
            out[f"{prefix}/tau_p10_dim{int(dim_id)}"] = float(self.tau_p10_by_dim[j])
            out[f"{prefix}/tau_p90_dim{int(dim_id)}"] = float(self.tau_p90_by_dim[j])
        return out


def _autocorr_fft(chains: torch.Tensor, max_lag: int) -> torch.Tensor:
    """
    chains: (B, T) float tensor
    returns: acf (B, max_lag+1) with acf[:,0]=1
    """
    B, T = chains.shape
    device = chains.device
    x = chains - chains.mean(dim=1, keepdim=True)
    var = x.var(dim=1, unbiased=True).clamp_min(1e-12)

    # next pow2 >= 2*T
    n_fft = 1 << int(np.ceil(np.log2(max(2 * T, 2))))
    fx = torch.fft.rfft(x, n=n_fft, dim=1)
    psd = fx * torch.conj(fx)
    ac = torch.fft.irfft(psd, n=n_fft, dim=1)[:, : T]
    # unbiased normalization for each lag
    denom = torch.arange(T, 0, -1, device=device, dtype=ac.dtype).view(1, T)
    ac = ac / denom
    ac = ac / var.view(B, 1)

    L = int(min(max_lag, T - 1))
    return ac[:, : L + 1]


def _ess_from_acf(acf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ESS and tau_int from ACF using Geyer's initial positive sequence heuristic.
    acf: (B, L+1), acf[:,0]=1
    Returns: (ess, tau_int) both (B,)
    """
    B, Lp1 = acf.shape
    T = None  # unknown here; caller computes ESS scale from window length
    # Use initial positive sequence on pairs: rho_{2k-1}+rho_{2k}
    # We'll do a simple "stop when pair sum < 0" rule.
    r = acf[:, 1:]  # (B, L)
    if r.numel() == 0:
        tau = torch.ones((B,), device=acf.device, dtype=acf.dtype)
        ess = torch.ones((B,), device=acf.device, dtype=acf.dtype)
        return ess, tau
    L = r.shape[1]
    # if L is odd, drop last for pairing
    L2 = (L // 2) * 2
    r2 = r[:, :L2].reshape(B, -1, 2)  # (B, K, 2)
    pair = r2.sum(dim=2)  # (B, K)
    # cumulative mask: keep pairs until first negative
    keep = (pair > 0.0)
    # Build per-row truncation index
    # If all keep True -> use all, else up to first False
    # We compute k_max = first index where keep==False, else K
    K = keep.shape[1]
    # argmax on (~keep) gives first True in not_keep, but if none True -> 0; handle separately
    not_keep = ~keep
    first_bad = torch.argmax(not_keep.to(torch.int64), dim=1)  # (B,)
    any_bad = not_keep.any(dim=1)
    k_max = torch.where(any_bad, first_bad, torch.full_like(first_bad, K))

    # sum autocorr up to 2*k_max lags
    # tau = 1 + 2 * sum_{t=1..M} r_t, where M = 2*k_max
    tau = torch.ones((B,), device=acf.device, dtype=acf.dtype)
    for i in range(B):
        m = int(2 * int(k_max[i].item()))
        if m > 0:
            tau[i] = 1.0 + 2.0 * r[i, :m].sum()
    tau = tau.clamp_min(1.0)
    # ESS needs T; caller will scale by T/tau. Here we return tau only; ESS placeholder.
    ess = 1.0 / tau
    return ess, tau


def compute_online_ess(
    *,
    samples: Sequence[torch.Tensor],
    event_idx: torch.Tensor,
    dims: Sequence[int] = (0, 1, 2),
    window: int = 512,
    max_lag: Optional[int] = None,
) -> OnlineESSResult:
    """
    samples: list of tensors, each (N,4) on CPU (dX snapshots)
    event_idx: (K,) int64 indices into events
    dims: which columns of dX to use (default X,Y,Z)
    window: how many most-recent samples to use
    max_lag: optional max lag; default min(1000, T//4)
    """
    if len(samples) == 0:
        raise ValueError("samples is empty")
    if event_idx.numel() == 0:
        raise ValueError("event_idx is empty")
    if any((d < 0 or d > 3) for d in dims):
        raise ValueError("dims must be subset of {0,1,2,3}")
    T = int(min(int(window), len(samples)))
    if T < 8:
        raise ValueError("need at least 8 samples to estimate autocorrelation")
    tail = samples[-T:]

    # Stack only selected events/dims: (T, K, D)
    K = int(event_idx.numel())
    D = int(len(list(dims)))
    sel = event_idx.to(dtype=torch.int64, device="cpu")
    out = torch.empty((T, K, D), dtype=torch.float32, device="cpu")
    for t, s in enumerate(tail):
        if not isinstance(s, torch.Tensor):
            raise TypeError("samples must be a sequence of torch.Tensors")
        x = s.index_select(0, sel).to(dtype=torch.float32, device="cpu")
        out[t] = x[:, list(dims)]

    # Flatten chains: (B, T) where B = K*D
    chains = out.permute(1, 2, 0).reshape(K * D, T).contiguous()

    if max_lag is None:
        max_lag = int(min(1000, T // 4))
    max_lag = int(max(1, min(max_lag, T - 1)))

    acf = _autocorr_fft(chains, max_lag=max_lag)  # (B, L+1)
    _, tau = _ess_from_acf(acf)  # tau (B,)
    ess = (float(T) / tau).clamp(min=1.0, max=float(T))

    # Summaries
    ess_np = ess.detach().cpu().numpy()
    tau_np = tau.detach().cpu().numpy()

    dims_list = [int(d) for d in dims]
    # Reshape back to (K, D) so we can report per-dimension stats across events.
    ess_kd = ess.reshape(K, D).detach().cpu().numpy()
    tau_kd = tau.reshape(K, D).detach().cpu().numpy()
    ess_median_by_dim = tuple(float(np.median(ess_kd[:, j])) for j in range(D))
    ess_p10_by_dim = tuple(float(np.quantile(ess_kd[:, j], 0.10)) for j in range(D))
    ess_p90_by_dim = tuple(float(np.quantile(ess_kd[:, j], 0.90)) for j in range(D))
    tau_median_by_dim = tuple(float(np.median(tau_kd[:, j])) for j in range(D))
    tau_p10_by_dim = tuple(float(np.quantile(tau_kd[:, j], 0.10)) for j in range(D))
    tau_p90_by_dim = tuple(float(np.quantile(tau_kd[:, j], 0.90)) for j in range(D))
    return OnlineESSResult(
        n_samples=int(T),
        n_events=int(K),
        n_dims=int(D),
        dims=tuple(dims_list),
        max_lag=int(max_lag),
        ess_median=float(np.median(ess_np)),
        ess_p10=float(np.quantile(ess_np, 0.10)),
        ess_p90=float(np.quantile(ess_np, 0.90)),
        tau_median=float(np.median(tau_np)),
        tau_p10=float(np.quantile(tau_np, 0.10)),
        tau_p90=float(np.quantile(tau_np, 0.90)),
        ess_median_by_dim=ess_median_by_dim,
        ess_p10_by_dim=ess_p10_by_dim,
        ess_p90_by_dim=ess_p90_by_dim,
        tau_median_by_dim=tau_median_by_dim,
        tau_p10_by_dim=tau_p10_by_dim,
        tau_p90_by_dim=tau_p90_by_dim,
    )


