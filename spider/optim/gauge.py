"""
Gauge projection helpers for SPIDER samplers.

Context:
- In a pure differential-time likelihood, there is a near-null translation mode in event locations:
  shifting *all* events by a constant vector can be weakly constrained or unconstrained depending on priors.
- A centroid prior is one way to regularize this. Another is to "gauge fix" by projecting out the
  translation mode in the sampler updates (hard constraint on the mean update).

This module provides a lightweight projection used by samplers (pSGLD/SGHMC/AdaptiveSGHMC) to:
- project the mean gradient out before preconditioner statistics update, and
- optionally project the injected noise / momentum to avoid centroid random-walk.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch


def _normalize_dims(dims: Optional[Iterable[int]], *, D: int) -> list[int]:
    if dims is None:
        return []
    out: list[int] = []
    for d in dims:
        try:
            di = int(d)
        except Exception:
            continue
        if 0 <= di < int(D):
            out.append(di)
    # unique, stable
    seen = set()
    uniq: list[int] = []
    for di in out:
        if di in seen:
            continue
        seen.add(di)
        uniq.append(di)
    return uniq


@torch.no_grad()
def project_event_mean_inplace(
    x: torch.Tensor,
    *,
    dims: Sequence[int] = (0, 1, 2),
    mode: str = "global",
    cluster_ids: Optional[torch.Tensor] = None,
    cluster_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    In-place projection to remove the mean (translation mode) over events.

    x is expected to be event-indexed with shape (N_events, D). Only dims in `dims`
    are projected. Other dims are left unchanged.

    mode:
    - "global": subtract global mean over events
    - "cluster": subtract per-cluster mean if cluster_ids/counts are compatible; else fall back to global
    """
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        return x
    N, D = int(x.shape[0]), int(x.shape[1])
    if N <= 0 or D <= 0:
        return x
    dims_i = _normalize_dims(dims, D=D)
    if not dims_i:
        return x

    mode_s = str(mode).strip().lower()
    want_cluster = (mode_s == "cluster")
    if want_cluster:
        if not (isinstance(cluster_ids, torch.Tensor) and cluster_ids.ndim == 1 and int(cluster_ids.numel()) == N):
            want_cluster = False
        if not (isinstance(cluster_counts, torch.Tensor) and cluster_counts.numel() > 0):
            want_cluster = False
        # Avoid implicit device copies inside the optimizer hot-path.
        if want_cluster and (cluster_ids.device != x.device):
            want_cluster = False
        if want_cluster and (cluster_counts.device != x.device):
            want_cluster = False

    if want_cluster:
        cid = cluster_ids.to(dtype=torch.int64)
        # cluster_counts may be (K,1) or (K,)
        cc = cluster_counts.view(-1).to(dtype=x.dtype)
        K = int(cc.numel())
        if K <= 0:
            want_cluster = False
        else:
            # sums: (K,D)
            sums = torch.zeros((K, D), device=x.device, dtype=x.dtype)
            sums.index_add_(0, cid, x)
            denom = cc.clamp_min(1.0).view(-1, 1)
            means = sums / denom
            mu = means.index_select(0, cid)  # (N,D)
            for d in dims_i:
                x[:, d].sub_(mu[:, d])
            return x

    # Global
    mu = x.mean(dim=0, keepdim=True)  # (1,D)
    for d in dims_i:
        x[:, d].sub_(mu[:, d])
    return x

