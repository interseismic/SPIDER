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
Post-processing utilities: Wasserstein distance between event prior and posterior.

We compute a per-event 2-Wasserstein distance W2 between:
  - Prior: Gaussian N(0, diag(std^2)) from config (priors.event.params.std)
  - Posterior: Gaussian approximation fitted to MCMC samples of (dX, dY, dZ, dt)

This is useful for burn-in selection: sweep burn-in values and check when W2 stats stabilize.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from spider.io.samples import read_all_samples


_DIM_NAMES = {0: "x", 1: "y", 2: "z", 3: "t"}


def _load_event_prior_std(config_path: str) -> np.ndarray:
    with open(config_path, "r") as f:
        cfg = json.load(f)
    # Accept nested config style (required by current hard-break schema)
    std = (
        cfg.get("model", {})
        .get("priors", {})
        .get("event", {})
        .get("params", {})
        .get("std", None)
    )
    if not isinstance(std, list) or len(std) != 4:
        raise ValueError(f"Expected model.priors.event.params.std to be a list[4], got: {std!r}")
    v = np.asarray([float(x) for x in std], dtype=np.float64)
    if not np.all(v > 0.0):
        raise ValueError(f"Event prior std must be >0, got: {v}")
    return v


# Public aliases (avoid importing underscored internals from other modules)
DIM_NAMES = _DIM_NAMES
load_event_prior_std = _load_event_prior_std


def _parse_burn(burn: str, n: int) -> int:
    """
    burn can be:
      - integer string: number of samples
      - float string in [0,1): fraction of samples
    """
    s = str(burn).strip()
    if s == "":
        return 0
    try:
        if "." in s or "e" in s.lower():
            frac = float(s)
            if not (0.0 <= frac < 1.0):
                raise ValueError
            return int(round(frac * n))
        else:
            k = int(s)
            return max(0, min(k, n))
    except Exception as e:
        raise ValueError(f"Invalid burn='{burn}'. Use an integer (count) or fraction in [0,1).") from e


def _parse_dims(dims: str | List[int] | Tuple[int, ...]) -> List[int]:
    if isinstance(dims, (list, tuple)):
        out = [int(x) for x in dims]
    else:
        s = str(dims).strip()
        if s == "":
            out = [0, 1, 2, 3]
        else:
            out = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    if not out:
        raise ValueError("dims must be non-empty (e.g. '0,1,2' for XYZ)")
    for d in out:
        if d not in (0, 1, 2, 3):
            raise ValueError(f"dims entries must be in {{0,1,2,3}}, got {out}")
    # preserve order, drop duplicates
    seen = set()
    uniq: List[int] = []
    for d in out:
        if d not in seen:
            uniq.append(d)
            seen.add(d)
    return uniq


def _stack_event_delta_samples(samples: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      event_ids: (N,) array[str]
      X: (N, S, 4) float32
    """
    def _to_numpy(x: Any) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    event_ids = samples.get("event_ids", None)
    if event_ids is None:
        raise ValueError("Samples dict missing 'event_ids' (unexpected HDF5 schema?)")
    # Stored as (N, S_total)
    X = samples.get("X", None)
    Y = samples.get("Y", None)
    Z = samples.get("Z", None)
    T = samples.get("delta_t", None)
    if X is None or Y is None or Z is None or T is None:
        raise ValueError("Samples dict missing one of {'X','Y','Z','delta_t'}")
    X = _to_numpy(X).astype(np.float32, copy=False)
    Y = _to_numpy(Y).astype(np.float32, copy=False)
    Z = _to_numpy(Z).astype(np.float32, copy=False)
    T = _to_numpy(T).astype(np.float32, copy=False)
    if X.ndim != 2 or Y.shape != X.shape or Z.shape != X.shape or T.shape != X.shape:
        raise ValueError(f"Unexpected sample shapes: X{X.shape} Y{Y.shape} Z{Z.shape} delta_t{T.shape}")
    N, S = X.shape
    out = np.stack([X, Y, Z, T], axis=2)  # (N, S, 4)
    return _to_numpy(event_ids), out


def gaussian_w2_prior_to_posterior(
    post_samples: torch.Tensor,
    *,
    prior_std: torch.Tensor,
    jitter: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute per-event 2-Wasserstein distance W2 between:
      prior N(0, diag(prior_std^2)) and
      posterior Gaussian approximation from samples.

    Args:
      post_samples: (N, S, 4) tensor
      prior_std: (4,) tensor
      jitter: diagonal jitter added to posterior cov estimate for numerical stability

    Returns:
      w2: (N,) tensor
      post_mean: (N,4) tensor
      post_cov: (N,4,4) tensor
    """
    if post_samples.ndim != 3:
        raise ValueError(f"post_samples must be (N,S,D), got {tuple(post_samples.shape)}")
    if prior_std.ndim != 1:
        raise ValueError(f"prior_std must be 1D (D,), got {tuple(prior_std.shape)}")
    N, S, D = post_samples.shape
    if prior_std.shape != (D,):
        raise ValueError(f"prior_std must be (D,), got {tuple(prior_std.shape)} for D={D}")
    if S < 2:
        raise ValueError(f"Need at least 2 samples after burn/thin, got S={S}")

    x = post_samples.to(dtype=torch.float64)
    mu = x.mean(dim=1)  # (N,4)
    xc = x - mu.unsqueeze(1)  # (N,S,4)
    cov = torch.einsum("nsi,nsj->nij", xc, xc) / float(S - 1)  # (N,4,4)
    cov = cov + (float(jitter) * torch.eye(D, dtype=cov.dtype, device=cov.device).unsqueeze(0))

    # Prior covariance and its sqrt (diagonal)
    s = prior_std.to(dtype=torch.float64, device=cov.device)
    C0 = torch.diag_embed(s * s).expand(N, D, D)  # (N,4,4)
    sqrtC0 = torch.diag_embed(s).expand(N, D, D)  # (N,4,4)

    # A = sqrt(C0) * C1 * sqrt(C0)
    A = sqrtC0 @ cov @ sqrtC0
    # sqrt(A) via batched eigh
    evals, evecs = torch.linalg.eigh(A)
    evals = evals.clamp_min(0.0)
    sqrtA = evecs @ torch.diag_embed(torch.sqrt(evals)) @ evecs.transpose(-1, -2)

    # W2^2 = ||m||^2 + Tr(C0 + C1 - 2 sqrtA)
    mean_term = (mu * mu).sum(dim=1)  # (N,)
    tr = torch.diagonal(C0 + cov - 2.0 * sqrtA, dim1=-2, dim2=-1).sum(dim=1)
    w2_sq = (mean_term + tr).clamp_min(0.0)
    w2 = torch.sqrt(w2_sq)
    return w2.to(dtype=torch.float64), mu, cov


def compute_event_wasserstein(
    *,
    samples_outfile: str,
    config_path: str,
    burn: str = "0",
    thin: int = 1,
    dims: str | List[int] = "0,1,2,3",
    device: str = "cpu",
    jitter: float = 1e-10,
) -> Dict[str, Any]:
    """
    Compute per-event W2 from prior to posterior (Gaussian approx) for a given burn/thin.

    dims selects which components of (dX,dY,dZ,dt) to include:
      - "0,1,2" for hypocenter-only (XYZ)
      - "0,1,2,3" for full space+time (default, backward compatible)
    """
    prior_std_np = _load_event_prior_std(config_path)
    dims_list = _parse_dims(dims)
    samples = read_all_samples({"samples_outfile": samples_outfile}, backend="numpy", thin=int(thin))
    event_ids, X = _stack_event_delta_samples(samples)  # (N,), (N,S,4)
    N, S_total, _ = X.shape
    burn_n = _parse_burn(burn, S_total)
    if burn_n >= S_total:
        raise ValueError(f"burn={burn} removes all samples (S_total={S_total})")
    X2 = X[:, burn_n:, :][:, :, dims_list]  # (N,S,D)

    t = torch.from_numpy(X2).to(torch.float64)
    if device and str(device).lower() != "cpu":
        t = t.to(device)
    prior_std_t = torch.from_numpy(prior_std_np[dims_list]).to(torch.float64)
    if device and str(device).lower() != "cpu":
        prior_std_t = prior_std_t.to(device)

    w2, mu, cov = gaussian_w2_prior_to_posterior(t, prior_std=prior_std_t, jitter=float(jitter))

    # Also provide per-dimension 1D W2 (Gaussian vs Gaussian) as a quick sanity check
    post_std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(0.0))  # (N,4)
    # For 1D Gaussians: W2^2 = (mu)^2 + (sigma_post - sigma_prior)^2
    w2_1d = torch.sqrt((mu * mu) + (post_std - prior_std_t.view(1, -1)).pow(2)).detach().cpu().numpy()

    out = {
        "event_ids": event_ids,
        "w2": w2.detach().cpu().numpy().astype(np.float64, copy=False),
        "post_mean": mu.detach().cpu().numpy().astype(np.float64, copy=False),
        "post_std": post_std.detach().cpu().numpy().astype(np.float64, copy=False),
        "w2_1d": w2_1d.astype(np.float64, copy=False),
        "meta": {
            "burn": burn,
            "burn_n": int(burn_n),
            "thin": int(thin),
            "dims": dims_list,
            "n_events": int(N),
            "n_samples_total": int(S_total),
            "n_samples_used": int(X2.shape[1]),
            "prior_std": prior_std_np.tolist(),
            "prior_std_used": prior_std_np[dims_list].tolist(),
        },
    }
    return out


def compute_event_wasserstein_from_samples(
    *,
    samples: Dict[str, Any],
    prior_std: List[float],
    burn_in: int = 0,
    thin: int = 1,
    dims: str | List[int] = "0,1,2,3",
    device: str = "cpu",
    jitter: float = 1e-10,
) -> Dict[str, Any]:
    """
    Compute per-event W2 from prior to posterior (Gaussian approx) using an in-memory samples dict.

    This is the same computation as `compute_event_wasserstein`, but avoids reading HDF5.

    Args:
      samples: dict with keys {'event_ids','X','Y','Z','delta_t'} (each (N,S))
      prior_std: list[4] prior std for (dX,dY,dZ,dt)
      burn_in: integer burn-in samples to drop (count, not fraction)
      thin: thinning factor to apply after burn-in
      dims: which dims to include (e.g. "0,1,2" for XYZ)
    """
    prior_std_np = np.asarray([float(x) for x in prior_std], dtype=np.float64)
    if prior_std_np.shape != (4,) or not np.all(prior_std_np > 0.0):
        raise ValueError(f"prior_std must be list[4] of positive floats, got: {prior_std!r}")
    dims_list = _parse_dims(dims)

    event_ids, X = _stack_event_delta_samples(samples)  # (N,), (N,S,4)
    N, S_total, _ = X.shape
    burn_n = int(burn_in)
    if burn_n < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_n}")
    if burn_n >= S_total:
        raise ValueError(f"burn_in={burn_n} removes all samples (S_total={S_total})")
    X2 = X[:, burn_n:, :]
    if int(thin) > 1:
        X2 = X2[:, :: int(thin), :]
    X2 = X2[:, :, dims_list]  # (N,S,D)

    t = torch.from_numpy(np.asarray(X2, dtype=np.float32)).to(torch.float64)
    if device and str(device).lower() != "cpu":
        t = t.to(device)
    prior_std_t = torch.from_numpy(prior_std_np[dims_list]).to(torch.float64)
    if device and str(device).lower() != "cpu":
        prior_std_t = prior_std_t.to(device)

    w2, mu, cov = gaussian_w2_prior_to_posterior(t, prior_std=prior_std_t, jitter=float(jitter))
    post_std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(0.0))
    w2_1d = torch.sqrt((mu * mu) + (post_std - prior_std_t.view(1, -1)).pow(2)).detach().cpu().numpy()

    return {
        "event_ids": event_ids,
        "w2": w2.detach().cpu().numpy().astype(np.float64, copy=False),
        "post_mean": mu.detach().cpu().numpy().astype(np.float64, copy=False),
        "post_std": post_std.detach().cpu().numpy().astype(np.float64, copy=False),
        "w2_1d": w2_1d.astype(np.float64, copy=False),
        "meta": {
            "burn_in": int(burn_n),
            "thin": int(thin),
            "dims": dims_list,
            "n_events": int(N),
            "n_samples_total": int(S_total),
            "n_samples_used": int(X2.shape[1]),
            "prior_std": prior_std_np.tolist(),
            "prior_std_used": prior_std_np[dims_list].tolist(),
        },
    }


def _write_csv(path: str, res: Dict[str, Any]) -> None:
    import csv

    event_ids = res["event_ids"]
    w2 = res["w2"]
    mu = res["post_mean"]
    std = res["post_std"]
    w2_1d = res["w2_1d"]
    dims = list(res.get("meta", {}).get("dims", list(range(mu.shape[1]))))
    names = [_DIM_NAMES.get(int(d), f"d{int(d)}") for d in dims]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["event_id", "w2"]
        header += [f"mu_{n}" for n in names]
        header += [f"std_{n}" for n in names]
        header += [f"w2_{n}" for n in names]
        w.writerow(header)
        for i in range(len(event_ids)):
            row = [str(event_ids[i]), float(w2[i])]
            row += [float(mu[i, j]) for j in range(mu.shape[1])]
            row += [float(std[i, j]) for j in range(std.shape[1])]
            row += [float(w2_1d[i, j]) for j in range(w2_1d.shape[1])]
            w.writerow(row)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compute per-event prior↔posterior Wasserstein (Gaussian-approx W2).")
    ap.add_argument("--samples", required=True, help="Path to samples_outfile HDF5 (e.g. SPIDER_samples_unclamped.h5)")
    ap.add_argument("--config", required=True, help="Path to SPIDER config JSON (for priors.event.params.std).")
    ap.add_argument("--burn", default="0", help="Burn-in: integer samples or fraction in [0,1). Example: 0.2 or 5000")
    ap.add_argument("--thin", type=int, default=1, help="Thinning factor (keep every nth sample)")
    ap.add_argument("--dims", default="0,1,2,3", help="Dims of (dX,dY,dZ,dt) to include. Use '0,1,2' for hypocenter-only.")
    ap.add_argument("--spatial_only", action="store_true", help="Shortcut for --dims 0,1,2")
    ap.add_argument("--device", default="cpu", help="Device for computation (cpu or cuda)")
    ap.add_argument("--jitter", type=float, default=1e-10, help="Diagonal jitter for posterior covariance")
    ap.add_argument("--out_csv", default="", help="Optional output CSV path")
    args = ap.parse_args(argv)

    res = compute_event_wasserstein(
        samples_outfile=args.samples,
        config_path=args.config,
        burn=str(args.burn),
        thin=int(args.thin),
        dims=("0,1,2" if bool(args.spatial_only) else str(args.dims)),
        device=str(args.device),
        jitter=float(args.jitter),
    )
    w2 = res["w2"]
    meta = res["meta"]
    # Summary prints for burn-in tuning
    q = np.quantile(w2, [0.5, 0.9, 0.99])
    _log(
        "W2(prior→posterior) summary: "
        f"n_events={meta['n_events']} n_samples_used={meta['n_samples_used']} "
        f"median={q[0]:.4g} p90={q[1]:.4g} p99={q[2]:.4g} max={float(np.max(w2)):.4g}"
    )
    if args.out_csv:
        _write_csv(args.out_csv, res)
        _log(f"Wrote per-event W2 CSV: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


