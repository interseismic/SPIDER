from __future__ import annotations
from spider.utils.console import info, warn

import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import json
import h5py

from spider.core.config_schema import validate_and_materialize_block1
from spider.core.priors_config import validate_and_materialize_priors



# Standardized stdout helper
def _log(*parts, section: str = "PLOT", **_kwargs) -> None:
    msg = " ".join(str(p) for p in parts)
    low = msg.strip().lower()
    if low.startswith("warning") or low.startswith("error"):
        warn(msg, section=section)
    else:
        info(msg, section=section)

def _load_noise_log_scale_series(checkpoint_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load time series of noise log-scales from checkpoints.

    Returns
    -------
    log_scales : np.ndarray, shape (T, 2)
        Rows over time; columns are [log_sigma_p, log_sigma_s].
    epochs : np.ndarray, shape (T,)
        Associated epoch indices parsed from filenames if possible, else 0..T-1.
    """
    paths: List[str] = []
    paths.extend(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*_epoch_*.pth")))
    paths = sorted(set(paths))
    if not paths:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int64)

    series: List[np.ndarray] = []
    epochs: List[int] = []
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=True)
        if "noise_log_scale" not in d:
            continue
        x = d["noise_log_scale"]
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size < 2:
            continue
        series.append(x[:2])
        # Extract epoch number from filename:
        try:
            base = os.path.basename(p)
            if "_epoch_" in base:
                n = int(base.split("_epoch_")[-1].split(".")[0])
            else:
                n = len(series) - 1
        except Exception:
            n = len(series) - 1
        epochs.append(n)

    if not series:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.stack(series, axis=0), np.asarray(epochs, dtype=np.int64)


def _load_noise_from_hdf5(samples_outfile: str, thin: int = 1) -> np.ndarray:
    """
    Load concatenated noise log-scales [log_sigma_p, log_sigma_s] from HDF5 batches.
    Returns array of shape (S, 2) where S is total samples (after thinning).
    """
    if not samples_outfile or not isinstance(samples_outfile, str):
        return np.empty((0, 2), dtype=np.float32)
    try:
        with h5py.File(samples_outfile, "r") as f:
            rows: List[np.ndarray] = []
            for name, grp in f.items():
                if not isinstance(grp, h5py.Group):
                    continue
                if not name.startswith("batch_"):
                    continue
                if "log_sigma_p" in grp and "log_sigma_s" in grp:
                    p = grp["log_sigma_p"][:]
                    s = grp["log_sigma_s"][:]
                    if thin is not None and int(thin) > 1:
                        p = p[::int(thin)]
                        s = s[::int(thin)]
                    if p.shape[0] and s.shape[0]:
                        rows.append(np.stack([p, s], axis=1).astype(np.float32))
            if not rows:
                return np.empty((0, 2), dtype=np.float32)
            return np.concatenate(rows, axis=0)
    except Exception as e:
        _log(f"_load_noise_from_hdf5: failed to read '{samples_outfile}': {e}")
        return np.empty((0, 2), dtype=np.float32)


def plot_noise_scale_posterior_vs_prior(
    params: Dict | str,
    out_path: str,
    checkpoint_dir: Optional[str] = None,
    names: Tuple[str, str] = ("P", "S"),
    num_bins: int = 40,
    use_density: bool = False,
    hdf5_thin: int = 1,
    burn_in: int = 0,
) -> bool:
    """
    Plot prior vs posterior for log sigma (noise scales) for P and S.

    Parameters
    ----------
    params : dict or str
        SPIDER run parameters dict OR a path to a params.json file. Uses keys:
          - 'checkpoint_dir' (if checkpoint_dir not provided)
          - 'samples_outfile' (preferred source; reads noise samples from HDF5)
          - 'noise_prior' (e.g., 'lognormal')
          - 'noise_prior_loc': [mu_p, mu_s]
          - 'noise_prior_scale': [sd_p, sd_s]
    out_path : str
        Path to save the figure.
    checkpoint_dir : str, optional
        Directory with checkpoints; defaults to params['checkpoint_dir'].
    names : tuple(str, str)
        Labels for the two phases.
    num_bins : int
        Histogram bins for posterior.

    Returns
    -------
    success : bool
        True if plot was successfully written, else False.
    """
    # Allow passing a path to params.json
    if isinstance(params, str):
        try:
            with open(params, "r") as f:
                params = json.load(f)
        except Exception as e:
            _log(f"plot_noise_scale_posterior_vs_prior: failed to read params from '{params}': {e}")
            return False

    # If params is a nested config dict (new schema), materialize legacy flat keys for this plot.
    # Do NOT attempt to re-validate if the caller already passed a materialized dict (which contains
    # legacy flat keys and would be rejected by strict validators).
    if isinstance(params, dict) and ("io" in params) and ("dtime_file" not in params):
        try:
            params = validate_and_materialize_block1(params)
            params = validate_and_materialize_priors(params)
        except Exception as e:
            _log(f"plot_noise_scale_posterior_vs_prior: invalid params config: {e}")
            return False

    # Import matplotlib lazily so importing spider.plotting works even in environments without a
    # working matplotlib binary (common with NumPy 2.x ABI mismatches).
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise ImportError(
            "matplotlib could not be imported. This is often due to a NumPy/matplotlib binary "
            "compatibility mismatch (e.g., NumPy 2.x with an older matplotlib build). "
            "Fix by reinstalling matplotlib built for your NumPy version, or pinning `numpy<2`.\n"
            f"Original error: {type(e).__name__}: {e}"
        )

    # Prefer HDF5 noise samples if available
    samples_outfile = params.get("samples_outfile", "")
    hdf5_logs = _load_noise_from_hdf5(samples_outfile, thin=int(hdf5_thin))
    if hdf5_logs.size > 0:
        log_scales = hdf5_logs  # shape (S, 2)
    else:
        # Fallback to checkpoints time series (sparser)
        cdir = checkpoint_dir or params.get("checkpoint_dir", "checkpoints/")
        log_scales, epochs = _load_noise_log_scale_series(cdir)
        if log_scales.size == 0:
            _log(f"plot_noise_scale_posterior_vs_prior: no noise samples found in HDF5 or checkpoints.")
            return False

    # Apply burn-in (drop first burn_in samples)
    b = int(max(0, burn_in))
    if b > 0 and log_scales.shape[0] > b:
        log_scales = log_scales[b:]

    # Prior in log-space if configured
    prior_type = str(params.get("noise_prior", "none")).lower()
    has_prior = prior_type in {"lognormal", "log_normal"} and "noise_prior_loc" in params and "noise_prior_scale" in params
    mu = np.asarray(params.get("noise_prior_loc", [0.0, 0.0]), dtype=np.float64)
    sd = np.asarray(params.get("noise_prior_scale", [1.0, 1.0]), dtype=np.float64)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for i, ax in enumerate(axes):
        x = log_scales[:, i]
        counts, edges, _ = ax.hist(
            x, bins=num_bins, density=use_density, alpha=0.5, color="#1f77b4", label="posterior (log σ)"
        )
        m = float(np.median(x))

        if has_prior:
            xs = np.linspace(min(np.min(x), mu[i] - 4.0 * sd[i]), max(np.max(x), mu[i] + 4.0 * sd[i]), 400)
            prior_pdf = (1.0 / (sd[i] * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((xs - mu[i]) / sd[i]) ** 2)
            if use_density:
                prior_curve = prior_pdf
            else:
                # Convert pdf to expected counts per bin: pdf * N * bin_width
                N = len(x)
                bin_w = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0
                prior_curve = prior_pdf * N * bin_w
            ax.plot(xs, prior_curve, color="#d62728", linewidth=1.5, label=f"prior N({mu[i]:.2f},{sd[i]:.2f}²)")
            # z-score
            z = (m - mu[i]) / sd[i] if sd[i] > 0 else np.nan
            ax.set_title(f"{names[i]} (z={z:.2f})")
        else:
            ax.set_title(f"{names[i]}")

        ax.set_xlabel("log σ")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("density" if use_density else "count")
    fig.suptitle("Noise scale prior vs posterior (log-space)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    _log(f"Wrote noise scale prior/posterior plot to {out_path}")
    return True


