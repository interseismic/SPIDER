import numpy as _np
import numpy as np
from typing import Union, Any


def _lazy_import_plt():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception as e:
        raise ImportError(
            "matplotlib could not be imported. This is often due to a NumPy/matplotlib binary "
            "compatibility mismatch (e.g., NumPy 2.x with an older matplotlib build). "
            "Fix by reinstalling matplotlib built for your NumPy version, or pinning `numpy<2`.\n"
            f"Original error: {type(e).__name__}: {e}"
        )


def _is_event_samples_summary(x: Any) -> bool:
    """
    Duck-typing check for EventSamplesSummary-like objects without importing spider.analysis
    (which may pull in heavy optional deps like pandas).
    """
    return (
        hasattr(x, "X")
        and hasattr(x, "Y")
        and hasattr(x, "Z")
        and hasattr(x, "T")
        and hasattr(x, "cat_dd")
    )


_SPATIAL_COORDS = {"X", "Y", "Z"}


def _coord_factor(coord: str, *, units: str) -> float:
    """Scaling for display: spatial coords are km->m when units='meters'; time stays seconds."""
    c = str(coord)
    u = str(units).lower()
    if c in _SPATIAL_COORDS:
        return 1000.0 if u.startswith("meter") else 1.0
    return 1.0


def _coord_unit_label(coord: str, *, units: str) -> str:
    c = str(coord)
    u = str(units).lower()
    if c in _SPATIAL_COORDS:
        return "m" if u.startswith("meter") else "km"
    if c in {"T", "delta_t"}:
        return "s"
    return ""


def plot_event_distributions(samples_data: Union[dict, Any],
                             n_rows=6,
                             n_cols=4,
                             coords=("X", "Y", "Z"),
                             burn_in=0,
                             units="meters",
                             kde_bw="silverman",
                             n_eval=512,
                             event_indices=None,
                             random_select=False,
                             figsize=(14, 10),
                             sharex=True,
                             sharey=True,
                             xlim=None,
                             ylim=None,
                             show_legend=True,
                             title=None):
    """Plot per-event 1D KDE distributions for selected coordinates.

    Args:
        samples_data: dict from `spider.io.samples.read_all_samples` with keys 'event_ids', 'X','Y','Z', ...
        n_rows, n_cols: grid size of subplots.
        coords: tuple/list among fields present in samples_data (default ("X","Y","Z")).
        burn_in: number of initial samples to ignore.
        units: 'meters' (default, assumes km inputs) or 'kilometers'.
        kde_bw: bandwidth for KDEpy FFTKDE.
        n_eval: number of evaluation points for KDE curve.
        event_indices: optional list of event indices to plot; if None uses first n_rows*n_cols or random.
        random_select: if True and event_indices is None, randomly choose events.
        figsize, sharex, sharey: matplotlib figure options.
        xlim, ylim: axis limits; if None, reasonable defaults per units.
        show_legend: whether to draw legend once.
        title: optional suptitle.

    Returns:
        (fig, ax): matplotlib objects.
    """
    try:
        from KDEpy import FFTKDE as _FFTKDE
        _use_fft = True
    except Exception:
        from scipy.stats import gaussian_kde as _gaussian_kde
        _use_fft = False

    # Support EventSamplesSummary-like objects or legacy dict
    is_summary = _is_event_samples_summary(samples_data)
    if is_summary:
        invalid = [c for c in coords if c not in ("X", "Y", "Z", "T")]
        if invalid:
            raise KeyError(f"When using EventSamplesSummary, only 'X','Y','Z','T' are supported for coords (got {invalid})")
        # Determine shape from first available coord
        shape_source = None
        for c in coords:
            arr = getattr(samples_data, c, None)
            if arr is not None:
                shape_source = arr
                break
        if shape_source is None:
            raise ValueError("EventSamplesSummary does not contain any of the requested coords")
        n_events, n_samples = shape_source.shape
        event_ids = samples_data.cat_dd["evid"].values if samples_data.cat_dd is not None and "evid" in samples_data.cat_dd.columns else _np.arange(n_events)
    else:
        assert 'event_ids' in samples_data, "samples_data must include 'event_ids'"
        for c in coords:
            key = "delta_t" if str(c) == "T" else c
            if key not in samples_data:
                raise KeyError(f"Coordinate '{c}' not found in samples_data")
        c0 = "delta_t" if str(coords[0]) == "T" else coords[0]
        n_events, n_samples = samples_data[c0].shape
        event_ids = samples_data['event_ids']
    num_to_plot = min(n_rows * n_cols, n_events)

    # Select events
    if event_indices is None:
        if random_select:
            rng = _np.random.default_rng()
            event_indices = rng.choice(n_events, size=num_to_plot, replace=False)
        else:
            event_indices = _np.arange(num_to_plot)
    else:
        event_indices = _np.asarray(event_indices)
        if event_indices.size > num_to_plot:
            event_indices = event_indices[:num_to_plot]

    # Units/xlim: if user plots non-spatial coords (e.g. 'T'), require an explicit xlim.
    only_spatial = set([str(c) for c in coords]).issubset(_SPATIAL_COORDS)
    if xlim is None and only_spatial:
        factor0 = _coord_factor("X", units=units)
        default_xlim = (-300, 300) if factor0 == 1000.0 else (-0.3, 0.3)
        xlim = default_xlim
    if xlim is None and (not only_spatial):
        raise ValueError("For non-spatial coords (e.g. 'T'), please pass an explicit xlim (units differ).")

    _plt = _lazy_import_plt()
    fig, ax = _plt.subplots(nrows=n_rows, ncols=n_cols, sharex=sharex, sharey=sharey, figsize=figsize)
    ax = _np.atleast_2d(ax)

    def _kde_curve(values):
        vals = _np.asarray(values)
        if vals.ndim != 1:
            vals = vals.ravel()
        # keep only finite
        vals = vals[_np.isfinite(vals)]
        if vals.size == 0:
            grid = _np.linspace(-1.0, 1.0, n_eval)
            return grid, _np.zeros_like(grid)
        # subtract mean as in original code
        vals = vals - _np.mean(vals)
        # Build grid that covers the data with margin to satisfy FFTKDE
        data_min = float(vals.min())
        data_max = float(vals.max())
        margin = 0.05 * (data_max - data_min + 1e-9)
        lo_data = data_min - margin
        hi_data = data_max + margin
        if xlim is not None:
            lo = min(xlim[0], lo_data)
            hi = max(xlim[1], hi_data)
        else:
            lo, hi = lo_data, hi_data
        if not _np.isfinite(lo) or not _np.isfinite(hi) or lo == hi:
            lo, hi = lo_data, hi_data
        grid = _np.linspace(lo, hi, n_eval)
        if _use_fft:
            try:
                f_eval = _FFTKDE(bw=kde_bw).fit(vals).evaluate(grid)
                return grid, f_eval
            except Exception:
                # Fallback to gaussian_kde or histogram
                pass
        try:
            from scipy.stats import gaussian_kde as _gaussian_kde  # lazy import fallback
            kde = _gaussian_kde(vals)
            return grid, kde(grid)
        except Exception:
            hist, edges = _np.histogram(vals, bins=n_eval, range=(lo, hi), density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            return centers, hist

    coordinates_colors = {"X": "tab:blue", "Y": "tab:orange", "Z": "tab:green",
                           "longitude": "tab:purple", "latitude": "tab:red", "depth": "tab:brown",
                           "delta_t": "tab:olive", "T": "tab:olive"}

    for idx, ev_idx in enumerate(event_indices):
        r = idx // n_cols
        c = idx % n_cols
        axis = ax[r, c]

        for coord in coords:
            coord_s = str(coord)
            dict_key = "delta_t" if (not is_summary and coord_s == "T") else coord_s
            f = _coord_factor(coord_s, units=units)
            unit = _coord_unit_label(coord_s, units=units)
            if is_summary:
                series = getattr(samples_data, coord_s)[ev_idx, :]
            else:
                series = samples_data[dict_key][ev_idx, burn_in:]
            x_vals = (series * float(f))
            x_kde, y_kde = _kde_curve(x_vals)
            lab = f"{coord_s} ({unit})" if unit else coord_s
            axis.plot(x_kde, y_kde, label=lab, color=coordinates_colors.get(coord_s, None))

        # Title with event id if available
        try:
            evid = event_ids[ev_idx]
        except Exception:
            evid = str(ev_idx)
        axis.set_title(str(evid), fontsize=9)

    # Label the bottom row
    for c in range(n_cols):
        if only_spatial:
            ax[-1, c].set_xlabel(f"Relative Uncertainty ({'meters' if _coord_factor('X', units=units)==1000.0 else 'km'})")
        else:
            ax[-1, c].set_xlabel("Relative Uncertainty")
    # Optional defaults similar to original
    for axes_row in ax:
        for axis in axes_row:
            axis.set_xlim(xlim)
            if ylim is not None:
                axis.set_ylim(ylim)

    if show_legend:
        ax[0, 0].legend(loc='upper right', fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, ax


def build_catalog_posterior_mean(samples_data: Union[dict, Any]):
    """
    Build a catalog of posterior means for each event.
    """
    if _is_event_samples_summary(samples_data):
        if samples_data.cat_dd is None:
            raise ValueError("EventSamplesSummary must include cat_dd to build the catalog")
        import polars as pl
        # Use cat_dd directly
        return pl.DataFrame(samples_data.cat_dd)

    event_samples = samples_data
    lats = event_samples["latitude"].mean(axis=1)
    lons = event_samples["longitude"].mean(axis=1)
    deps = event_samples["depth"].mean(axis=1)
    X = event_samples["X"].mean(axis=1)
    Y = event_samples["Y"].mean(axis=1)
    Z = event_samples["Z"].mean(axis=1)
    sigma_x = 0.5*(np.percentile(event_samples["X"], 99.5) - np.percentile(event_samples["X"], 0.5))
    sigma_y = 0.5*(np.percentile(event_samples["Y"], 99.5) - np.percentile(event_samples["Y"], 0.5))
    sigma_z = 0.5*(np.percentile(event_samples["Z"], 99.5) - np.percentile(event_samples["Z"], 0.5))
    import polars as pl
    catalog = pl.DataFrame(data={"latitude": lats, "longitude": lons, "depth": deps, "X": X, "Y": Y, "Z": Z,
                           "sigma_x": sigma_x, "sigma_y": sigma_y, "sigma_z": sigma_z})
    return catalog


def plot_event_chains(samples_data: Union[dict, Any],
                      n_rows=2,
                      n_cols=2,
                      coords=("X", "Y", "Z"),
                      event_indices=None,
                      event_ids=None,
                      random_select=False,
                      burn_in=0,
                      thin=1,
                      units="meters",
                      figsize=(14, 8),
                      sharex=True,
                      sharey=False,
                      xlim=None,
                      ylim=None,
                      alpha=0.9,
                      linewidth=1.0,
                      show_mean=True,
                      mean_style=None,
                      title=None):
    """Plot MCMC chains (sample index vs value) for selected events.

    Args:
        samples_data: dict from `spider.io.samples.read_all_samples` with keys 'event_ids', 'X','Y','Z', ...
        n_rows, n_cols: grid size of subplots (events per figure = n_rows*n_cols).
        coords: tuple/list among fields present in samples_data to plot per panel.
        event_indices: optional list of integer event indices to plot.
        event_ids: optional list of event ID strings corresponding to samples_data['event_ids'].
        random_select: if True and no explicit selection, randomly choose events.
        burn_in: number of initial samples to discard.
        thin: keep every 'thin'-th sample after burn-in.
        units: 'meters' (km->m) or 'kilometers'.
        figsize, sharex, sharey: matplotlib options.
        xlim, ylim: axis limits; if None, determined automatically.
        alpha, linewidth: line appearance for chains.
        show_mean: if True, draw horizontal mean line for each coordinate.
        mean_style: dict of matplotlib kwargs for mean line (defaults set if None).
        title: optional suptitle.

    Returns:
        (fig, ax): matplotlib figure and axes grid.
    """
    # Validate fields
    is_summary = _is_event_samples_summary(samples_data)
    if is_summary:
        invalid = [c for c in coords if c not in ("X", "Y", "Z", "T")]
        if invalid:
            raise KeyError(f"When using EventSamplesSummary, only 'X','Y','Z','T' are supported for coords (got {invalid})")
        shape_source = None
        for c in coords:
            arr = getattr(samples_data, c, None)
            if arr is not None:
                shape_source = arr
                break
        if shape_source is None:
            raise ValueError("EventSamplesSummary does not contain any of the requested coords")
        n_events, n_samples = shape_source.shape
        event_ids_resolved = samples_data.cat_dd["evid"].values if samples_data.cat_dd is not None and "evid" in samples_data.cat_dd.columns else _np.arange(n_events)
    else:
        assert 'event_ids' in samples_data, "samples_data must include 'event_ids'"
        for c in coords:
            key = "delta_t" if str(c) == "T" else c
            if key not in samples_data:
                raise KeyError(f"Coordinate '{c}' not found in samples_data")
        c0 = "delta_t" if str(coords[0]) == "T" else coords[0]
        n_events, n_samples = samples_data[c0].shape
        event_ids_resolved = samples_data['event_ids']
    num_panels = min(n_rows * n_cols, n_events)

    # Resolve event selection
    selection = None
    if event_indices is not None:
        selection = _np.asarray(event_indices, dtype=int)
    elif event_ids is not None:
        # Map event IDs to indices
        eid_arr = _np.asarray(event_ids_resolved)
        idxs = []
        for eid in event_ids:
            matches = _np.nonzero(eid_arr == eid)[0]
            if matches.size == 0:
                continue
            idxs.append(matches[0])
        if len(idxs) == 0:
            raise ValueError("None of the provided event_ids were found in samples_data['event_ids']")
        selection = _np.asarray(idxs, dtype=int)
    else:
        if random_select:
            rng = _np.random.default_rng()
            selection = rng.choice(n_events, size=num_panels, replace=False)
        else:
            selection = _np.arange(num_panels)

    if selection.size > num_panels:
        selection = selection[:num_panels]

    # Units scaling is per-coordinate (spatial coords can be shown in m/km; time stays seconds)

    # Prepare figure
    _plt = _lazy_import_plt()
    fig, ax = _plt.subplots(nrows=n_rows, ncols=n_cols, sharex=sharex, sharey=sharey, figsize=figsize)
    ax = _np.atleast_2d(ax)

    # Colors per coordinate
    coordinates_colors = {"X": "tab:blue", "Y": "tab:orange", "Z": "tab:green",
                           "longitude": "tab:purple", "latitude": "tab:red", "depth": "tab:brown",
                           "delta_t": "tab:olive", "T": "tab:olive"}

    if mean_style is None:
        mean_style = {"linestyle": "--", "color": "k", "linewidth": 1.0, "alpha": 0.7}

    # Plot chains for each selected event
    for idx, ev_idx in enumerate(selection):
        r = idx // n_cols
        c = idx % n_cols
        axis = ax[r, c]

        # x-axis: sample indices after burn-in & thinning
        sample_idx = _np.arange(burn_in, n_samples, thin)
        for coord in coords:
            coord_s = str(coord)
            dict_key = "delta_t" if (not is_summary and coord_s == "T") else coord_s
            f = _coord_factor(coord_s, units=units)
            unit = _coord_unit_label(coord_s, units=units)
            if is_summary:
                series = getattr(samples_data, coord_s)[ev_idx, :]
            else:
                series = samples_data[dict_key][ev_idx, burn_in:]
            if thin > 1:
                series = series[::thin]
            y_vals = (series * float(f))
            y_vals -= y_vals.mean()
            lab = f"{coord_s} ({unit})" if unit else coord_s
            axis.plot(sample_idx[: y_vals.shape[0]], y_vals, label=lab,
                      alpha=alpha, linewidth=linewidth, color=coordinates_colors.get(coord_s, None))
            if show_mean and y_vals.size > 0:
                axis.axhline(y_vals.mean(), **mean_style)

        # Title with event id
        try:
            evid = event_ids_resolved[ev_idx]
        except Exception:
            evid = str(ev_idx)
        axis.set_title(str(evid), fontsize=9)

    # Labels and limits
    for c in range(n_cols):
        ax[-1, c].set_xlabel("Sample index")
    coord_units = [_coord_unit_label(str(c), units=units) for c in coords]
    uniq_units = sorted({u for u in coord_units if u})
    if len(uniq_units) == 1:
        ax[0, 0].set_ylabel(f"Value ({uniq_units[0]})")
    else:
        ax[0, 0].set_ylabel("Value")

    if xlim is not None:
        for axes_row in ax:
            for axis in axes_row:
                axis.set_xlim(xlim)
    if ylim is not None:
        for axes_row in ax:
            for axis in axes_row:
                axis.set_ylim(ylim)

    # Legend
    ax[0, 0].legend(loc='upper right', fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, ax


def plot_uncertainty_histograms(
    df,
    columns,
    labels=None,
    colors=None,
    scale=1.0,
    xlim=(0, 600),
    xlabel="Hypocenter uncertainty per event (meters)",
    ylabel="Count",
    cumulative=True,
    density=True,
    bins=None,
    output_path=None,
    legend_loc="lower right"
):
    """
    Plot cumulative histograms for selected dataframe columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data.
    columns : list of str
        Column names to plot.
    labels : list of str, optional
        Labels for legend; defaults to column names.
    colors : list of str, optional
        Colors for each histogram; defaults to matplotlib's cycle.
    scale : float, optional
        Multiply values by this scale before plotting.
    xlim : tuple, optional
        X-axis limits.
    xlabel, ylabel : str, optional
        Axis labels.
    cumulative : bool, optional
        Whether to plot cumulative histograms.
    density : bool, optional
        Whether to normalize histograms.
    bins : int or sequence, optional
        Bins for histogram; defaults to range(len(data)).
    output_path : str, optional
        Path to save figure; if None, figure is not saved.
    legend_loc : str, optional
        Location for the legend.
    """
    plt = _lazy_import_plt()
    if labels is None:
        labels = columns
    if colors is None:
        colors = [None] * len(columns)

    plt.figure()
    for col, label, color in zip(columns, labels, colors):
        data = df[col] * scale
        if bins is None:
            bins_used = np.arange(len(data))
        else:
            bins_used = bins
        plt.hist(
            data,
            bins=bins_used,
            histtype='step',
            label=label,
            color=color,
            cumulative=cumulative,
            density=density
        )

    plt.xlim(xlim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=legend_loc)

    if output_path:
        plt.savefig(output_path)
    plt.show()


def plot_event_marginal_hist2d(samples_data: Union[dict, Any],
                               event_index=None,
                               event_id=None,
                               coords=("X", "Y", "Z"),
                               burn_in=0,
                               units="km",
                               xlim=(-0.1, 0.1),
                               bins=None,
                               bin_width=0.003,
                               figsize=(12, 6),
                               constrained_layout=True,
                               height_ratios=(1, 3),
                               sharex='col',
                               sharey='row',
                               cmap='Blues',
                               vmin=None,
                               vmax=None,
                               density=True,
                               top_row_ylim=None,
                               n_contours=8,
                               contour_log=False,
                               equal_aspect=True,
                               title=None,
                               center_data=False):
    """Plot 1D marginals and 2D histograms with contours for a single event.

    Top row: 1D histograms for coords[0], coords[1], coords[2]
    Bottom row: 2D histograms with contours for (coords[0], coords[1]), (coords[0], coords[2]), (coords[1], coords[2])

    Args:
        samples_data: dict from `spider.io.samples.read_all_samples` with fields including coords and 'event_ids'.
        event_index: integer index of the event to plot.
        event_id: optional event id string to locate index (overrides event_index if provided and found).
        coords: tuple of three coordinate names present in samples_data (default ("X","Y","Z")).
        burn_in: number of initial samples to discard.
        units: 'km' (no scaling) or 'meters' (scales values by 1000).
        xlim: tuple (min, max) range for histogram axes in the chosen units.
        bins: optional numpy array of bin edges; if None, uses bin_width across xlim.
        bin_width: used when bins is None to create np.arange(xlim[0], xlim[1], bin_width).
        figsize, constrained_layout, height_ratios, sharex, sharey: matplotlib layout options.
        cmap: colormap for 2D hist imshow.
        vmin, vmax: color scale limits for 2D hist imshow (shared across panels).
        density: if True, 1D hist uses density=True.
        top_row_ylim: optional y-limit for top-row histograms.
        n_contours: number of contour levels (excluding min).
        contour_log: if True, use log-spaced contour levels (positive bins only).
        equal_aspect: if True, set equal aspect for bottom row plots.
        title: optional suptitle for the figure.
        center_data: if True, subtract the mean from each coordinate before plotting.

    Returns:
        (fig, axes): matplotlib figure and axes array of shape (2, 3).
    """
    # Validate coords
    if len(coords) != 3:
        raise ValueError("coords must be a tuple/list of exactly three field names")
    is_summary = _is_event_samples_summary(samples_data)
    for c in coords:
        if is_summary:
            if c not in ("X", "Y", "Z"):
                raise KeyError(f"When using EventSamplesSummary, only 'X','Y','Z' are supported (got '{c}')")
        else:
            if c not in samples_data:
                raise KeyError(f"Coordinate '{c}' not found in samples_data")

    # Resolve event index
    event_ids_resolved = None
    if is_summary:
        if getattr(samples_data, 'cat_dd', None) is not None and 'evid' in samples_data.cat_dd.columns:
            event_ids_resolved = _np.asarray(samples_data.cat_dd['evid'].values)
    else:
        if 'event_ids' in samples_data:
            event_ids_resolved = _np.asarray(samples_data['event_ids'])

    if event_id is not None and event_ids_resolved is not None:
        eid_arr = event_ids_resolved
        matches = _np.nonzero(eid_arr == event_id)[0]
        if matches.size > 0:
            event_index = int(matches[0])
    if event_index is None:
        event_index = 0

    # Extract series and apply burn-in and units
    factor = 1000.0 if units.lower().startswith('meter') else 1.0
    series = []
    for c in coords:
        if is_summary:
            arr = getattr(samples_data, c)[event_index, :]
        else:
            arr = samples_data[c][event_index, burn_in:]
        arr = _np.asarray(arr) * factor
        # Flatten any 2D/ND arrays to 1D vector of samples
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        if center_data and arr.size > 0:
            arr = arr - _np.mean(arr)
        series.append(arr)
    Xv, Yv, Zv = series  # using names to mirror example

    # Build bins
    if bins is None:
        if bin_width is None:
            bin_width = (xlim[1] - xlim[0]) / 100.0
        bins = _np.arange(xlim[0], xlim[1] + 0.5 * bin_width, bin_width)

    # Setup figure
    _plt = _lazy_import_plt()
    fig, axes = _plt.subplots(
        nrows=2, ncols=3, figsize=figsize,
        constrained_layout=constrained_layout,
        gridspec_kw={'height_ratios': list(height_ratios)},
        sharex=sharex, sharey=sharey
    )

    # Top row: 1D histograms
    colors = {coords[0]: 'red', coords[1]: 'blue', coords[2]: 'black'}
    for j, (ax, data, label) in enumerate(zip(axes[0, :], (Xv, Yv, Zv), coords)):
        data = _np.asarray(data)
        data = data[_np.isfinite(data)]
        use_density = bool(density) and data.size > 0
        if data.size > 0:
            ax.hist(data, bins=bins, color=colors[label], density=use_density)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(f"{label} ({'km' if factor==1.0 else 'm'})")
        ax.set_ylabel("Density" if use_density else "Count")
        ax.set_xlim(xlim)
        if top_row_ylim is not None:
            ax.set_ylim(top_row_ylim)

    # Remove y-ticks/labels on all but first histogram
    for ax in axes[0, 1:]:
        ax.tick_params(left=False, labelleft=False)

    # 2D histogram with contours helper
    def _plot_2d_with_contour(ax, x, y):
        x = _np.asarray(x)
        y = _np.asarray(y)
        mask = _np.isfinite(x) & _np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size == 0 or y.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            return
        hist, xedges, yedges = _np.histogram2d(x, y, bins=[bins, bins])
        xcenters = 0.5 * (xedges[1:] + xedges[:-1])
        ycenters = 0.5 * (yedges[1:] + yedges[:-1])
        Xgrid, Ygrid = _np.meshgrid(xcenters, ycenters)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        ax.imshow(hist.T, origin='lower', extent=extent, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        # Contours (skip zero-only)
        hmin, hmax = float(hist.min()), float(hist.max())
        if hmax > 0.0 and n_contours and n_contours > 0:
            if contour_log:
                pos = hist[hist > 0.0]
                if pos.size > 0:
                    vmin_pos = float(pos.min())
                    levels = _np.logspace(_np.log10(vmin_pos), _np.log10(hmax), int(n_contours))
                else:
                    levels = _np.linspace(hmin, hmax, int(n_contours) + 1)[1:]
            else:
                levels = _np.linspace(hmin, hmax, int(n_contours) + 1)[1:]
            ax.contour(Xgrid, Ygrid, hist.T, levels=levels, colors='black', linewidths=1)
        return

    # Bottom row 2D plots: (X,Y), (X,Z), (Y,Z)
    _plot_2d_with_contour(axes[1, 0], Xv, Yv)
    _plot_2d_with_contour(axes[1, 1], Xv, Zv)
    _plot_2d_with_contour(axes[1, 2], Yv, Zv)

    # Label axes
    axes[1, 0].set_xlabel(f"{coords[0]} ({'km' if factor==1.0 else 'm'})")
    axes[1, 0].set_ylabel(f"{coords[1]} ({'km' if factor==1.0 else 'm'})")
    axes[1, 1].set_xlabel(f"{coords[0]} ({'km' if factor==1.0 else 'm'})")
    axes[1, 1].set_ylabel(f"{coords[2]} ({'km' if factor==1.0 else 'm'})")
    axes[1, 2].set_xlabel(f"{coords[1]} ({'km' if factor==1.0 else 'm'})")
    axes[1, 2].set_ylabel(f"{coords[2]} ({'km' if factor==1.0 else 'm'})")

    # Ticks styling
    for ax_row in axes:
        for ax in ax_row:
            ax.tick_params(axis='both', which='both', direction='out', bottom=True, top=False,
                           left=True, right=False, labelbottom=True, labelleft=True)

    # Equal aspect for bottom row if requested
    if equal_aspect:
        for ax in axes[1]:
            ax.set_aspect('equal', adjustable='box')

    # Optional title
    if title:
        fig.suptitle(title)

    if not constrained_layout:
        fig.tight_layout()
    return fig, axes


def plot_event_marginal_kde2d(samples_data: Union[dict, Any],
                              event_index=None,
                              event_id=None,
                              coords=("X", "Y", "Z"),
                              burn_in=0,
                              units="km",
                              xlim=(-0.1, 0.1),
                              grid_size=100,
                              figsize=(12, 6),
                              constrained_layout=True,
                              height_ratios=(1, 3),
                              sharex='col',
                              sharey='row',
                              cmap='Blues',
                              vmin=None,
                              vmax=None,
                              density=True,
                              top_row_ylim=None,
                              n_contours=8,
                              contour_log=False,
                              equal_aspect=True,
                              title=None,
                              center_data=False,
                              bw_method=None,
                              inter_event_k=None,
                              inter_event_row=False,
                              inter_event_max_abs=None,
                              inter_event_mode="centered_pairs",
                              inter_event_scale_figsize=True,
                              subplot_wspace=None,
                              subplot_hspace=None,
                              top_row_hide_yaxis=False):
    """Plot 1D KDE marginals and 2D KDE contours for a single event.

    Args mirror plot_event_marginal_hist2d, but use KDEs instead of binning.
    bw_method: passed to scipy.stats.gaussian_kde (None for default).
    inter_event_row: if True, add a third row showing k-nearest inter-event structure.
    inter_event_k: number of nearest neighbors to use (None -> all other events).
    inter_event_max_abs: optional symmetric limit for inter-event axes (km or m).
    inter_event_mode:
        "centered_pairs" -> KDE of centered (x_i vs x_j), (y_i vs y_j), (z_i vs z_j)
    inter_event_scale_figsize: if True, scale figure height to keep subplot size similar.
    subplot_wspace/subplot_hspace: optional spacing overrides (passed to subplots_adjust).
    top_row_hide_yaxis: if True, hide y-axis ticks/labels for top row except first.
    """
    plt = _lazy_import_plt()
    try:
        from scipy.stats import gaussian_kde  # type: ignore
    except Exception as e:
        raise ImportError("plot_event_marginal_kde2d requires scipy") from e

    # Validate coords
    if len(coords) != 3:
        raise ValueError("coords must be a tuple/list of exactly three field names")
    is_summary = _is_event_samples_summary(samples_data)
    for c in coords:
        if is_summary:
            if c not in ("X", "Y", "Z"):
                raise KeyError(f"When using EventSamplesSummary, only 'X','Y','Z' are supported (got '{c}')")
        else:
            if c not in samples_data:
                raise KeyError(f"Coordinate '{c}' not found in samples_data")

    # Resolve event index
    event_ids_resolved = None
    if is_summary:
        if getattr(samples_data, 'cat_dd', None) is not None and 'evid' in samples_data.cat_dd.columns:
            event_ids_resolved = _np.asarray(samples_data.cat_dd['evid'].values)
    else:
        if 'event_ids' in samples_data:
            event_ids_resolved = _np.asarray(samples_data['event_ids'])
    if event_id is not None and event_ids_resolved is not None:
        eid_arr = event_ids_resolved
        matches = _np.nonzero(eid_arr == event_id)[0]
        if matches.size > 0:
            event_index = int(matches[0])
    if event_index is None:
        event_index = 0

    # Extract series and apply burn-in and units
    factor = 1000.0 if units.lower().startswith('meter') else 1.0
    series = []
    for c in coords:
        if is_summary:
            arr = getattr(samples_data, c)[event_index, :]
        else:
            arr = samples_data[c][event_index, burn_in:]
        arr = _np.asarray(arr) * factor
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        if center_data and arr.size > 0:
            arr = arr - _np.mean(arr)
        series.append(arr)
    Xv, Yv, Zv = series

    n_rows = 3 if inter_event_row else 2
    if inter_event_row and len(height_ratios) == 2:
        height_ratios = (height_ratios[0], height_ratios[1], height_ratios[1])
    if inter_event_row and inter_event_scale_figsize and len(height_ratios) == 3:
        base_sum = float(height_ratios[0] + height_ratios[1])
        new_sum = float(sum(height_ratios))
        if base_sum > 0.0 and new_sum > base_sum:
            scale = new_sum / base_sum
            figsize = (figsize[0], figsize[1] * scale)
    # Inter-event row uses different axis scales; disable shared axes.
    sharex_used = sharex
    sharey_used = sharey
    if inter_event_row:
        sharex_used = False
        sharey_used = False
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=figsize,
        constrained_layout=constrained_layout,
        gridspec_kw={'height_ratios': height_ratios},
        sharex=sharex_used,
        sharey=sharey_used
    )
    if (subplot_wspace is not None) or (subplot_hspace is not None):
        if constrained_layout:
            fig.set_constrained_layout(False)
        fig.subplots_adjust(
            wspace=subplot_wspace if subplot_wspace is not None else 0.2,
            hspace=subplot_hspace if subplot_hspace is not None else 0.2,
        )

    # 1D KDEs
    xs = _np.linspace(xlim[0], xlim[1], int(grid_size))
    for ax, data, label in zip(axes[0], (Xv, Yv, Zv), coords):
        data = _np.asarray(data)
        if data.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        kde = gaussian_kde(data, bw_method=bw_method)
        ys = kde(xs)
        ax.plot(xs, ys, color='black')
        ax.set_title(label)
        ax.set_xlim(xlim)
        ax.set_ylabel("Density" if density else "KDE")
        if top_row_ylim is not None:
            ax.set_ylim(top_row_ylim)

    if top_row_hide_yaxis:
        for ax in axes[0, 1:]:
            ax.tick_params(left=False, labelleft=False)

    def _plot_2d_kde(ax, x, y, *, xlim_local=None, ylim_local=None):
        x = _np.asarray(x)
        y = _np.asarray(y)
        mask = _np.isfinite(x) & _np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size == 0 or y.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            return
        if xlim_local is None:
            xlim_local = xlim
        if ylim_local is None:
            ylim_local = xlim
        xgrid = _np.linspace(xlim_local[0], xlim_local[1], int(grid_size))
        ygrid = _np.linspace(ylim_local[0], ylim_local[1], int(grid_size))
        Xg, Yg = _np.meshgrid(xgrid, ygrid)
        kde = gaussian_kde(_np.vstack([x, y]), bw_method=bw_method)
        Zg = kde(_np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
        ax.imshow(Zg, origin='lower', extent=[xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]], cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        if n_contours and n_contours > 0:
            zmin = float(Zg.min())
            zmax = float(Zg.max())
            if contour_log:
                pos = Zg[Zg > 0.0]
                if pos.size > 0:
                    zmin = float(pos.min())
                    levels = _np.logspace(_np.log10(zmin), _np.log10(zmax), int(n_contours))
                else:
                    levels = _np.linspace(zmin, zmax, int(n_contours) + 1)[1:]
            else:
                levels = _np.linspace(zmin, zmax, int(n_contours) + 1)[1:]
            ax.contour(Xg, Yg, Zg, levels=levels, colors='black', linewidths=1)

    _plot_2d_kde(axes[1, 0], Xv, Yv, xlim_local=xlim, ylim_local=xlim)
    _plot_2d_kde(axes[1, 1], Xv, Zv, xlim_local=xlim, ylim_local=xlim)
    _plot_2d_kde(axes[1, 2], Yv, Zv, xlim_local=xlim, ylim_local=xlim)

    # Optional inter-event row (k-nearest structure)
    if inter_event_row:
        if is_summary:
            X_all = _np.asarray(samples_data.X) * factor
            Y_all = _np.asarray(samples_data.Y) * factor
            Z_all = _np.asarray(samples_data.Z) * factor
            Xm = _np.mean(X_all, axis=1)
            Ym = _np.mean(Y_all, axis=1)
            Zm = _np.mean(Z_all, axis=1)
        else:
            X_all = _np.asarray(samples_data[coords[0]]) * factor
            Y_all = _np.asarray(samples_data[coords[1]]) * factor
            Z_all = _np.asarray(samples_data[coords[2]]) * factor
            Xm = _np.mean(X_all[:, burn_in:], axis=1)
            Ym = _np.mean(Y_all[:, burn_in:], axis=1)
            Zm = _np.mean(Z_all[:, burn_in:], axis=1)
        if center_data:
            Xm = Xm - _np.mean(Xm)
            Ym = Ym - _np.mean(Ym)
            Zm = Zm - _np.mean(Zm)

        dx = Xm - Xm[event_index]
        dy = Ym - Ym[event_index]
        dz = Zm - Zm[event_index]
        d = _np.sqrt(dx * dx + dy * dy + dz * dz)
        mask = _np.isfinite(d)
        mask[event_index] = False
        dx = dx[mask]
        dy = dy[mask]
        dz = dz[mask]
        d = d[mask]

        if d.size > 0:
            order = _np.argsort(d)
            if inter_event_k is not None:
                k = max(1, int(inter_event_k))
                order = order[:min(k, order.size)]
            dx = dx[order]
            dy = dy[order]
            dz = dz[order]
            d = d[order]
            neighbor_idx = _np.nonzero(mask)[0][order]
        else:
            neighbor_idx = _np.array([], dtype=int)

        def _auto_sym_lim(vals, fallback):
            vals = _np.asarray(vals)
            vals = vals[_np.isfinite(vals)]
            if vals.size == 0:
                return fallback
            vmax = float(_np.nanpercentile(_np.abs(vals), 99.5))
            if not _np.isfinite(vmax) or vmax <= 0.0:
                return fallback
            return (-vmax, vmax)

        if inter_event_max_abs is not None:
            lim_dx = (-float(inter_event_max_abs), float(inter_event_max_abs))
            lim_dy = lim_dx
            lim_dz = lim_dx
        else:
            # Default: match the main plot limits for comparability.
            lim_dx = xlim
            lim_dy = xlim
            lim_dz = xlim

        if inter_event_mode == "centered_pairs":
            # Build centered sample pairs for event i vs neighbors j
            if is_summary:
                xi = X_all[event_index, :]
                yi = Y_all[event_index, :]
                zi = Z_all[event_index, :]
            else:
                xi = X_all[event_index, burn_in:]
                yi = Y_all[event_index, burn_in:]
                zi = Z_all[event_index, burn_in:]
            xi = _np.asarray(xi).reshape(-1)
            yi = _np.asarray(yi).reshape(-1)
            zi = _np.asarray(zi).reshape(-1)
            xi = xi - _np.mean(xi) if xi.size > 0 else xi
            yi = yi - _np.mean(yi) if yi.size > 0 else yi
            zi = zi - _np.mean(zi) if zi.size > 0 else zi

            xj_all = []
            yj_all = []
            zj_all = []
            for j in neighbor_idx.tolist():
                if is_summary:
                    xj = X_all[j, :]
                    yj = Y_all[j, :]
                    zj = Z_all[j, :]
                else:
                    xj = X_all[j, burn_in:]
                    yj = Y_all[j, burn_in:]
                    zj = Z_all[j, burn_in:]
                xj = _np.asarray(xj).reshape(-1)
                yj = _np.asarray(yj).reshape(-1)
                zj = _np.asarray(zj).reshape(-1)
                if xj.size == 0 or yj.size == 0 or zj.size == 0:
                    continue
                xj = xj - _np.mean(xj)
                yj = yj - _np.mean(yj)
                zj = zj - _np.mean(zj)
                xj_all.append(xj)
                yj_all.append(yj)
                zj_all.append(zj)

            if xj_all:
                xj_all = _np.concatenate(xj_all, axis=0)
                yj_all = _np.concatenate(yj_all, axis=0)
                zj_all = _np.concatenate(zj_all, axis=0)
            else:
                xj_all = _np.array([], dtype=float)
                yj_all = _np.array([], dtype=float)
                zj_all = _np.array([], dtype=float)

            def _match_lengths(a, b):
                a = _np.asarray(a)
                b = _np.asarray(b)
                a = a[_np.isfinite(a)]
                b = b[_np.isfinite(b)]
                if a.size == 0 or b.size == 0:
                    return _np.array([], dtype=float), _np.array([], dtype=float)
                if a.size == b.size:
                    return a, b
                n = int(min(a.size, b.size))
                rng = _np.random.default_rng(0)
                if a.size > n:
                    a = a[rng.choice(a.size, size=n, replace=False)]
                if b.size > n:
                    b = b[rng.choice(b.size, size=n, replace=False)]
                return a, b

            # Pairwise KDEs: x_i vs x_j, y_i vs y_j, z_i vs z_j
            xi_plot, xj_plot = _match_lengths(xi, xj_all)
            yi_plot, yj_plot = _match_lengths(yi, yj_all)
            zi_plot, zj_plot = _match_lengths(zi, zj_all)
            _plot_2d_kde(axes[2, 0], xi_plot, xj_plot, xlim_local=lim_dx, ylim_local=lim_dx)
            _plot_2d_kde(axes[2, 1], yi_plot, yj_plot, xlim_local=lim_dy, ylim_local=lim_dy)
            _plot_2d_kde(axes[2, 2], zi_plot, zj_plot, xlim_local=lim_dz, ylim_local=lim_dz)
            axes[2, 0].set_xlabel(f"{coords[0]}_i (centered)")
            axes[2, 0].set_ylabel(f"{coords[0]}_j (centered)")
            axes[2, 1].set_xlabel(f"{coords[1]}_i (centered)")
            axes[2, 1].set_ylabel(f"{coords[1]}_j (centered)")
            axes[2, 2].set_xlabel(f"{coords[2]}_i (centered)")
            axes[2, 2].set_ylabel(f"{coords[2]}_j (centered)")
        else:
            raise ValueError(f"Unknown inter_event_mode='{inter_event_mode}'")

    axes[1, 0].set_xlabel(f"{coords[0]} ({'km' if factor==1.0 else 'm'})")
    axes[1, 0].set_ylabel(f"{coords[1]} ({'km' if factor==1.0 else 'm'})")
    axes[1, 1].set_xlabel(f"{coords[0]} ({'km' if factor==1.0 else 'm'})")
    axes[1, 1].set_ylabel(f"{coords[2]} ({'km' if factor==1.0 else 'm'})")
    axes[1, 2].set_xlabel(f"{coords[1]} ({'km' if factor==1.0 else 'm'})")
    axes[1, 2].set_ylabel(f"{coords[2]} ({'km' if factor==1.0 else 'm'})")

    if equal_aspect:
        for ax in axes[1, :]:
            ax.set_aspect('equal', adjustable='box')
        if inter_event_row:
            for ax in axes[2, :]:
                ax.set_aspect('equal', adjustable='box')

    if title:
        fig.suptitle(title)
    return fig, axes
