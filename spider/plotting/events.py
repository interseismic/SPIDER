import numpy as _np
import numpy as np
import matplotlib.pyplot as _plt


def plot_event_distributions(samples_data,
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
        samples_data: dict from read_all_samples_parallel with keys 'event_ids', 'X','Y','Z', ...
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

    # Validate inputs
    assert 'event_ids' in samples_data, "samples_data must include 'event_ids'"
    for c in coords:
        if c not in samples_data:
            raise KeyError(f"Coordinate '{c}' not found in samples_data")

    n_events, n_samples = samples_data[coords[0]].shape
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

    # Units factor
    factor = 1000.0 if units.lower().startswith('meter') else 1.0
    default_xlim = (-300, 300) if factor == 1000.0 else (-0.3, 0.3)
    if xlim is None:
        xlim = default_xlim

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
                           "delta_t": "tab:olive"}

    for idx, ev_idx in enumerate(event_indices):
        r = idx // n_cols
        c = idx % n_cols
        axis = ax[r, c]

        for coord in coords:
            series = samples_data[coord][ev_idx, burn_in:]
            x_vals = (series * factor)
            x_kde, y_kde = _kde_curve(x_vals)
            axis.plot(x_kde, y_kde, label=coord, color=coordinates_colors.get(coord, None))

        # Title with event id if available
        try:
            evid = samples_data['event_ids'][ev_idx]
        except Exception:
            evid = str(ev_idx)
        axis.set_title(str(evid), fontsize=9)

    # Label the bottom row
    for c in range(n_cols):
        ax[-1, c].set_xlabel(f"Relative Uncertainty ({'meters' if factor==1000.0 else 'km'})")
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


def build_catalog_posterior_mean(event_samples):
    """
    Build a catalog of posterior means for each event.
    """
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


def plot_event_chains(samples_data,
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
        samples_data: dict from read_all_samples_parallel with keys 'event_ids', 'X','Y','Z', ...
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
    assert 'event_ids' in samples_data, "samples_data must include 'event_ids'"
    for c in coords:
        if c not in samples_data:
            raise KeyError(f"Coordinate '{c}' not found in samples_data")

    n_events, n_samples = samples_data[coords[0]].shape
    num_panels = min(n_rows * n_cols, n_events)

    # Resolve event selection
    selection = None
    if event_indices is not None:
        selection = _np.asarray(event_indices, dtype=int)
    elif event_ids is not None:
        # Map event IDs to indices
        eid_arr = _np.asarray(samples_data['event_ids'])
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

    # Units factor
    factor = 1000.0 if units.lower().startswith('meter') else 1.0

    # Prepare figure
    fig, ax = _plt.subplots(nrows=n_rows, ncols=n_cols, sharex=sharex, sharey=sharey, figsize=figsize)
    ax = _np.atleast_2d(ax)

    # Colors per coordinate
    coordinates_colors = {"X": "tab:blue", "Y": "tab:orange", "Z": "tab:green",
                           "longitude": "tab:purple", "latitude": "tab:red", "depth": "tab:brown",
                           "delta_t": "tab:olive"}

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
            series = samples_data[coord][ev_idx, burn_in:]
            if thin > 1:
                series = series[::thin]
            y_vals = (series * factor)
            axis.plot(sample_idx[: y_vals.shape[0]], y_vals, label=coord,
                      alpha=alpha, linewidth=linewidth, color=coordinates_colors.get(coord, None))
            if show_mean and y_vals.size > 0:
                axis.axhline(y_vals.mean(), **mean_style)

        # Title with event id
        try:
            evid = samples_data['event_ids'][ev_idx]
        except Exception:
            evid = str(ev_idx)
        axis.set_title(str(evid), fontsize=9)

    # Labels and limits
    for c in range(n_cols):
        ax[-1, c].set_xlabel("Sample index")
    if units.lower().startswith('meter'):
        ax[0, 0].set_ylabel("Value (m)")
    else:
        ax[0, 0].set_ylabel("Value (km)")

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
    import matplotlib.pyplot as plt
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


def plot_event_marginal_hist2d(samples_data,
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
                               density=True,
                               top_row_ylim=None,
                               n_contours=8,
                               equal_aspect=True,
                               title=None):
    """Plot 1D marginals and 2D histograms with contours for a single event.

    Top row: 1D histograms for coords[0], coords[1], coords[2]
    Bottom row: 2D histograms with contours for (coords[0], coords[1]), (coords[0], coords[2]), (coords[1], coords[2])

    Args:
        samples_data: dict from read_all_samples_parallel with fields including coords and 'event_ids'.
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
        density: if True, 1D hist uses density=True.
        top_row_ylim: optional y-limit for top-row histograms.
        n_contours: number of contour levels (excluding min).
        equal_aspect: if True, set equal aspect for bottom row plots.
        title: optional suptitle for the figure.

    Returns:
        (fig, axes): matplotlib figure and axes array of shape (2, 3).
    """
    # Validate coords
    if len(coords) != 3:
        raise ValueError("coords must be a tuple/list of exactly three field names")
    for c in coords:
        if c not in samples_data:
            raise KeyError(f"Coordinate '{c}' not found in samples_data")

    # Resolve event index
    if event_id is not None and 'event_ids' in samples_data:
        eid_arr = _np.asarray(samples_data['event_ids'])
        matches = _np.nonzero(eid_arr == event_id)[0]
        if matches.size > 0:
            event_index = int(matches[0])
    if event_index is None:
        event_index = 0

    # Extract series and apply burn-in and units
    factor = 1000.0 if units.lower().startswith('meter') else 1.0
    series = []
    for c in coords:
        arr = samples_data[c][event_index, burn_in:]
        arr = arr * factor
        series.append(_np.asarray(arr))
    Xv, Yv, Zv = series  # using names to mirror example

    # Build bins
    if bins is None:
        if bin_width is None:
            bin_width = (xlim[1] - xlim[0]) / 100.0
        bins = _np.arange(xlim[0], xlim[1] + 0.5 * bin_width, bin_width)

    # Setup figure
    fig, axes = _plt.subplots(
        nrows=2, ncols=3, figsize=figsize,
        constrained_layout=constrained_layout,
        gridspec_kw={'height_ratios': list(height_ratios)},
        sharex=sharex, sharey=sharey
    )

    # Top row: 1D histograms
    colors = {coords[0]: 'red', coords[1]: 'blue', coords[2]: 'black'}
    for j, (ax, data, label) in enumerate(zip(axes[0, :], (Xv, Yv, Zv), coords)):
        ax.hist(data, bins=bins, color=colors[label], density=density)
        ax.set_xlabel(f"{label} ({'km' if factor==1.0 else 'm'})")
        ax.set_ylabel("Density" if density else "Count")
        ax.set_xlim(xlim)
        if top_row_ylim is not None:
            ax.set_ylim(top_row_ylim)

    # Remove y-ticks/labels on all but first histogram
    for ax in axes[0, 1:]:
        ax.tick_params(left=False, labelleft=False)

    # 2D histogram with contours helper
    def _plot_2d_with_contour(ax, x, y):
        hist, xedges, yedges = _np.histogram2d(x, y, bins=[bins, bins])
        xcenters = 0.5 * (xedges[1:] + xedges[:-1])
        ycenters = 0.5 * (yedges[1:] + yedges[:-1])
        Xgrid, Ygrid = _np.meshgrid(xcenters, ycenters)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        ax.imshow(hist.T, origin='lower', extent=extent, cmap=cmap, aspect='auto')
        # Contours (skip zero-only)
        hmin, hmax = float(hist.min()), float(hist.max())
        if hmax > 0.0 and n_contours and n_contours > 0:
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

    fig.tight_layout()
    return fig, axes
