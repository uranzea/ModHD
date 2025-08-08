
import numpy as np
import matplotlib.pyplot as plt

def nse(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    m = np.nanmean(obs)
    num = np.nansum((obs - sim)**2)
    den = np.nansum((obs - m)**2)
    return 1 - num/den if den > 0 else np.nan

def kge(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    r = np.corrcoef(obs, sim)[0,1]
    alpha = np.nanstd(sim)/np.nanstd(obs)
    beta = np.nanmean(sim)/np.nanmean(obs)
    return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

def bias_pct(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    return 100.0 * (np.nansum(sim) - np.nansum(obs)) / np.nansum(obs)


def plot_error_metrics_heatmap(obs, sim, ax=None, cmap="viridis"):
    """Plot a heatmap with common error metrics.

    The function computes NSE, KGE and percentage bias for the provided
    observed and simulated series and displays them in a single-row
    heatmap.  Each cell is annotated with the metric value so the user can
    quickly read the numbers.

    Parameters
    ----------
    obs, sim : array-like
        Observed and simulated series.  They are converted to ``numpy``
        arrays and may contain ``NaN`` values.
    ax : matplotlib.axes.Axes, optional
        Axes to draw the heatmap on.  If ``None`` a new figure and axes are
        created.
    cmap : str, optional
        Name of the matplotlib colormap used for the heatmap.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the heatmap so that callers can further
        customise or save the figure.
    """

    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    metric_funcs = {
        "NSE": nse,
        "KGE": kge,
        "Bias%": bias_pct,
    }
    labels = list(metric_funcs)
    values = [metric_funcs[name](obs, sim) for name in labels]
    data = np.array(values, dtype=float).reshape(1, -1)

    if ax is None:
        fig_width = 1.2 * data.shape[1]
        _, ax = plt.subplots(figsize=(fig_width, 2))

    im = ax.imshow(data, aspect="auto", cmap=cmap)
    mid = (np.nanmax(data) + np.nanmin(data)) / 2

    for j, val in enumerate(data[0]):
        color = "white" if val < mid else "black"
        ax.text(j, 0, f"{val:.3f}", ha="center", va="center", color=color)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks([])
    ax.set_title("Error metrics")
    plt.colorbar(im, ax=ax, label="Value")
    return ax
