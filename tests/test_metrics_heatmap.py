import sys
from pathlib import Path

import numpy as np
import matplotlib

# Use a non-interactive backend for tests
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local package is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from tank_model.metrics import plot_error_metrics_heatmap, nse


def test_plot_error_metrics_heatmap_annotations():
    obs = np.array([1.0, 2.0, 3.0])
    sim = np.array([1.0, 2.0, 3.0])

    ax = plot_error_metrics_heatmap(obs, sim)

    assert ax is not None
    nse_val = nse(obs, sim)
    texts = [t.get_text() for t in ax.texts]
    assert any(f"{nse_val:.3f}" == txt for txt in texts)

    plt.close(ax.figure)
