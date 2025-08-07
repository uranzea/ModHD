
import numpy as np

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
