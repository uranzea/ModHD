
"""Utilidades de entrada/salida y pre-proceso."""
import pandas as pd

def load_csv(path, date_col="date", tz=None):
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.set_index(date_col).sort_index()
    if tz:
        df.index = df.index.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT").tz_convert(tz)
    return df

def write_csv(df, path):
    df.to_csv(path)

def subset_period(df, start=None, end=None):
    if start is not None:
        df = df[df.index >= pd.to_datetime(start)]
    if end is not None:
        df = df[df.index <= pd.to_datetime(end)]
    return df

def resample_mean(df, freq="D"):
    return df.resample(freq).mean()

def tag_hydrology(df, col="P_mm", dry_q=0.25, wet_q=0.75, min_days=5):
    """Clasifica períodos secos/húmedos según cuantiles móviles de la variable col.
    Devuelve una serie 'regime' con valores {'dry','normal','wet'}.
    """
    roll = df[col].rolling(window=min_days, min_periods=1).mean()
    q1 = roll.quantile(dry_q)
    q3 = roll.quantile(wet_q)
    regime = roll.copy()*0.0
    regime[roll <= q1] = -1
    regime[(roll > q1) & (roll < q3)] = 0
    regime[roll >= q3] = 1
    mapping = {-1:"dry", 0:"normal", 1:"wet"}
    return regime.map(mapping)
