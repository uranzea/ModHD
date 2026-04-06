from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import sys

sys.path.append(str(Path(__file__).resolve().parent))

import geopandas as gpd
import pandas as pd

from common import INPUT_DIR, OUTPUT_DIR, ROOT

CATALOG_DIR = ROOT.parent / "catalogo_capas"


@dataclass
class LayerEntry:
    name: str
    path: Path
    source_group: str


def list_input_csvs() -> list[Path]:
    return sorted(INPUT_DIR.glob("*.csv"))


def list_output_images() -> list[Path]:
    return sorted(OUTPUT_DIR.glob("*.png"))


def list_output_tables() -> list[Path]:
    return sorted(OUTPUT_DIR.glob("*.csv"))


def list_catalog_layers() -> list[LayerEntry]:
    entries: list[LayerEntry] = []
    if not CATALOG_DIR.exists():
        return entries
    for path in sorted(CATALOG_DIR.rglob("*")):
        if path.suffix.lower() not in {".shp", ".gpkg"}:
            continue
        try:
            group = path.relative_to(CATALOG_DIR).parts[0]
        except Exception:
            group = "catalogo"
        entries.append(LayerEntry(name=path.stem, path=path, source_group=group))
    return entries


def load_series_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()


def detect_point_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    aliases = [
        ("lon", "lat"),
        ("longitude", "latitude"),
        ("x", "y"),
        ("X", "Y"),
        ("Lon", "Lat"),
    ]
    cols = set(df.columns)
    for lon_col, lat_col in aliases:
        if lon_col in cols and lat_col in cols:
            return lon_col, lat_col
    return None, None


def load_layer(path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


def layer_summary(gdf: gpd.GeoDataFrame) -> dict[str, str]:
    if gdf.empty:
        return {
            "features": "0",
            "geometry": "n/a",
            "crs": str(gdf.crs),
            "bounds": "n/a",
        }
    bounds = ", ".join(f"{v:.4f}" for v in gdf.total_bounds)
    geometry_types = ",".join(sorted({str(v) for v in gdf.geometry.geom_type.unique()}))
    return {
        "features": str(len(gdf)),
        "geometry": geometry_types,
        "crs": str(gdf.crs),
        "bounds": bounds,
    }


def point_overlay_from_dataframe(df: pd.DataFrame) -> gpd.GeoDataFrame | None:
    lon_col, lat_col = detect_point_columns(df)
    if not lon_col or not lat_col:
        return None
    subset = df[[lon_col, lat_col]].dropna().copy()
    if subset.empty:
        return None
    return gpd.GeoDataFrame(
        subset,
        geometry=gpd.points_from_xy(subset[lon_col], subset[lat_col]),
        crs="EPSG:4326",
    )


def centroid_overlay(gdf: gpd.GeoDataFrame, limit: int = 250) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    geom_types = set(gdf.geometry.geom_type.unique())
    if geom_types <= {"Point", "MultiPoint"}:
        return gdf.head(limit)
    projected = gdf.to_crs(gdf.estimate_utm_crs() or gdf.crs)
    centroids = projected.geometry.centroid.to_crs(gdf.crs)
    out = gdf.copy().head(limit)
    out = out.set_geometry(centroids.head(limit))
    return out


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None
