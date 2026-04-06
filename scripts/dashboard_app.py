from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

from common import INPUT_DIR, OUTPUT_DIR
from dashboard_support import (
    LayerEntry,
    centroid_overlay,
    layer_summary,
    list_catalog_layers,
    list_input_csvs,
    list_output_images,
    list_output_tables,
    load_layer,
    load_series_dataframe,
    numeric_columns,
    point_overlay_from_dataframe,
)

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk


class ModHDDashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ModHD Dashboard")
        self.geometry("1500x920")
        self.minsize(1280, 760)

        self.series_files: list[Path] = []
        self.output_images: list[Path] = []
        self.output_tables: list[Path] = []
        self.catalog_layers: list[LayerEntry] = []
        self.current_df: pd.DataFrame | None = None
        self.current_image: ImageTk.PhotoImage | None = None

        self._build_layout()
        self.refresh_all()

    def _build_layout(self) -> None:
        toolbar = ttk.Frame(self, padding=(12, 10))
        toolbar.pack(fill=tk.X)
        ttk.Label(toolbar, text="Dashboard Hidrológico ModHD", font=("Helvetica", 18, "bold")).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Refrescar", command=self.refresh_all).pack(side=tk.RIGHT)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self._build_overview_tab()
        self._build_series_tab()
        self._build_calibration_tab()
        self._build_images_tab()
        self._build_map_tab()

    def _build_overview_tab(self) -> None:
        self.tab_overview = ttk.Frame(self.notebook, padding=16)
        self.notebook.add(self.tab_overview, text="Resumen")

        self.summary_vars = {
            "inputs": tk.StringVar(value="0"),
            "outputs": tk.StringVar(value="0"),
            "layers": tk.StringVar(value="0"),
            "images": tk.StringVar(value="0"),
        }

        cards = ttk.Frame(self.tab_overview)
        cards.pack(fill=tk.X)
        for idx, (label, var) in enumerate(self.summary_vars.items()):
            card = ttk.LabelFrame(cards, text=label.capitalize(), padding=16)
            card.grid(row=0, column=idx, padx=8, pady=8, sticky="nsew")
            ttk.Label(card, textvariable=var, font=("Helvetica", 22, "bold")).pack()
            cards.columnconfigure(idx, weight=1)

        self.overview_text = tk.Text(self.tab_overview, height=24, wrap="word")
        self.overview_text.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

    def _build_series_tab(self) -> None:
        self.tab_series = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.tab_series, text="Series")

        controls = ttk.Frame(self.tab_series)
        controls.pack(fill=tk.X)

        self.series_file_var = tk.StringVar()
        self.series_x_var = tk.StringVar(value="date")
        self.series_y_var = tk.StringVar()

        ttk.Label(controls, text="Serie").grid(row=0, column=0, sticky="w")
        self.series_combo = ttk.Combobox(controls, textvariable=self.series_file_var, state="readonly", width=40)
        self.series_combo.grid(row=1, column=0, padx=(0, 12), sticky="ew")
        self.series_combo.bind("<<ComboboxSelected>>", lambda _: self.on_series_change())

        ttk.Label(controls, text="Eje X").grid(row=0, column=1, sticky="w")
        self.series_x_combo = ttk.Combobox(controls, textvariable=self.series_x_var, state="readonly", width=18)
        self.series_x_combo.grid(row=1, column=1, padx=(0, 12), sticky="ew")

        ttk.Label(controls, text="Variable").grid(row=0, column=2, sticky="w")
        self.series_y_combo = ttk.Combobox(controls, textvariable=self.series_y_var, state="readonly", width=24)
        self.series_y_combo.grid(row=1, column=2, padx=(0, 12), sticky="ew")

        ttk.Button(controls, text="Graficar", command=self.plot_selected_series).grid(row=1, column=3, sticky="ew")
        ttk.Button(controls, text="Abrir CSV", command=self.open_series_file).grid(row=1, column=4, padx=(12, 0), sticky="ew")

        for idx in range(5):
            controls.columnconfigure(idx, weight=1 if idx < 3 else 0)

        content = ttk.Frame(self.tab_series)
        content.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        left = ttk.Frame(content)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.LabelFrame(content, text="Estadísticos", padding=8)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(12, 0))

        self.series_fig, self.series_ax = plt.subplots(figsize=(8, 5))
        self.series_canvas = FigureCanvasTkAgg(self.series_fig, master=left)
        self.series_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.series_stats = tk.Text(right, width=42, wrap="word")
        self.series_stats.pack(fill=tk.BOTH, expand=True)

    def _build_calibration_tab(self) -> None:
        self.tab_calibration = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.tab_calibration, text="Calibración")

        top = ttk.Frame(self.tab_calibration)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Parámetros óptimos y métricas del último run", font=("Helvetica", 12, "bold")).pack(anchor="w")

        panes = ttk.PanedWindow(self.tab_calibration, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        self.params_table = ttk.Treeview(panes, columns=("param", "value"), show="headings", height=18)
        self.params_table.heading("param", text="Parámetro")
        self.params_table.heading("value", text="Valor")
        self.params_table.column("param", width=180, anchor="w")
        self.params_table.column("value", width=180, anchor="e")

        self.metrics_table = ttk.Treeview(
            panes,
            columns=("subset", "NSE", "BIAS_pct", "RMSE_m3s", "R2"),
            show="headings",
            height=18,
        )
        for col in ("subset", "NSE", "BIAS_pct", "RMSE_m3s", "R2"):
            self.metrics_table.heading(col, text=col)
            self.metrics_table.column(col, width=120, anchor="center")

        panes.add(self.params_table, weight=1)
        panes.add(self.metrics_table, weight=1)

    def _build_images_tab(self) -> None:
        self.tab_images = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.tab_images, text="Imágenes")

        layout = ttk.Frame(self.tab_images)
        layout.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(layout, text="Imágenes generadas", padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.LabelFrame(layout, text="Vista previa dinámica", padding=8)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(12, 0))

        self.images_list = tk.Listbox(left, width=44, height=28)
        self.images_list.pack(fill=tk.BOTH, expand=True)
        self.images_list.bind("<<ListboxSelect>>", lambda _: self.show_selected_image())

        self.image_info = ttk.Label(right, text="Selecciona una imagen", anchor="w", justify="left")
        self.image_info.pack(fill=tk.X, pady=(0, 8))
        self.image_label = ttk.Label(right)
        self.image_label.pack(fill=tk.BOTH, expand=True)

    def _build_map_tab(self) -> None:
        self.tab_map = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.tab_map, text="Mapa")

        controls = ttk.Frame(self.tab_map)
        controls.pack(fill=tk.X)
        self.layer_var = tk.StringVar()
        self.map_series_var = tk.StringVar()

        ttk.Label(controls, text="Capa catálogo").grid(row=0, column=0, sticky="w")
        self.layer_combo = ttk.Combobox(controls, textvariable=self.layer_var, state="readonly", width=50)
        self.layer_combo.grid(row=1, column=0, padx=(0, 12), sticky="ew")
        self.layer_combo.bind("<<ComboboxSelected>>", lambda _: self.plot_selected_layer())

        ttk.Label(controls, text="Serie para puntos").grid(row=0, column=1, sticky="w")
        self.map_series_combo = ttk.Combobox(controls, textvariable=self.map_series_var, state="readonly", width=36)
        self.map_series_combo.grid(row=1, column=1, padx=(0, 12), sticky="ew")
        self.map_series_combo.bind("<<ComboboxSelected>>", lambda _: self.plot_selected_layer())

        ttk.Button(controls, text="Redibujar mapa", command=self.plot_selected_layer).grid(row=1, column=2, sticky="ew")
        controls.columnconfigure(0, weight=2)
        controls.columnconfigure(1, weight=1)

        content = ttk.Frame(self.tab_map)
        content.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        left = ttk.Frame(content)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.LabelFrame(content, text="Resumen de capa", padding=8)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(12, 0))

        self.map_fig, self.map_ax = plt.subplots(figsize=(8, 6))
        self.map_canvas = FigureCanvasTkAgg(self.map_fig, master=left)
        self.map_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.map_info = tk.Text(right, width=42, wrap="word")
        self.map_info.pack(fill=tk.BOTH, expand=True)

    def refresh_all(self) -> None:
        self.series_files = list_input_csvs()
        self.output_images = list_output_images()
        self.output_tables = list_output_tables()
        self.catalog_layers = list_catalog_layers()

        self.summary_vars["inputs"].set(str(len(self.series_files)))
        self.summary_vars["outputs"].set(str(len(self.output_tables)))
        self.summary_vars["layers"].set(str(len(self.catalog_layers)))
        self.summary_vars["images"].set(str(len(self.output_images)))

        self._refresh_overview()
        self._refresh_series_controls()
        self._refresh_calibration_tables()
        self._refresh_images_list()
        self._refresh_map_controls()

    def _refresh_overview(self) -> None:
        self.overview_text.delete("1.0", tk.END)
        lines = [
            "Estructura actual del dashboard",
            "",
            f"- Input: {INPUT_DIR}",
            f"- Output: {OUTPUT_DIR}",
            f"- Series disponibles: {len(self.series_files)}",
            f"- Tablas de salida: {len(self.output_tables)}",
            f"- Imágenes generadas: {len(self.output_images)}",
            f"- Capas del catálogo: {len(self.catalog_layers)}",
            "",
            "Este dashboard está pensado como base local para una futura integración con CLI-15.",
            "Las imágenes se refrescan dinámicamente, las series se grafican desde data/input y las capas se leen desde catalogo_capas.",
        ]
        self.overview_text.insert("1.0", "\n".join(lines))

    def _refresh_series_controls(self) -> None:
        names = [path.name for path in self.series_files]
        self.series_combo["values"] = names
        self.map_series_combo["values"] = ["(sin overlay)"] + names
        if names and not self.series_file_var.get():
            self.series_file_var.set(names[0])
            self.on_series_change()
        if names and not self.map_series_var.get():
            self.map_series_var.set("(sin overlay)")

    def _refresh_calibration_tables(self) -> None:
        for table in (self.params_table, self.metrics_table):
            for item in table.get_children():
                table.delete(item)

        params_path = OUTPUT_DIR / "parametros_optimos.csv"
        metrics_path = OUTPUT_DIR / "metricas_desempeno.csv"
        if params_path.exists():
            params_df = pd.read_csv(params_path)
            if not params_df.empty:
                for key, value in params_df.iloc[0].items():
                    self.params_table.insert("", tk.END, values=(key, f"{value}"))
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            for _, row in metrics_df.iterrows():
                subset = row.get("subset", row.get("Unnamed: 0", ""))
                self.metrics_table.insert(
                    "",
                    tk.END,
                    values=(
                        subset,
                        f"{row.get('NSE', float('nan')):.4f}",
                        f"{row.get('BIAS_pct', float('nan')):.2f}",
                        f"{row.get('RMSE_m3s', float('nan')):.4f}",
                        f"{row.get('R2', float('nan')):.4f}",
                    ),
                )

    def _refresh_images_list(self) -> None:
        self.images_list.delete(0, tk.END)
        for path in self.output_images:
            self.images_list.insert(tk.END, path.name)
        if self.output_images:
            self.images_list.selection_clear(0, tk.END)
            self.images_list.selection_set(0)
            self.show_selected_image()

    def _refresh_map_controls(self) -> None:
        layer_labels = [f"{entry.source_group} :: {entry.name}" for entry in self.catalog_layers]
        self.layer_combo["values"] = layer_labels
        if layer_labels and not self.layer_var.get():
            self.layer_var.set(layer_labels[0])
            self.plot_selected_layer()

    def on_series_change(self) -> None:
        path = self._selected_series_path(self.series_file_var.get())
        if path is None:
            return
        self.current_df = load_series_dataframe(path)
        cols = list(self.current_df.columns)
        numeric = numeric_columns(self.current_df)
        self.series_x_combo["values"] = cols
        self.series_y_combo["values"] = numeric
        self.series_x_var.set("date" if "date" in cols else cols[0])
        if numeric:
            preferred = "Qobs_m3s" if "Qobs_m3s" in numeric else numeric[0]
            self.series_y_var.set(preferred)
        self.plot_selected_series()

    def open_series_file(self) -> None:
        path = filedialog.askopenfilename(initialdir=str(INPUT_DIR), filetypes=[("CSV", "*.csv")])
        if not path:
            return
        name = Path(path).name
        if name not in self.series_combo["values"]:
            self.series_files.append(Path(path))
            self.series_combo["values"] = [p.name for p in self.series_files]
        self.series_file_var.set(name)
        self.on_series_change()

    def plot_selected_series(self) -> None:
        if self.current_df is None:
            return
        x_col = self.series_x_var.get()
        y_col = self.series_y_var.get()
        if not x_col or not y_col:
            return
        self.series_ax.clear()
        df = self.current_df.copy()
        self.series_ax.plot(df[x_col], df[y_col], color="#0b6e4f", linewidth=1.5)
        self.series_ax.set_title(f"{y_col} vs {x_col}")
        self.series_ax.set_xlabel(x_col)
        self.series_ax.set_ylabel(y_col)
        self.series_ax.grid(True, alpha=0.3)
        self.series_fig.autofmt_xdate()
        self.series_canvas.draw()

        stats = df[[y_col]].describe().to_string()
        self.series_stats.delete("1.0", tk.END)
        self.series_stats.insert(
            "1.0",
            f"Archivo: {self.series_file_var.get()}\n\nColumnas: {', '.join(df.columns)}\n\n{stats}",
        )

    def show_selected_image(self) -> None:
        selection = self.images_list.curselection()
        if not selection:
            return
        path = self.output_images[selection[0]]
        img = Image.open(path)
        img.thumbnail((960, 680))
        self.current_image = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.current_image)
        self.image_info.configure(text=f"{path.name}\n{path}")

    def plot_selected_layer(self) -> None:
        if not self.layer_var.get():
            return
        entry = self.catalog_layers[self.layer_combo.current()]
        gdf = load_layer(entry.path)
        self.map_ax.clear()
        gdf.plot(ax=self.map_ax, facecolor="#d9e8f5", edgecolor="#0f4c5c", linewidth=0.8)

        points_note = "No hay overlay de puntos."
        map_series_name = self.map_series_var.get()
        if map_series_name and map_series_name != "(sin overlay)":
            series_path = self._selected_series_path(map_series_name)
            if series_path is not None:
                df = load_series_dataframe(series_path)
                points_gdf = point_overlay_from_dataframe(df)
                if points_gdf is not None and not points_gdf.empty:
                    points_gdf.plot(ax=self.map_ax, color="#c1121f", markersize=28)
                    points_note = f"Overlay con puntos desde {series_path.name}"
                else:
                    centroids = centroid_overlay(gdf)
                    if not centroids.empty:
                        centroids.plot(ax=self.map_ax, color="#c1121f", markersize=18, alpha=0.7)
                        points_note = "No se detectaron columnas lon/lat; se muestran centroides de la capa."
        else:
            centroids = centroid_overlay(gdf)
            if not centroids.empty:
                centroids.plot(ax=self.map_ax, color="#f77f00", markersize=12, alpha=0.5)
                points_note = "Se muestran centroides de referencia de la capa."

        self.map_ax.set_title(f"{entry.name} ({entry.source_group})")
        self.map_ax.set_axis_off()
        self.map_canvas.draw()

        summary = layer_summary(gdf)
        self.map_info.delete("1.0", tk.END)
        self.map_info.insert(
            "1.0",
            "\n".join(
                [
                    f"Capa: {entry.name}",
                    f"Grupo: {entry.source_group}",
                    f"Ruta: {entry.path}",
                    f"Features: {summary['features']}",
                    f"Geometría: {summary['geometry']}",
                    f"CRS: {summary['crs']}",
                    f"Bounds: {summary['bounds']}",
                    "",
                    points_note,
                ]
            ),
        )

    def _selected_series_path(self, name: str) -> Path | None:
        for path in self.series_files:
            if path.name == name:
                return path
        return None


def main() -> None:
    try:
        app = ModHDDashboard()
        app.mainloop()
    except Exception as exc:
        messagebox.showerror("Dashboard error", str(exc))


if __name__ == "__main__":
    main()
