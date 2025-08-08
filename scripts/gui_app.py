
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ensure the package is importable when running this script directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from tank_model import TankModel, ModelConfig
from tank_model.parameters import Parameters
from tank_model.et import ensure_pet
from tank_model.io import load_csv, subset_period, tag_hydrology
from tank_model.metrics import nse, kge, bias_pct

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Modelo de Tanques – Cuencas Pequeñas")
        self.geometry("1200x700")
        self.df = None
        self.sim = None
        self.pet_method = tk.StringVar(value="column")

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, width=500)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=2)

        self.nb = ttk.Notebook(left)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self._tab_datos()
        self._tab_param()
        self._tab_pet()
        self._tab_sim()
        self._tab_analisis()

        self.fig, self.ax = plt.subplots(figsize=(6,5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _tab_datos(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="1) Datos & IO")
        frm = ttk.Frame(tab, padding=10); frm.pack(fill=tk.BOTH, expand=True)

        self.entry_path = ttk.Entry(frm, width=50)
        btn_browse = ttk.Button(frm, text="Cargar CSV", command=self._browse_csv)
        ttk.Label(frm, text="date,P_mm,PET_mm (o columnas para PET)").grid(row=0, column=0, sticky="w")
        self.entry_path.grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        btn_browse.grid(row=1, column=1, padx=5)

    def _tab_param(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="2) Parámetros")
        frm = ttk.Frame(tab, padding=10); frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Parámetros del modelo").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(frm, text="A1").grid(row=1, column=0, sticky="e")
        ttk.Entry(frm, width=10).grid(row=1, column=1, padx=2, pady=2, sticky="w")
        ttk.Label(frm, text="B1").grid(row=2, column=0, sticky="e")
        ttk.Entry(frm, width=10).grid(row=2, column=1, padx=2, pady=2, sticky="w")

    def _tab_pet(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="3) PET")
        frm = ttk.Frame(tab, padding=10); frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Método de PET").grid(row=0, column=0, sticky="w")
        methods = [("Columna CSV", "column"), ("Hamon", "hamon"), ("Hargreaves", "hargreaves")]
        for i, (txt, val) in enumerate(methods, start=1):
            ttk.Radiobutton(frm, text=txt, variable=self.pet_method, value=val).grid(row=i, column=0, sticky="w")

    def _tab_sim(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="4) Simulación")
        frm = ttk.Frame(tab, padding=10); frm.pack(fill=tk.BOTH, expand=True)

        ttk.Button(
            frm,
            text="Ejecutar simulación",
            command=lambda: messagebox.showinfo("Simulación", "No implementada"),
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")

    def _tab_analisis(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="5) Análisis")
        frm = ttk.Frame(tab, padding=10); frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Resultados de análisis aparecerán aquí").pack(anchor="w")

    def _browse_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV","*.csv"), ("All","*.*")])
        if not path: return
        self.entry_path.delete(0, tk.END)
        self.entry_path.insert(0, path)
        try:
            self.df = load_csv(path)
            messagebox.showinfo("OK", f"Cargado {len(self.df)} filas.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    App().mainloop()


