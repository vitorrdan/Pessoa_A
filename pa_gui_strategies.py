# -*- coding: utf-8 -*-
"""PESSOA A - Aba 'Estrategias de corte' (item c): compara central, maior area,
media, maximo e media ponderada pela area tumoral no paciente carregado."""
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from pa_gui_common import make_canvas, run_async
from pa_models import make_predict_fn
from pa_slices import STRATEGIES, central_slice, max_area_slice, predict_volume


class StrategiesTab(ttk.Frame):
    def __init__(self, parent, state):
        super().__init__(parent)
        self.state = state
        state.listeners.append(self._on_state)
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", padx=6, pady=4)
        ttk.Label(top, text="Modelo:").pack(side="left")
        self.model_var = tk.StringVar()
        self.model_cb = ttk.Combobox(top, textvariable=self.model_var, width=34, state="readonly")
        self.model_cb.pack(side="left", padx=4)
        ttk.Button(top, text="Comparar estratégias no paciente atual", command=self.run
                   ).pack(side="left", padx=6)
        self.status = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.status).pack(side="left")
        cols = ["estrategia", "P(pCR)", "classe prevista", "cortes usados"]
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=5)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=260 if c == "estrategia" else 120, anchor="center")
        self.tree.pack(side="top", fill="x", padx=6)
        self.fig, self.canvas = make_canvas(self, (9, 3.8))

    def _on_state(self, what):
        if what == "models":
            names = list(self.state.models)
            self.model_cb["values"] = names
            if names and not self.model_var.get():
                self.model_var.set(names[0])

    def run(self):
        if not self.state.need_case():
            return
        name = self.model_var.get()
        if not name:
            messagebox.showinfo("Modelo", "Carregue um modelo (botão 'Carregar pesos…' no topo).")
            return
        spec, lc = self.state.models[name], self.state.case
        missing = [p for p in spec.phases_used if p not in lc.vols]
        if missing:
            messagebox.showerror("Fases", f"O paciente atual não tem as fases {missing} exigidas pelo modelo.")
            return
        self.status.set("Calculando…")
        strategies = [s for s in STRATEGIES if lc.mask is not None or s not in ("max_area", "tumor_wmean")]

        def work():
            pf = make_predict_fn(spec, lc.vols)
            per_slice = np.asarray(pf(list(range(lc.n_slices))))      # 1 inferencia por corte, reaproveitada
            lookup = lambda zs: per_slice[list(zs)]
            return {s: predict_volume(lookup, lc.n_slices, lc.mask, s) for s in strategies}, per_slice

        run_async(self.winfo_toplevel(), work, lambda r: self._show(r, spec, lc),
                  lambda e: (self.status.set(""), messagebox.showerror("Estratégias", str(e))))

    def _show(self, result, spec, lc):
        res, per = result
        self.status.set("")
        self.tree.delete(*self.tree.get_children())
        for s, r in res.items():
            sl = r["slices"]
            txt = f"z={sl[0]}" if len(sl) == 1 else f"todos ({len(sl)})"
            self.tree.insert("", "end", values=(STRATEGIES[s], f"{r['prob']:.3f}",
                                                "pCR" if r["prob"] >= spec.threshold else "sem pCR", txt))
        self.fig.clear()
        ax = self.fig.subplots()
        z = np.arange(lc.n_slices)
        ax.plot(z, per, "o-", color="#c44e52", label="P(pCR) por corte")
        ax.axhline(spec.threshold, color="gray", ls=":", label=f"limiar {spec.threshold:.2f}")
        ax.axvline(central_slice(lc.n_slices), color="#4c72b0", ls="--", label="corte central")
        if lc.mask is not None:
            ax.axvline(max_area_slice(lc.mask, lc.n_slices), color="#55a868", ls="--", label="maior área")
            ax2 = ax.twinx()
            ax2.fill_between(z, lc.mask.sum(axis=(0, 1)), alpha=0.15, color="green")
            ax2.set_ylabel("área tumoral (voxels)")
        ax.set_ylim(0, 1)
        ax.set_xlabel("corte z")
        ax.set_ylabel("P(pCR)")
        ax.set_title(f"{lc.pid} - rótulo real: {lc.label}")
        ax.legend(fontsize=7, loc="upper left")
        self.fig.tight_layout()
        self.canvas.draw_idle()
