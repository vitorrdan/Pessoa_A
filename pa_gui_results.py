# -*- coding: utf-8 -*-
"""PESSOA A - Aba 'Treino/Resultados': exibe curvas de loss e tabela de metricas
produzidas por B e C (arquivos .json). Nao treina nada.

FORMATO DO JSON (1 arquivo por experimento):
 {"name": "ResNet50 - fusao", "train_loss": [...], "val_loss": [...],
  "train_time_s": 842.0,
  "metrics": {"sensibilidade":..,"especificidade":..,"precisao":..,"acuracia":..,"f1":..,"auc":..}}
Campos opcionais: "val_auc": [...], "notes": "lr=1e-4, bs=16 ..."."""
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from pa_gui_common import make_canvas

COLS = ["name", "sensibilidade", "especificidade", "precisao", "acuracia", "f1", "auc", "train_time_s"]
# Nomes alternativos aceitos nos JSONs (ingles; prefixos m05_ = limiar 0,5 e mt_ = limiar ajustado,
# usados pela Pessoa C). Ordem de busca: nome puro, m05_, mt_.
ALIASES = {"sensibilidade": ["sensitivity", "sens", "recall"],
           "especificidade": ["specificity", "spec"],
           "precisao": ["precision"], "acuracia": ["accuracy", "acc"],
           "f1": ["f1_score"], "auc": ["roc_auc", "auc_roc"]}


def get_metric(m: dict, key: str):
    for prefix in ("", "m05_", "mt_"):
        for name in [key] + ALIASES.get(key, []):
            if prefix + name in m:
                return m[prefix + name]
    return None


class ResultsTab(ttk.Frame):
    def __init__(self, parent, state):
        super().__init__(parent)
        self.state = state
        self.runs = []
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", padx=6, pady=4)
        ttk.Button(top, text="Carregar resultados (.json)…", command=self.load).pack(side="left")
        ttk.Button(top, text="Limpar", command=self.clear).pack(side="left", padx=4)
        ttk.Button(top, text="Exportar tabela CSV…", command=self.export).pack(side="left")
        self.tree = ttk.Treeview(self, columns=COLS, show="headings", height=7)
        for c in COLS:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=150 if c == "name" else 95, anchor="center")
        self.tree.pack(side="top", fill="x", padx=6)
        self.fig, self.canvas = make_canvas(self, (9, 3.6))

    def load(self):
        for p in filedialog.askopenfilenames(title="Resultados", filetypes=[("JSON", "*.json")]):
            try:
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                d.setdefault("name", p.split("/")[-1])
                self.runs.append(d)
            except Exception as e:
                messagebox.showerror("Resultados", f"{p}\n{e}")
        self.redraw()

    def clear(self):
        self.runs = []
        self.redraw()

    def redraw(self):
        self.tree.delete(*self.tree.get_children())
        for r in self.runs:
            m = r.get("metrics") or r              # metricas aninhadas ou no topo do JSON
            vals = [r.get("name", "?")] + [self._fmt(get_metric(m, c)) for c in COLS[1:7]] + \
                   [self._fmt(r.get("train_time_s", r.get("train_time")), 0)]
            self.tree.insert("", "end", values=vals)
        self.fig.clear()
        ax = self.fig.subplots()
        for r in self.runs:
            if r.get("train_loss"):
                ln, = ax.plot(r["train_loss"], label=f"{r['name']} (treino)")
                if r.get("val_loss"):
                    ax.plot(r["val_loss"], "--", color=ln.get_color(), label=f"{r['name']} (val)")
        ax.set_xlabel("época")
        ax.set_ylabel("loss")
        ax.set_title("Convergência da loss")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    @staticmethod
    def _fmt(v, nd=3):
        return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v))

    def export(self):
        if not self.runs:
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not p:
            return
        import pandas as pd
        rows = [{"name": r.get("name"), **{c: get_metric(r.get("metrics") or r, c) for c in COLS[1:7]},
                 "train_time_s": r.get("train_time_s", r.get("train_time"))} for r in self.runs]
        pd.DataFrame(rows).to_csv(p, index=False)
