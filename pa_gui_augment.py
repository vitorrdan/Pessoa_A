# -*- coding: utf-8 -*-
"""PESSOA A - Aba 'Aumento de dados' (item b): original + 10 variacoes e
grafico de balanceamento de classes (antes/depois)."""
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from pa_augment import augment_variants, balance_plan
from pa_config import AUG_ANGLES, PHASES
from pa_gui_common import make_canvas, phase_choices, phase_from_choice
from pa_nii_io import orient_like
from pa_config import DEFAULT_ORIENTATION, DEFAULT_FLIP_LR


class AugmentTab(ttk.Frame):
    def __init__(self, parent, state):
        super().__init__(parent)
        self.state = state
        self._build()

    def _build(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self.f_var = ttk.Frame(nb)
        self.f_bal = ttk.Frame(nb)
        nb.add(self.f_var, text="Variações de uma imagem")
        nb.add(self.f_bal, text="Balanceamento de classes")

        # --- variacoes
        top = ttk.Frame(self.f_var)
        top.pack(side="top", fill="x", padx=6, pady=4)
        ttk.Label(top, text="Fase:").pack(side="left")
        self.phase_var = tk.StringVar(value=phase_choices()[1])
        ttk.Combobox(top, textvariable=self.phase_var, values=phase_choices(), width=30,
                     state="readonly").pack(side="left", padx=4)
        ttk.Label(top, text="Corte z:").pack(side="left", padx=(8, 0))
        self.z_var = tk.IntVar(value=0)
        self.z_spin = ttk.Spinbox(top, from_=0, to=0, textvariable=self.z_var, width=5)
        self.z_spin.pack(side="left", padx=4)
        ttk.Button(top, text="Usar corte de maior área tumoral", command=self._use_max_area
                   ).pack(side="left", padx=4)
        ttk.Button(top, text="Gerar variações", command=self.show_variants).pack(side="left", padx=4)
        self.fig_v, self.canvas_v = make_canvas(self.f_var, (11, 4.8))

        # --- balanceamento
        top2 = ttk.Frame(self.f_bal)
        top2.pack(side="top", fill="x", padx=6, pady=4)
        ttk.Label(top2, text="Dados:").pack(side="left")
        self.bal_split = tk.StringVar(value="treino+validação")
        ttk.Combobox(top2, textvariable=self.bal_split, width=18, state="readonly",
                     values=["treino+validação", "treino", "validação", "todos"]).pack(side="left", padx=4)
        ttk.Button(top2, text="Calcular do dataset", command=self.from_dataset).pack(side="left", padx=4)
        ttk.Label(top2, text="ou manual  sem pCR:").pack(side="left", padx=(10, 0))
        self.n0 = tk.StringVar(value="")
        self.n1 = tk.StringVar(value="")
        ttk.Entry(top2, textvariable=self.n0, width=6).pack(side="left")
        ttk.Label(top2, text="com pCR:").pack(side="left")
        ttk.Entry(top2, textvariable=self.n1, width=6).pack(side="left")
        ttk.Button(top2, text="Calcular", command=self.from_manual).pack(side="left", padx=4)
        self.bal_txt = tk.StringVar(value="")
        ttk.Label(self.f_bal, textvariable=self.bal_txt, justify="left").pack(side="bottom", anchor="w",
                                                                              padx=8, pady=4)
        self.fig_b, self.canvas_b = make_canvas(self.f_bal, (9, 4))

    # ---- variacoes ------------------------------------------------------------
    def _use_max_area(self):
        lc = self.state.case
        if not self.state.need_case():
            return
        if lc.mask is None:
            messagebox.showinfo("Máscara", "Este paciente não tem máscara carregada.")
            return
        from pa_slices import max_area_slice
        self.z_var.set(max_area_slice(lc.mask, lc.n_slices))
        self.show_variants()

    def show_variants(self):
        if not self.state.need_case():
            return
        lc = self.state.case
        self.z_spin.config(to=lc.n_slices - 1)
        ph = phase_from_choice(self.phase_var.get())
        if ph not in lc.vols:
            ph = next(iter(lc.vols))
        z = int(np.clip(self.z_var.get(), 0, lc.n_slices - 1))
        base = lc.vols[ph][:, :, z]
        variants = augment_variants(base, AUG_ANGLES)
        self.fig_v.clear()
        axes = self.fig_v.subplots(3, 5)
        n = len(AUG_ANGLES)
        o, fl = DEFAULT_ORIENTATION, DEFAULT_FLIP_LR
        for ax in axes.ravel():
            ax.axis("off")
        axes[0, 2].imshow(orient_like(base, o, fl), cmap="gray",
                          origin="lower" if o == "T_lower" else "upper", vmin=0, vmax=1)
        axes[0, 2].set_title(f"ORIGINAL ({ph}, z={z})", fontsize=8)
        for i, (tag, img) in enumerate(variants):
            ax = axes[1 + i // n, i % n]
            ax.imshow(orient_like(img, o, fl), cmap="gray",
                      origin="lower" if o == "T_lower" else "upper", vmin=0, vmax=1)
            ax.set_title(tag, fontsize=8)
        self.fig_v.suptitle("Aumento de dados: 5 rotações × (original, espelhada) = 10 variações", fontsize=10)
        self.fig_v.tight_layout()
        self.canvas_v.draw_idle()

    # ---- balanceamento ---------------------------------------------------------
    def from_dataset(self):
        idx = self.state.index
        if idx is None:
            messagebox.showinfo("Dataset", "Abra o dataset na aba 'Visualizar' primeiro.")
            return
        sel = self.bal_split.get()
        splits = {"treino+validação": ["train", "val"], "treino": ["train"],
                  "validação": ["val"], "todos": [None]}[sel]
        counts = {0: 0, 1: 0}
        for sp in splits:
            for k, v in idx.class_counts(sp).items():
                counts[k] += v
        self._plot_balance(counts)

    def from_manual(self):
        try:
            self._plot_balance({0: int(self.n0.get()), 1: int(self.n1.get())})
        except ValueError:
            messagebox.showerror("Contagem", "Informe números inteiros para as duas classes.")

    def _plot_balance(self, counts):
        try:
            plan = balance_plan(counts)
        except ValueError as e:
            messagebox.showerror("Balanceamento", str(e))
            return
        self.fig_b.clear()
        ax = self.fig_b.subplots()
        labels = ["Original", "Aumento 10×\n(uniforme)", "Aumento\nproporcional"]
        groups = [plan["before"], plan["after_aug"], plan["after_prop"]]
        x = np.arange(3)
        w = 0.38
        b0 = ax.bar(x - w / 2, [g[0] for g in groups], w, label="sem pCR (0)", color="#4c72b0")
        b1 = ax.bar(x + w / 2, [g[1] for g in groups], w, label="pCR (1)", color="#dd8452")
        ax.bar_label(b0, fontsize=8)
        ax.bar_label(b1, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("nº de imagens")
        ax.legend()
        ax.set_title("Distribuição das classes (antes e depois do aumento de dados)")
        self.fig_b.tight_layout()
        self.canvas_b.draw_idle()
        vp = plan["variants_per_class"]
        cw = plan["class_weights"]
        self.bal_txt.set(
            f"Razão maioria/minoria = {plan['ratio']:.2f}  ->  desbalanceamento "
            f"{'RELEVANTE' if plan['relevant'] else 'pequeno'} (limiar 1,5)\n"
            f"O aumento 10× uniforme mantém a razão. Aumento proporcional: {vp[0]} variações/img (sem pCR) "
            f"e {vp[1]} variações/img (pCR).\n"
            f"Pesos para a loss: w0={cw[0]:.2f}, w1={cw[1]:.2f}  |  pos_weight (BCEWithLogits) = "
            f"{plan['pos_weight']:.2f}")
