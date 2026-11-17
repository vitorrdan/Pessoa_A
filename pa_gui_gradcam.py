import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from pa_config import DEFAULT_FLIP_LR, DEFAULT_ORIENTATION
from pa_gradcam import GradCAM, generate_report_figures, heat_in_mask_metrics, resize_map
from pa_gui_common import make_canvas, run_async
from pa_models import make_predict_fn
from pa_nii_io import load_case, orient_like
from pa_slices import STRATEGIES, predict_volume, representative_slice


class GradcamTab(ttk.Frame):
    def __init__(self, parent, state):
        super().__init__(parent)
        self.state = state
        state.listeners.append(self._on_state)
        r1 = ttk.Frame(self)
        r1.pack(side="top", fill="x", padx=6, pady=4)
        ttk.Label(r1, text="Modelo:").pack(side="left")
        self.model_var = tk.StringVar()
        self.model_cb = ttk.Combobox(r1, textvariable=self.model_var, width=30, state="readonly")
        self.model_cb.pack(side="left", padx=4)
        ttk.Label(r1, text="Estratégia:").pack(side="left")
        self.strat_var = tk.StringVar(value="max_area")
        ttk.Combobox(r1, textvariable=self.strat_var, width=14, state="readonly",
                     values=list(STRATEGIES)).pack(side="left", padx=4)
        ttk.Label(r1, text="Explicar classe:").pack(side="left")
        self.cls_var = tk.StringVar(value="pCR")
        ttk.Combobox(r1, textvariable=self.cls_var, width=7, state="readonly",
                     values=["pCR", "sem pCR"]).pack(side="left", padx=4)
        ttk.Label(r1, text="Camada:").pack(side="left")
        self.layer_var = tk.StringVar(value="média")
        self.layer_cb = ttk.Combobox(r1, textvariable=self.layer_var, width=8, state="readonly",
                                     values=["média"])
        self.layer_cb.pack(side="left", padx=4)
        r2 = ttk.Frame(self)
        r2.pack(side="top", fill="x", padx=6)
        ttk.Button(r2, text="Classificar + Grad-CAM", command=self.run).pack(side="left")
        ttk.Button(r2, text="Figuras de acertos/erros p/ relatório…", command=self.report
                   ).pack(side="left", padx=8)
        ttk.Label(r2, text="Transparência do calor:").pack(side="left", padx=(10, 2))
        self.alpha = tk.DoubleVar(value=0.45)
        tk.Scale(r2, from_=0.1, to=0.9, resolution=0.05, orient="horizontal", length=120,
                 variable=self.alpha, command=lambda v: self._redraw()).pack(side="left")
        self.result_txt = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.result_txt, justify="left", font=("TkDefaultFont", 10, "bold")
                  ).pack(side="top", anchor="w", padx=8, pady=2)
        self.fig, self.canvas = make_canvas(self, (11, 4.2))
        self._last = None

    def _on_state(self, what):
        if what == "models":
            names = list(self.state.models)
            self.model_cb["values"] = names
            if names and not self.model_var.get():
                self.model_var.set(names[0])
            self._update_layers()

    def _update_layers(self):
        name = self.model_var.get()
        if name in self.state.models:
            n = len(self.state.models[name].target_layers)
            self.layer_cb["values"] = ["média"] + [f"camada {i}" for i in range(n)] if n > 1 else ["média"]


    def run(self):
        if not self.state.need_case():
            return
        name = self.model_var.get()
        if name not in self.state.models:
            messagebox.showinfo("Modelo", "Carregue um modelo (botão 'Carregar pesos…' no topo).")
            return
        spec, lc, strat = self.state.models[name], self.state.case, self.strat_var.get()
        missing = [p for p in spec.phases_used if p not in lc.vols]
        if missing:
            messagebox.showerror("Fases", f"O paciente atual não tem as fases {missing}.")
            return
        if strat in ("max_area", "tumor_wmean") and lc.mask is None:
            messagebox.showerror("Máscara", f"A estratégia '{strat}' exige a máscara de segmentação.")
            return
        target = 1 if self.cls_var.get() == "pCR" else 0
        self.result_txt.set("Calculando…")

        def work():
            res = predict_volume(make_predict_fn(spec, lc.vols), lc.n_slices, lc.mask, strat)
            z = representative_slice(res, strat, lc.n_slices)
            ex = GradCAM(spec).explain_slices(lc.vols, [z], target_class=target)
            return res, z, ex

        run_async(self.winfo_toplevel(), work, lambda r: self._done(r, spec, lc, strat),
                  lambda e: (self.result_txt.set(""), messagebox.showerror("Grad-CAM", str(e))))

    def _done(self, r, spec, lc, strat):
        res, z, ex = r
        self._last = dict(spec=spec, lc=lc, z=z, res=res, ex=ex, strat=strat)
        self._redraw()

    def _redraw(self):
        if not self._last:
            return
        L = self._last
        spec, lc, z, res, ex = L["spec"], L["lc"], L["z"], L["res"], L["ex"]
        sel = self.layer_var.get()
        cam = ex["cams"][0] if sel == "média" or not sel.startswith("camada") \
            else ex["per_layer"][int(sel.split()[1])][0]
        ph = spec.phases_used[-1]
        base = lc.vols[ph][:, :, z]
        cam = resize_map(cam, base.shape)
        mk = lc.mask[:, :, z] if lc.mask is not None else None
        o, fl = DEFAULT_ORIENTATION, DEFAULT_FLIP_LR
        origin = "lower" if o == "T_lower" else "upper"
        self.fig.clear()
        a1, a2, a3 = self.fig.subplots(1, 3)
        b = orient_like(base, o, fl)
        a1.imshow(b, cmap="gray", origin=origin)
        a1.set_title(f"Corte z={z} ({ph})", fontsize=9)
        a2.imshow(orient_like(cam, o, fl), cmap="jet", origin=origin, vmin=0, vmax=1)
        a2.set_title("Grad-CAM", fontsize=9)
        a3.imshow(b, cmap="gray", origin=origin)
        a3.imshow(orient_like(cam, o, fl), cmap="jet", origin=origin, alpha=self.alpha.get(), vmin=0, vmax=1)
        if mk is not None and mk.sum() > 0:
            for a in (a1, a3):
                a.contour(orient_like(mk, o, fl), levels=[0.5], colors="white", linewidths=1.0)
        a3.set_title("Sobreposição (contorno = tumor)", fontsize=9)
        for a in (a1, a2, a3):
            a.axis("off")
        self.fig.tight_layout()
        self.canvas.draw_idle()

        pred = "pCR" if res["prob"] >= spec.threshold else "sem pCR"
        real = {None: "desconhecido", 0: "sem pCR", 1: "pCR"}[lc.label]
        txt = (f"P(pCR) = {res['prob']:.3f}  ->  previsto: {pred}   |   real: {real}   "
               f"|   estratégia: {L['strat']}")
        if mk is not None:
            m = heat_in_mask_metrics(cam, mk)
            if not np.isnan(m["heat_fraction"]):
                txt += (f"\nCalor dentro do tumor: {100 * m['heat_fraction']:.1f}% "
                        f"(tumor ocupa {100 * m['area_fraction']:.1f}% da imagem; "
                        f"enriquecimento {m['enrichment']:.2f}×; pico no tumor: "
                        f"{'sim' if m['peak_in_mask'] else 'não'})")
            else:
                txt += "\n(sem tumor neste corte ou mapa nulo: métrica de coerência indisponível)"
        self.result_txt.set(txt)


    def report(self):
        idx = self.state.index
        name = self.model_var.get()
        if idx is None or name not in self.state.models:
            messagebox.showinfo("Relatório", "Abra o dataset e carregue um modelo primeiro.")
            return
        out = filedialog.askdirectory(title="Pasta de saída das figuras")
        if not out:
            return
        spec, strat = self.state.models[name], self.strat_var.get()
        pids = [p for p in idx.pids("test") if idx.cases[p].label is not None] or \
               [p for p in idx.pids() if idx.cases[p].label is not None]
        pids = pids[:60]
        self.result_txt.set(f"Gerando figuras para {len(pids)} pacientes…")

        def work():
            cases = [load_case(idx.cases[p]) for p in pids]
            return generate_report_figures(spec, cases, out, strategy=strat)

        run_async(self.winfo_toplevel(), work,
                  lambda csv: (self.result_txt.set(f"Figuras e métricas salvas em {out}"),
                               messagebox.showinfo("Relatório", f"Pronto.\nCSV de coerência: {csv}")),
                  lambda e: (self.result_txt.set(""), messagebox.showerror("Relatório", str(e))))
