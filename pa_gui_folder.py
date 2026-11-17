import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from pa_folder_eval import classify_folder, save_rows_csv
from pa_gui_common import run_async
from pa_metrics import format_metrics
from pa_slices import STRATEGIES


class FolderTab(ttk.Frame):
    def __init__(self, parent, state):
        super().__init__(parent)
        self.state = state
        state.listeners.append(self._on_state)
        self.rows = []
        r1 = ttk.Frame(self)
        r1.pack(side="top", fill="x", padx=6, pady=4)
        ttk.Label(r1, text="Modelo:").pack(side="left")
        self.model_var = tk.StringVar()
        self.model_cb = ttk.Combobox(r1, textvariable=self.model_var, width=30, state="readonly")
        self.model_cb.pack(side="left", padx=4)
        ttk.Label(r1, text="Estratégia:").pack(side="left")
        self.strat_var = tk.StringVar(value="mean")
        ttk.Combobox(r1, textvariable=self.strat_var, width=14, state="readonly",
                     values=list(STRATEGIES)).pack(side="left", padx=4)
        ttk.Label(r1, text="Limiar:").pack(side="left")
        self.thr_var = tk.StringVar(value="")
        ttk.Entry(r1, textvariable=self.thr_var, width=5).pack(side="left", padx=4)
        r2 = ttk.Frame(self)
        r2.pack(side="top", fill="x", padx=6)
        ttk.Button(r2, text="Escolher pasta e classificar…", command=self.run).pack(side="left")
        ttk.Button(r2, text="Exportar previsões (CSV)…", command=self.export).pack(side="left", padx=6)
        self.prog = ttk.Progressbar(r2, length=200, mode="determinate")
        self.prog.pack(side="left", padx=6)
        self.status = tk.StringVar(value="Sem rótulos no CSV da pasta, só as previsões são geradas.")
        ttk.Label(self, textvariable=self.status, justify="left").pack(side="top", anchor="w", padx=8)
        cols = ["pid", "label", "P(pCR)", "pred", "estratégia usada", "erro"]
        self.tree = ttk.Treeview(self, columns=cols, show="headings")
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=130 if c != "erro" else 260, anchor="center")
        self.tree.pack(side="top", fill="both", expand=True, padx=6, pady=4)

    def _on_state(self, what):
        if what == "models":
            names = list(self.state.models)
            self.model_cb["values"] = names
            if names and not self.model_var.get():
                self.model_var.set(names[0])

    def run(self):
        name = self.model_var.get()
        if name not in self.state.models:
            messagebox.showinfo("Modelo", "Carregue um modelo (botão 'Carregar pesos…' no topo).")
            return
        folder = filedialog.askdirectory(title="Pasta com os .nii (e CSV, se houver rótulos)")
        if not folder:
            return
        spec, strat = self.state.models[name], self.strat_var.get()
        try:
            thr = float(self.thr_var.get()) if self.thr_var.get().strip() else None
        except ValueError:
            messagebox.showerror("Limiar", "Limiar inválido.")
            return

        self._progress = (0, 1, "")
        self._running = True
        self._poll_progress()

        def progress(i, n, pid):
            self._progress = (i, n, pid)

        run_async(self.winfo_toplevel(),
                  lambda: classify_folder(spec, folder, strat, threshold=thr, progress=progress),
                  self._done, self._failed)

    def _poll_progress(self):
        i, n, pid = self._progress
        self.prog.config(maximum=max(n, 1), value=i)
        self.status.set(f"{i}/{n}  {pid}")
        if self._running:
            self.after(200, self._poll_progress)

    def _failed(self, e):
        self._running = False
        messagebox.showerror("Pasta", str(e))

    def _done(self, out):
        self._running = False
        self.rows = out["rows"]
        self.tree.delete(*self.tree.get_children())
        for r in self.rows:
            self.tree.insert("", "end", values=(r["pid"], r["label"], f"{r['prob']:.3f}", r["pred"],
                                                r["strategy_used"], r["erro"]))
        n_err = sum(1 for r in self.rows if r["erro"])
        txt = f"{len(self.rows)} pacientes em {out['seconds']:.1f}s ({n_err} com erro)."
        if out["metrics"]:
            txt += "\n" + format_metrics(out["metrics"])
        else:
            txt += "\nSem rótulos: métricas não calculadas."
        self.status.set(txt)

    def export(self):
        if not self.rows:
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            save_rows_csv(self.rows, p)
