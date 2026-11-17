import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from pa_config import (PHASES, DEFAULT_ORIENTATION, DEFAULT_FLIP_LR)
from pa_gui_common import make_canvas, phase_choices, phase_from_choice, run_async
from pa_nii_io import (DatasetIndex, LoadedCase, load_case, load_mask, load_nii,
                       normalize_volume, orient_slice)


class ViewerTab(ttk.Frame):
    def __init__(self, parent, state):
        super().__init__(parent)
        self.state = state
        self.z = 0
        self._build()
        state.listeners.append(self._on_state)


    def _build(self):
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", padx=6, pady=4)
        ttk.Button(top, text="Abrir pasta do dataset…", command=self.open_dataset).pack(side="left")
        ttk.Button(top, text="Abrir .nii…", command=self.open_nii).pack(side="left", padx=4)
        ttk.Button(top, text="Abrir máscara .nii…", command=self.open_mask).pack(side="left")

        row2 = ttk.Frame(self)
        row2.pack(side="top", fill="x", padx=6)
        ttk.Label(row2, text="Split:").pack(side="left")
        self.split_var = tk.StringVar(value="todos")
        sp = ttk.Combobox(row2, textvariable=self.split_var, width=7, state="readonly",
                          values=["todos", "train", "val", "test"])
        sp.pack(side="left", padx=(2, 8))
        sp.bind("<<ComboboxSelected>>", lambda e: self._fill_patients())
        ttk.Label(row2, text="Paciente:").pack(side="left")
        self.pid_var = tk.StringVar()
        self.pid_cb = ttk.Combobox(row2, textvariable=self.pid_var, width=22, state="readonly")
        self.pid_cb.pack(side="left", padx=(2, 8))
        self.pid_cb.bind("<<ComboboxSelected>>", lambda e: self.load_selected())
        ttk.Label(row2, text="Fase:").pack(side="left")
        self.phase_var = tk.StringVar(value=phase_choices()[0])
        ph = ttk.Combobox(row2, textvariable=self.phase_var, width=30, state="readonly",
                          values=phase_choices())
        ph.pack(side="left", padx=(2, 8))
        ph.bind("<<ComboboxSelected>>", lambda e: self.refresh())

        row3 = ttk.Frame(self)
        row3.pack(side="top", fill="x", padx=6, pady=2)
        ttk.Label(row3, text="Orientação:").pack(side="left")
        self.orient_var = tk.StringVar(value=DEFAULT_ORIENTATION)
        oc = ttk.Combobox(row3, textvariable=self.orient_var, width=9, state="readonly",
                          values=["T_lower", "rot90", "none"])
        oc.pack(side="left", padx=(2, 8))
        oc.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        self.flip_var = tk.BooleanVar(value=DEFAULT_FLIP_LR)
        ttk.Checkbutton(row3, text="Espelhar E/D", variable=self.flip_var,
                        command=self.refresh).pack(side="left", padx=4)
        self.contour_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row3, text="Contorno da máscara sobre a imagem",
                        variable=self.contour_var, command=self.refresh).pack(side="left", padx=4)

        self.info = tk.StringVar(value="Nenhum volume carregado.")
        ttk.Label(self, textvariable=self.info, foreground="#444").pack(side="top", anchor="w", padx=8)

        self.slider = tk.Scale(self, from_=0, to=0, orient="horizontal", label="Corte z",
                               command=self._on_slider, length=400)
        self.slider.pack(side="bottom", fill="x", padx=8, pady=4)

        frame = ttk.Frame(self)
        frame.pack(side="top", fill="both", expand=True)
        self.fig, self.canvas = make_canvas(frame, (9, 4.2))
        self.ax_img, self.ax_mask = self.fig.subplots(1, 2)
        self.canvas.get_tk_widget().bind("<MouseWheel>", self._wheel)
        self.canvas.get_tk_widget().bind("<Button-4>", lambda e: self._step(1))
        self.canvas.get_tk_widget().bind("<Button-5>", lambda e: self._step(-1))
        w = self.canvas.get_tk_widget()
        w.bind("<Enter>", lambda e: w.focus_set())
        w.bind("<Left>", lambda e: self._step(-1))
        w.bind("<Right>", lambda e: self._step(1))


    def open_dataset(self):
        folder = filedialog.askdirectory(title="Pasta raiz do dataset (com .nii e o CSV)")
        if not folder:
            return
        self.info.set("Indexando…")
        self.update_idletasks()
        try:
            idx = DatasetIndex(folder)
        except Exception as e:
            messagebox.showerror("Dataset", f"Falha ao indexar:\n{e}")
            return
        self.state.index = idx
        self.info.set(idx.summary())
        if not idx.pids():
            messagebox.showwarning(
                "Dataset", "Nenhum paciente com as 3 fases foi encontrado.\n"
                "Ajuste PHASE_PATTERNS/MASK_PATTERNS em pa_config.py conforme os nomes dos arquivos.")
        self._fill_patients()
        self.state.notify("index")

    def _fill_patients(self):
        if not self.state.index:
            return
        sp = self.split_var.get()
        pids = self.state.index.pids(None if sp == "todos" else sp)
        self.pid_cb["values"] = pids
        if pids:
            self.pid_var.set(pids[0])
            self.load_selected()

    def load_selected(self):
        pid = self.pid_var.get()
        if not pid or not self.state.index:
            return
        self.info.set(f"Carregando {pid}…")
        self.update_idletasks()
        case = self.state.index.cases[pid]
        run_async(self.winfo_toplevel(), lambda: load_case(case), self._loaded,
                  lambda e: messagebox.showerror("Carregar paciente", str(e)))

    def _loaded(self, lc: LoadedCase):
        self.state.case = lc
        self.slider.config(to=lc.n_slices - 1)
        self.z = lc.n_slices // 2
        self.slider.set(self.z)
        self.refresh()
        self.state.notify("case")

    def open_nii(self):
        path = filedialog.askopenfilename(title="Imagem .nii",
                                          filetypes=[("NIfTI", "*.nii *.nii.gz"), ("Todos", "*.*")])
        if not path:
            return
        try:
            v = load_nii(path)
        except Exception as e:
            messagebox.showerror("Abrir .nii", str(e))
            return
        ph = phase_from_choice(self.phase_var.get())
        mask = self.state.case.mask if self.state.case and \
            self.state.case.mask is not None and self.state.case.mask.shape == v.data.shape else None
        self.state.case = LoadedCase("(arquivo avulso)", None, {ph: normalize_volume(v.data)},
                                     {ph: v.data}, mask, {ph: v})
        self.slider.config(to=v.n_slices - 1)
        self.z = v.n_slices // 2
        self.slider.set(self.z)
        self.refresh()
        self.state.notify("case")

    def open_mask(self):
        if not self.state.need_case():
            return
        path = filedialog.askopenfilename(title="Máscara .nii",
                                          filetypes=[("NIfTI", "*.nii *.nii.gz"), ("Todos", "*.*")])
        if not path:
            return
        m = load_mask(path)
        shp = next(iter(self.state.case.vols.values())).shape
        if m.data.shape != shp:
            messagebox.showerror("Máscara", f"Dimensões diferentes: máscara {m.data.shape} vs imagem {shp}")
            return
        self.state.case.mask = m.data
        self.refresh()
        self.state.notify("case")

    def _on_state(self, what):
        pass


    def _on_slider(self, val):
        self.z = int(float(val))
        self.refresh()

    def _step(self, d):
        if self.state.case is None:
            return
        self.z = int(np.clip(self.z + d, 0, self.state.case.n_slices - 1))
        self.slider.set(self.z)

    def _wheel(self, e):
        self._step(1 if e.delta > 0 else -1)

    def current_phase(self):
        ph = phase_from_choice(self.phase_var.get())
        lc = self.state.case
        return ph if lc and ph in lc.vols else (next(iter(lc.vols)) if lc else ph)

    def refresh(self):
        lc = self.state.case
        for ax in (self.ax_img, self.ax_mask):
            ax.clear()
            ax.axis("off")
        if lc is None:
            self.canvas.draw_idle()
            return
        ph = self.current_phase()
        o, fl = self.orient_var.get(), self.flip_var.get()
        z = min(self.z, lc.n_slices - 1)
        img, origin = orient_slice(lc.vols[ph][:, :, z], o, fl)
        self.ax_img.imshow(img, cmap="gray", origin=origin, vmin=0, vmax=1)
        self.ax_img.set_title(f"{lc.pid} | {ph} | z={z}/{lc.n_slices - 1}", fontsize=9)
        if lc.mask is not None:
            mk, _ = orient_slice(lc.mask[:, :, z], o, fl)
            self.ax_mask.imshow(mk, cmap="gray", origin=origin, vmin=0, vmax=1)
            self.ax_mask.set_title(f"Segmentação (área = {int(lc.mask[:, :, z].sum())} vox)", fontsize=9)
            if self.contour_var.get() and mk.sum() > 0:
                self.ax_img.contour(mk, levels=[0.5], colors="lime", linewidths=1.0)
        else:
            self.ax_mask.set_title("(sem máscara)", fontsize=9)
        v = lc.info.get(ph)
        if v is not None:
            self.info.set(f"shape={v.shape} | espaçamento={tuple(round(s, 2) for s in v.spacing)} mm | "
                          f"orientação do arquivo={v.axcodes} | rótulo pCR={lc.label}")
        self.canvas.draw_idle()
