import queue
import threading
import tkinter as tk
from tkinter import messagebox

from pa_config import PHASES, PHASE_LABELS


class AppState:

    def __init__(self, root):
        self.root = root
        self.index = None
        self.case = None
        self.models = {}
        self.listeners = []

    def notify(self, what: str):
        for cb in self.listeners:
            try:
                cb(what)
            except Exception as e:
                print(f"[aviso] listener falhou ({what}): {e}")

    def need_case(self) -> bool:
        if self.case is None:
            messagebox.showinfo("Paciente", "Carregue um paciente na aba 'Visualizar' primeiro.")
            return False
        return True


def run_async(root, work, on_done, on_error=None, poll_ms=100):
    q = queue.Queue()

    def target():
        try:
            q.put(("ok", work()))
        except Exception as e:
            q.put(("err", e))

    threading.Thread(target=target, daemon=True).start()

    def poll():
        try:
            kind, val = q.get_nowait()
        except queue.Empty:
            root.after(poll_ms, poll)
            return
        if kind == "ok":
            on_done(val)
        elif on_error:
            on_error(val)
        else:
            messagebox.showerror("Erro", str(val))
    root.after(poll_ms, poll)


def phase_choices():
    return [f"{PHASE_LABELS[p]} ({p})" for p in PHASES]


def phase_from_choice(txt: str) -> str:
    for p in PHASES:
        if txt.endswith(f"({p})"):
            return p
    return PHASES[0]


def make_canvas(parent, figsize=(8, 4), toolbar=True):
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    fig = Figure(figsize=figsize, dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=parent)
    if toolbar:
        tb = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        tb.update()
        tb.pack(side="bottom", fill="x")
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
    return fig, canvas
