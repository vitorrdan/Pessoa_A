import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from pa_config import APP_TITLE
from pa_gui_augment import AugmentTab
from pa_gui_common import AppState
from pa_gui_folder import FolderTab
from pa_gui_gradcam import GradcamTab
from pa_gui_results import ResultsTab
from pa_gui_strategies import StrategiesTab
from pa_gui_viewer import ViewerTab


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(APP_TITLE)
        root.geometry("1180x780")
        self.state = AppState(root)

        bar = ttk.Frame(root)
        bar.pack(side="top", fill="x", padx=6, pady=4)
        ttk.Button(bar, text="Carregar pesos (.pt/.pth)…", command=self.load_weights).pack(side="left")
        ttk.Button(bar, text="Modelo DEMO (teste da interface)", command=self.load_demo).pack(side="left", padx=4)
        self.models_lbl = tk.StringVar(value="Modelos carregados: nenhum")
        ttk.Label(bar, textvariable=self.models_lbl).pack(side="left", padx=10)

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True)
        self.viewer = ViewerTab(nb, self.state)
        tabs = [(self.viewer, "Visualizar"), (AugmentTab(nb, self.state), "Aumento de dados"),
                (ResultsTab(nb, self.state), "Treino/Resultados"),
                (StrategiesTab(nb, self.state), "Estratégias de corte"),
                (GradcamTab(nb, self.state), "Classificar + Grad-CAM"),
                (FolderTab(nb, self.state), "Classificar pasta nova")]
        for frame, title in tabs:
            nb.add(frame, text=title)

    def _register(self, spec):
        self.state.models[spec.name] = spec
        self.models_lbl.set("Modelos carregados: " + ", ".join(self.state.models))
        self.state.notify("models")

    def load_weights(self):
        paths = filedialog.askopenfilenames(title="Pesos salvos",
                                            filetypes=[("PyTorch", "*.pt *.pth *.ckpt"), ("Todos", "*.*")])
        if not paths:
            return
        import pa_models
        import_errors = self._import_team_architectures()
        try:
            for p in paths:
                self._register(pa_models.load_model_spec(p))
        except Exception as e:
            msg = str(e)
            if import_errors:
                msg += "\n\nFalhas ao importar os modulos da equipe:\n" + "\n".join(import_errors)
            messagebox.showerror("Carregar pesos", msg)

    @staticmethod
    def _import_team_architectures():
        """INTEGRACAO: importar pc_models registra 'cnn', 'resnet50' e o modelo de ramos via
        pa_models.register_arch. No arquivo unico final isso ja estara no mesmo arquivo.
        Retorna a lista de falhas (modulo existe mas nao importou) para mostrar ao usuario;
        modulo inexistente e ignorado."""
        errors = []
        for mod in ("pc_models", "pb_models"):
            try:
                __import__(mod)
            except ModuleNotFoundError as e:
                if e.name != mod:
                    errors.append(f"{mod}: falta a biblioteca '{e.name}'")
            except Exception as e:
                errors.append(f"{mod}: {type(e).__name__}: {e}")
        return errors

    def load_demo(self):
        import pa_models
        self._register(pa_models.make_demo_spec("fusion"))
        self._register(pa_models.make_demo_spec("single:late", seed=1))


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
