import os
from typing import Dict, List, Optional

import numpy as np

from pa_models import ModelSpec, build_input, outputs_to_prob


class GradCAM:
    def __init__(self, spec: ModelSpec, target_layers: Optional[List] = None):
        self.spec = spec
        self.model = spec.model
        self.layers = list(target_layers or spec.target_layers)
        self._acts: List = []
        self._hooks = []


    def _attach(self):
        self._acts = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            def fwd(_m, _inp, out, i=i):
                self._acts[i] = out
            self._hooks.append(layer.register_forward_hook(fwd))

    def _detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._acts = []


    def __call__(self, x, target_class: int = 1) -> Dict:
        """x: tensor (N,C,H,W). Retorna {'cams': (N,H,W) em [0,1], 'prob': (N,),
        'per_layer': lista de (N,H,W)}."""
        import torch
        import torch.nn.functional as F
        self.model.eval()
        self._attach()
        try:
            with torch.enable_grad():
                x = x.clone().requires_grad_(True)
                out = self.model(x)
                flat = out.reshape(out.shape[0], -1)
                kind = self.spec.output_kind
                if kind == "logit2":
                    score = flat[:, target_class]
                else:
                    score = flat[:, 0] if target_class == 1 else -flat[:, 0]
                acts = list(self._acts)
                grads = torch.autograd.grad(score.sum(), acts, retain_graph=False)
            per_layer = []
            H, W = x.shape[-2:]
            for a, g in zip(acts, grads):
                w = g.mean(dim=(2, 3), keepdim=True)
                cam = F.relu((w * a).sum(dim=1, keepdim=True))
                cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
                cam = cam[:, 0].detach().cpu().numpy()
                per_layer.append(_norm01(cam))
            cams = _norm01(np.mean(per_layer, axis=0)) if len(per_layer) > 1 else per_layer[0]
            return {"cams": cams, "per_layer": per_layer,
                    "prob": outputs_to_prob(out, kind)}
        finally:
            self._detach()

    def explain_slices(self, vols: Dict[str, np.ndarray], zs: List[int],
                       target_class: int = 1) -> Dict:
        x = build_input(self.spec, vols, list(zs))
        return self(x, target_class)


def _norm01(cam: np.ndarray) -> np.ndarray:
    """Normaliza cada mapa do lote para [0,1]; mapa todo zero permanece zero."""
    cam = np.asarray(cam, dtype=np.float32)
    mn = cam.min(axis=(-2, -1), keepdims=True)
    mx = cam.max(axis=(-2, -1), keepdims=True)
    return (cam - mn) / np.maximum(mx - mn, 1e-8) * (mx > mn)


def resize_map(cam: np.ndarray, shape) -> np.ndarray:
    """Reamostra (h,w)->shape (ex.: se input_size != tamanho do corte)."""
    if cam.shape == tuple(shape):
        return cam
    from scipy import ndimage
    return ndimage.zoom(cam, (shape[0] / cam.shape[0], shape[1] / cam.shape[1]), order=1)





def heat_in_mask_metrics(cam: np.ndarray, mask: np.ndarray, top_frac: float = 0.10) -> Dict:
    """cam, mask: (H,W). Retorna:
      heat_fraction : sum(cam dentro da mascara) / sum(cam)   (o pedido no plano)
      area_fraction : area da mascara / area da imagem        (baseline do acaso)
      enrichment    : heat_fraction / area_fraction  (>1 = calor concentrado no tumor)
      top_in_mask   : fracao dos `top_frac` pixels mais quentes (com calor>0) que caem no tumor
      peak_in_mask  : o pixel de maior calor esta no tumor? (pointing game)
    NaN quando a mascara esta vazia no corte ou o mapa e todo zero."""
    cam = np.asarray(cam, dtype=float)
    m = np.asarray(mask) > 0.5
    if m.shape != cam.shape:
        cam = resize_map(cam, m.shape)
    total = cam.sum()
    area = m.mean()
    nan = float("nan")
    if m.sum() == 0 or total <= 0:
        return {"heat_fraction": nan, "area_fraction": float(area), "enrichment": nan,
                "top_in_mask": nan, "peak_in_mask": nan}
    hf = float(cam[m].sum() / total)


    k = max(1, min(int(round(top_frac * cam.size)), int((cam > 0).sum())))
    idx = np.argpartition(cam.ravel(), -k)[-k:]
    top_in = float(m.ravel()[idx].mean())
    peak = bool(m.ravel()[int(np.argmax(cam))])
    return {"heat_fraction": hf, "area_fraction": float(area), "enrichment": hf / area,
            "top_in_mask": top_in, "peak_in_mask": float(peak)}


def summarize_heat_metrics(rows: List[Dict]) -> Dict:
    """Media dos indicadores sobre varios casos (ignora NaN)."""
    out = {}
    for k in ("heat_fraction", "area_fraction", "enrichment", "top_in_mask", "peak_in_mask"):
        v = np.array([r[k] for r in rows if r.get(k) is not None], dtype=float)
        v = v[~np.isnan(v)]
        out[k] = float(v.mean()) if len(v) else float("nan")
        out[k + "_n"] = int(len(v))
    return out





def overlay_figure(base, cam, mask, title="", orient=("T_lower", False), alpha=0.45, ax=None):
    """Desenha corte + calor + contorno da mascara. Retorna o eixo."""
    import matplotlib.pyplot as plt
    from pa_nii_io import orient_like
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    o, fl = orient
    cam = resize_map(cam, base.shape)
    b = orient_like(base, o, fl)
    origin = "lower" if o == "T_lower" else "upper"
    ax.imshow(b, cmap="gray", origin=origin)
    ax.imshow(orient_like(cam, o, fl), cmap="jet", alpha=alpha, origin=origin, vmin=0, vmax=1)
    if mask is not None and mask.sum() > 0:
        ax.contour(orient_like(mask, o, fl), levels=[0.5], colors="white", linewidths=1.0)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    return ax


def generate_report_figures(spec: ModelSpec, cases, out_dir: str, strategy: str = "max_area",
                            per_group: int = 3, threshold: Optional[float] = None) -> str:
    """Gera PNGs de Grad-CAM de acertos e erros (TP/TN/FP/FN) + CSV com as
    metricas de coerencia calor-tumor. `cases`: lista de LoadedCase com rotulo.
    Retorna o caminho do CSV."""
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    from pa_models import make_predict_fn
    from pa_slices import predict_volume, representative_slice

    os.makedirs(out_dir, exist_ok=True)
    thr = spec.threshold if threshold is None else threshold
    gc = GradCAM(spec)
    rows, groups = [], {"TP": [], "TN": [], "FP": [], "FN": []}

    for lc in cases:
        if lc.label is None:
            continue
        pf = make_predict_fn(spec, lc.vols)
        res = predict_volume(pf, lc.n_slices, lc.mask, strategy)
        z = representative_slice(res, strategy, lc.n_slices)
        pred = int(res["prob"] >= thr)
        g = ("TP" if pred and lc.label else "TN" if not pred and not lc.label
             else "FP" if pred else "FN")
        cam = gc.explain_slices(lc.vols, [z], target_class=1)["cams"][0]
        mk = lc.mask[:, :, z] if lc.mask is not None else None
        hm = heat_in_mask_metrics(cam, mk) if mk is not None else {}
        rows.append({"pid": lc.pid, "label": lc.label, "prob": res["prob"], "pred": pred,
                     "grupo": g, "z": z, **hm})
        groups[g].append((abs(res["prob"] - thr), lc, z, cam, mk, res["prob"]))

    for g, items in groups.items():
        items.sort(key=lambda t: -t[0])
        items = items[:per_group]
        if not items:
            continue
        fig, axes = plt.subplots(1, len(items), figsize=(3.4 * len(items), 3.6), squeeze=False)
        for ax, (_c, lc, z, cam, mk, p) in zip(axes[0], items):
            base = lc.vols[spec.phases_used[-1]][:, :, z]
            overlay_figure(base, cam, mk, f"{g} | {lc.pid} z={z}\nP(pCR)={p:.2f} real={lc.label}", ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"gradcam_{g}.png"), dpi=150)
        plt.close(fig)

    csv = os.path.join(out_dir, "gradcam_coerencia.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv
