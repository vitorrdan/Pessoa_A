# -*- coding: utf-8 -*-
"""
PESSOA A - Estrategias para combinar os cortes de um volume (item c).

Independente de torch: recebe `predict_fn(zs) -> probs` (ver pa_models.make_predict_fn).

Estrategias:
  central     : corte central (Z//2)
  max_area    : corte com maior area tumoral (exige mascara)
  mean        : media das probabilidades de TODOS os cortes
  max         : maximo das probabilidades de TODOS os cortes
  tumor_wmean : media ponderada pela area tumoral de cada corte (agregacao "outra");
                cortes sem tumor tem peso 0. Justificativa: cortes sem tumor nao
                carregam informacao sobre a resposta ao tratamento.
"""
from typing import Callable, Dict, Optional

import numpy as np

STRATEGIES = {
    "central": "Corte central",
    "max_area": "Corte de maior área tumoral",
    "mean": "Todos os cortes - média",
    "max": "Todos os cortes - máximo",
    "tumor_wmean": "Todos os cortes - média ponderada pela área tumoral",
}


def tumor_area_per_slice(mask: Optional[np.ndarray], n_slices: int) -> np.ndarray:
    if mask is None:
        return np.zeros(n_slices)
    return mask.sum(axis=(0, 1)).astype(float)


def central_slice(n_slices: int) -> int:
    return n_slices // 2


def max_area_slice(mask: Optional[np.ndarray], n_slices: int) -> int:
    areas = tumor_area_per_slice(mask, n_slices)
    return int(np.argmax(areas)) if areas.max() > 0 else central_slice(n_slices)


def predict_volume(predict_fn: Callable, n_slices: int, mask: Optional[np.ndarray],
                   strategy: str) -> Dict:
    """Retorna {'prob', 'slices' (usados), 'per_slice' (todos), 'areas', 'weights'}.
    Para estrategias de 1 corte so esse corte e inferido (rapido)."""
    areas = tumor_area_per_slice(mask, n_slices)
    all_z = list(range(n_slices))

    if strategy in ("central", "max_area"):
        if strategy == "max_area" and mask is None:
            raise ValueError("Estrategia 'max_area' exige a mascara de segmentacao.")
        z = central_slice(n_slices) if strategy == "central" else max_area_slice(mask, n_slices)
        p = float(predict_fn([z])[0])
        return {"prob": p, "slices": [z], "per_slice": None, "areas": areas, "weights": None}

    per = np.asarray(predict_fn(all_z), dtype=float)
    if strategy == "mean":
        prob, w = float(per.mean()), np.full(n_slices, 1.0 / n_slices)
    elif strategy == "max":
        prob, w = float(per.max()), None
    elif strategy == "tumor_wmean":
        if areas.sum() <= 0:
            raise ValueError("Estrategia 'tumor_wmean' exige a mascara de segmentacao.")
        w = areas / areas.sum()
        prob = float((per * w).sum())
    else:
        raise ValueError(f"Estrategia desconhecida: {strategy}")
    return {"prob": prob, "slices": all_z, "per_slice": per, "areas": areas, "weights": w}


def representative_slice(result: Dict, strategy: str, n_slices: int) -> int:
    """Corte a mostrar/explicar (Grad-CAM) para o resultado de uma estrategia."""
    if strategy in ("central", "max_area"):
        return result["slices"][0]
    per = result["per_slice"]
    if strategy == "max":
        return int(np.argmax(per))
    areas = result["areas"]
    return int(np.argmax(areas)) if areas.max() > 0 else int(np.argmax(per))
