# -*- coding: utf-8 -*-
"""
PESSOA A - "Classificar pasta nova" (conjunto-teste surpresa, sem retreinar).

Fluxo: indexa a pasta (mesma logica do dataset) -> carrega cada paciente ->
aplica a estrategia de cortes -> probabilidade de pCR -> metricas (se houver
rotulos no CSV da pasta). Nao treina nada: so usa os pesos salvos.
"""
import time
from typing import Callable, Dict, Optional

from pa_metrics import binary_metrics
from pa_nii_io import DatasetIndex, load_case
from pa_slices import predict_volume

NEEDS_MASK = ("max_area", "tumor_wmean")


def classify_folder(spec, folder: str, strategy: str = "mean", csv_path: Optional[str] = None,
                    threshold: Optional[float] = None, split: Optional[str] = None,
                    progress: Optional[Callable[[int, int, str], None]] = None) -> Dict:
    """Retorna {'rows': [dict], 'metrics': dict|None, 'seconds': float, 'index': DatasetIndex}.
    Se o caso nao tem mascara e a estrategia a exige, usa 'mean' (anotado em
    'strategy_used'). `split` restringe a um subconjunto (ex.: 'test')."""
    from pa_models import make_predict_fn
    thr = spec.threshold if threshold is None else threshold
    idx = DatasetIndex(folder, csv_path)
    pids = idx.pids(split)
    rows, t0 = [], time.time()
    for i, pid in enumerate(pids):
        if progress:
            progress(i, len(pids), pid)
        try:
            lc = load_case(idx.cases[pid])
            strat = strategy
            if strat in NEEDS_MASK and lc.mask is None:
                strat = "mean"
            res = predict_volume(make_predict_fn(spec, lc.vols), lc.n_slices, lc.mask, strat)
            rows.append({"pid": pid, "label": lc.label, "prob": res["prob"],
                         "pred": int(res["prob"] >= thr), "strategy_used": strat, "erro": ""})
        except Exception as e:                       # um paciente ruim nao derruba o lote
            rows.append({"pid": pid, "label": idx.cases[pid].label, "prob": float("nan"),
                         "pred": -1, "strategy_used": strategy, "erro": str(e)[:120]})
    if progress:
        progress(len(pids), len(pids), "fim")
    ok = [r for r in rows if r["label"] is not None and r["pred"] >= 0]
    metrics = binary_metrics([r["label"] for r in ok], [r["prob"] for r in ok], thr) if ok else None
    return {"rows": rows, "metrics": metrics, "seconds": time.time() - t0, "index": idx}


def save_rows_csv(rows, path):
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)
