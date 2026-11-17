from typing import Dict, Sequence

import numpy as np


def _div(a, b):
    return float(a) / float(b) if b else 0.0


def binary_metrics(y_true: Sequence[int], y_prob: Sequence[float],
                   threshold: float = 0.5) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = _div(tp, tp + fn)
    spec = _div(tn, tn + fp)
    prec = _div(tp, tp + fp)
    acc = _div(tp + tn, len(y_true))
    f1 = _div(2 * prec * sens, prec + sens)
    return {"threshold": threshold, "n": int(len(y_true)),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensibilidade": sens, "especificidade": spec, "precisao": prec,
            "acuracia": acc, "f1": f1, "auc": auc_roc(y_true, y_prob)}


def auc_roc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    pos, neg = y_prob[y_true == 1], y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv), dtype=float)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_pos = ranks[np.argsort(order)][:len(pos)]
    u = rank_pos.sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def format_metrics(m: Dict) -> str:
    return (f"n={m['n']}  TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}\n"
            f"Sensibilidade: {m['sensibilidade']:.3f}   Especificidade: {m['especificidade']:.3f}\n"
            f"Precisão: {m['precisao']:.3f}   Acurácia: {m['acuracia']:.3f}   "
            f"F1: {m['f1']:.3f}   AUC: {m['auc']:.3f}")
