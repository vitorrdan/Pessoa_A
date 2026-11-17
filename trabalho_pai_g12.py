# """
# Processamento e Analise de Imagens - Trabalho Pratico
# 
# Integrantes:
# - Daniel Victor Rocha Costa | Matricula: 813213 | Ciencia da Computacao | Campus: Coracao Eucaristico
# - Gabriela [Sobrenome]      | Matricula: XXXXXXX | Ciencia da Computacao | Campus: Coracao Eucaristico
# - Suzane Lemos de Lima        | Matricula: 770855 | Ciencia da Computacao | Campus: Coracao Eucaristico
# """
import math
from typing import Dict, List, Sequence, Tuple
import numpy as np
from scipy import ndimage
from typing import Dict, Sequence
import inspect
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import os
import re
from typing import Dict, List, Optional
from typing import Callable, Dict, Optional
import time
import queue
import threading
import tkinter as tk
from tkinter import messagebox
from tkinter import messagebox, ttk
from tkinter import filedialog, messagebox, ttk
import json
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import torch
import torch.nn as nn
import argparse
import copy
import random
from dataclasses import asdict, dataclass, field
import pandas as pd
import torch.nn.functional as F
                                                                                                   
PHASES = ["pre", "early", "late"]
PHASE_LABELS = {
    "pre": "Pré-contraste",
    "early": "Pós-contraste precoce",
    "late": "Pós-contraste tardio",
}


PHASE_PATTERNS = {
    "pre":   ["_aqc_0", "_0000", "_pre", "pre_", "phase0", "ph0", "_t0"],
    "early": ["_aqc_1", "_0001", "_early", "_post1", "phase1", "ph1", "_t1"],
    "late":  ["_aqc_2", "_0002", "_late", "_post2", "phase2", "ph2", "_t2"],
}

MASK_PATTERNS = ["seg", "mask", "label", "tumor"]


CSV_ID_COLS = ["patient_id", "pid", "id", "case", "case_id", "subject", "patient"]
CSV_LABEL_COLS = ["pcr", "pcr_label", "label", "class", "target", "y"]
CSV_SPLIT_COLS = ["split", "partition", "subset", "set", "test", "fold_set"]

CSV_DATASET_COL = "dataset"
CSV_DATASET_VALUE = "spy1"


SPLIT_NUMERIC_COLS = {"test": {0: "train", 1: "test", 2: "val"}}

CSV_PHASE_COLS = {"pre": "pre", "early": "post_early", "late": "post_late"}
CSV_MASK_COLS = ["mask", "seg", "segmentation", "mask_file", "mask_path"]


EXTRA_CSV_NAMES = ("splits.csv", "split_x_classe.csv")


SPLITS_CSV_NAME = "splits.csv"

MASK_DIR_NAMES = ["mask", "masks", "seg", "segmentation", "label", "labels"]


Z_AXIS = None

SPLIT_ALIASES = {
    "train": ["train", "training", "treino", "tr"],
    "val":   ["val", "valid", "validation", "validacao"],
    "test":  ["test", "testing", "teste", "ts"],
}


AUG_ANGLES = [-20, -10, 0, 10, 20]


DEFAULT_ORIENTATION = "T_lower"
DEFAULT_FLIP_LR = False


INPUT_SIZE = 256
DEFAULT_THRESHOLD = 0.5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NORM_P_LOW, NORM_P_HIGH = 1.0, 99.0


APP_TITLE = "DCE-MRI | Resposta ao tratamento (pCR) - Interface e Grad-CAM"

                                                                               
                      
                                                                               

def rotate_img(img: np.ndarray, angle: float) -> np.ndarray:
                                                                 
    if angle == 0:
        return img.copy()
    return ndimage.rotate(img, angle, axes=(0, 1), reshape=False, order=1,
                          mode="constant", cval=0.0)


def augment_variants(img: np.ndarray, angles: Sequence[float] = AUG_ANGLES
                     ) -> List[Tuple[str, np.ndarray]]:
                                                                            
                                                                               
                                                                                
    out = []
    for mirrored in (False, True):
        base = img[:, ::-1].copy() if mirrored else img
        for a in angles:
            tag = f"{a:+d}°" + (" espelhada" if mirrored else " original")
            out.append((tag, rotate_img(base, a)))
    return out


def augment_mask_like(mask: np.ndarray, angle: float, mirrored: bool) -> np.ndarray:
                                                                                     
    m = mask[:, ::-1] if mirrored else mask
    if angle == 0:
        return m.copy()
    return ndimage.rotate(m, angle, axes=(0, 1), reshape=False, order=0,
                          mode="constant", cval=0.0)


def balance_plan(counts: Dict[int, int], n_variants: int = 10,
                 relevant_ratio: float = 1.5) -> Dict:
                                                           

                                                                       
                           
                                       
                                                                               
                                         
                                                                                
                                                                                          
                                                                                      
                                                                             
                                                            
       
    n0, n1 = int(counts.get(0, 0)), int(counts.get(1, 0))
    if n0 == 0 or n1 == 0:
        raise ValueError("Uma das classes nao tem exemplos - verifique o CSV.")
    maj_cls = 0 if n0 >= n1 else 1
    n_maj, n_min = max(n0, n1), min(n0, n1)
    ratio = n_maj / n_min

    v_maj = n_variants
    v_min = int(math.ceil(n_variants * ratio))
    var = {maj_cls: v_maj, 1 - maj_cls: v_min}
    total = n0 + n1
    return {
        "ratio": ratio,
        "relevant": ratio >= relevant_ratio,
        "majority_class": maj_cls,
        "before": {0: n0, 1: n1},
        "after_aug": {0: n0 * n_variants, 1: n1 * n_variants},
        "after_prop": {0: n0 * var[0], 1: n1 * var[1]},
        "variants_per_class": var,
        "class_weights": {0: total / (2 * n0), 1: total / (2 * n1)},
        "pos_weight": n0 / n1,
    }


def extra_angles_for_minority(n_extra: int) -> List[float]:
                                                                      
                                                                              
                                               
    pool = [-15, -5, 5, 15, -17.5, -12.5, -7.5, -2.5, 2.5, 7.5, 12.5, 17.5]
    return pool[:max(0, n_extra)]

                                                                               
                      
                                                                               

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

                                                                               
                     
                                                                               

ARCH_BUILDERS: Dict[str, Callable] = {}


def register_arch(name: str, builder: Callable):
                                                             
    ARCH_BUILDERS[name] = builder


@dataclass
class ModelSpec:
    name: str
    model: object
    mode: str = "fusion"
    in_channels: int = 3
    input_size: int = INPUT_SIZE
    output_kind: str = "logit"
    imagenet_norm: bool = False
    threshold: float = DEFAULT_THRESHOLD
    target_layers: List = field(default_factory=list)
    device: str = "cpu"
    norm: str = "percentile"

    @property
    def phases_used(self) -> List[str]:
        if self.mode == "fusion":
            return list(PHASES)
        return [self.mode.split(":", 1)[1]]


def find_target_layers(model) -> List:
    import torch.nn as nn
    layers = getattr(model, "gradcam_layers", None)
    if layers:
        return list(layers)
    if hasattr(model, "layer4"):
        return [model.layer4]
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise ValueError("Nenhuma camada convolucional encontrada para o Grad-CAM.")
    return [last]


def _filter_kwargs(fn, cfg: dict) -> dict:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(cfg)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(cfg)
    return {k: v for k, v in cfg.items() if k in sig.parameters}


def get_device(prefer_gpu=True) -> str:
    import torch
    return "cuda" if prefer_gpu and torch.cuda.is_available() else "cpu"


def load_model_spec(path: str, name: Optional[str] = None, device: Optional[str] = None
                    ) -> ModelSpec:
    import torch
    device = device or get_device()
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        arch = ckpt.get("arch")
        if arch not in ARCH_BUILDERS:
            raise KeyError(f"Arquitetura '{arch}' nao registrada. Chame "
                           f"pa_models.register_arch('{arch}', builder) antes de carregar. "
                           f"Registradas: {list(ARCH_BUILDERS)}")
        cfg = {k: v for k, v in ckpt.items() if k != "state_dict"}
        builder = ARCH_BUILDERS[arch]
        model = builder(**_filter_kwargs(builder, cfg))
        model.load_state_dict(ckpt["state_dict"])
    else:
        model, cfg = ckpt, {}
    model.to(device).eval()
    norm = str(cfg.get("norm", "percentile"))
    if "percentil" not in norm.lower():
        warnings.warn(f"Checkpoint treinado com normalizacao '{norm}', mas a interface aplica "
                      f"percentil 1-99: as previsoes podem divergir do treino.")
    return ModelSpec(
        name=name or cfg.get("arch") or type(model).__name__,
        model=model,
        mode=cfg.get("mode", "fusion"),
        in_channels=int(cfg.get("in_channels", 3)),
        input_size=int(cfg.get("input_size", INPUT_SIZE)),
        output_kind=cfg.get("output_kind", "logit"),
        imagenet_norm=bool(cfg.get("imagenet_norm", False)),
        threshold=float(cfg.get("threshold", DEFAULT_THRESHOLD)),
        target_layers=find_target_layers(model),
        device=device,
        norm=norm,
    )


def build_input(spec: ModelSpec, vols: Dict[str, np.ndarray], zs: List[int]):
    import torch
    import torch.nn.functional as F
    chans = []
    for ph in spec.phases_used:
        chans.append(np.stack([vols[ph][:, :, z] for z in zs], axis=0))
    x = np.stack(chans, axis=1).astype(np.float32)
    if x.shape[1] == 1 and spec.in_channels == 3:
        x = np.repeat(x, 3, axis=1)
    elif x.shape[1] != spec.in_channels:
        raise ValueError(f"Modelo espera {spec.in_channels} canais, entrada tem {x.shape[1]}")
    t = torch.from_numpy(x)
    if t.shape[-1] != spec.input_size or t.shape[-2] != spec.input_size:
        t = F.interpolate(t, size=(spec.input_size, spec.input_size),
                          mode="bilinear", align_corners=False)
    if spec.imagenet_norm:
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        t = (t - mean) / std
    return t.to(spec.device)


def outputs_to_prob(out, output_kind: str):
    import torch
    out = out.detach().float().reshape(out.shape[0], -1)
    if output_kind == "prob":
        p = out[:, 0]
    elif output_kind == "logit2":
        p = torch.softmax(out, dim=1)[:, 1]
    else:
        p = torch.sigmoid(out[:, 0])
    return p.cpu().numpy()


def make_predict_fn(spec: ModelSpec, vols: Dict[str, np.ndarray], batch_size: int = 16):
    import torch

    def predict(zs: List[int]) -> np.ndarray:
        probs = []
        with torch.no_grad():
            for i in range(0, len(zs), batch_size):
                x = build_input(spec, vols, list(zs[i:i + batch_size]))
                probs.append(outputs_to_prob(spec.model(x), spec.output_kind))
        return np.concatenate(probs) if probs else np.zeros(0)
    return predict


def make_demo_spec(mode="fusion", seed=0, device="cpu") -> ModelSpec:
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    cin = 3 if mode == "fusion" else 1

    class Demo(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv2d(cin, 8, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(4),
                nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(4))
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 1))

        def forward(self, x):
            return self.head(self.f(x))

    m = Demo().to(device).eval()
    return ModelSpec(name=f"DEMO-{mode}", model=m, mode=mode, in_channels=cin,
                     target_layers=find_target_layers(m), device=device)

                                                                               
                     
                                                                               

@dataclass
class Volume:
    data: np.ndarray
    spacing: tuple = (1.0, 1.0, 1.0)
    axcodes: str = "?"
    path: str = ""
    z_axis_original: int = 2
    shape_original: tuple = ()

    @property
    def shape(self):
        return self.data.shape

    @property
    def n_slices(self):
        return self.data.shape[2]


def detect_z_axis(shape, forced=None) -> int:
    if forced is not None:
        return int(forced)
    for a in range(3):
        o = [shape[b] for b in range(3) if b != a]
        if o[0] == o[1] and shape[a] != o[0]:
            return a
    return int(np.argmin(shape))


def load_nii(path: str, z_axis=None) -> Volume:
    import nibabel as nib
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    data = np.squeeze(data)
    if data.ndim == 2:
        data = data[:, :, None]
    if data.ndim != 3:
        raise ValueError(f"Volume com dimensoes inesperadas {data.shape} em {path}")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    try:
        axcodes = "".join(nib.aff2axcodes(img.affine))
    except Exception:
        axcodes = "?"
    shape0 = tuple(data.shape)
    za = detect_z_axis(shape0, Z_AXIS if z_axis is None else z_axis)
    perm = [a for a in range(3) if a != za] + [za]
    if perm != [0, 1, 2]:
        data = np.ascontiguousarray(np.transpose(data, perm))
        zooms = tuple(zooms[i] for i in perm) if len(zooms) == 3 else zooms
        if len(axcodes) == 3:
            axcodes = "".join(axcodes[i] for i in perm)
    return Volume(data=data, spacing=zooms, axcodes=axcodes, path=path,
                  z_axis_original=za, shape_original=shape0)


def load_mask(path: str) -> Volume:
    vol = load_nii(path)
    vol.data = (vol.data > 0.5).astype(np.float32)
    return vol


def normalize_volume(data: np.ndarray, p_low=NORM_P_LOW, p_high=NORM_P_HIGH) -> np.ndarray:
    lo, hi = np.percentile(data, [p_low, p_high])
    if hi - lo < 1e-8:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def orient_slice(sl: np.ndarray, orientation: str = "T_lower", flip_lr: bool = False):
    if orientation == "T_lower":
        out, origin = sl.T, "lower"
    elif orientation == "rot90":
        out, origin = np.rot90(sl), "upper"
    else:
        out, origin = sl, "upper"
    if flip_lr:
        out = out[:, ::-1]
    return out, origin


def orient_like(sl, orientation, flip_lr):
    return orient_slice(sl, orientation, flip_lr)[0]


@dataclass
class Case:
    pid: str
    label: Optional[int] = None
    split: Optional[str] = None
    phases: Dict[str, str] = field(default_factory=dict)
    mask: Optional[str] = None
    clinical: dict = field(default_factory=dict)

    def mask_range(self):
        try:
            return int(self.clinical["mask_start"]), int(self.clinical["mask_end"])
        except (KeyError, TypeError, ValueError):
            return None

    def complete(self, need_mask=False) -> bool:
        ok = all(p in self.phases for p in PHASES)
        return ok and (self.mask is not None or not need_mask)


def _pick_col(columns, candidates):
    low = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def _norm_split(v, colname=None):
    if colname is not None and colname.lower() in SPLIT_NUMERIC_COLS:
        try:
            return SPLIT_NUMERIC_COLS[colname.lower()][int(float(v))]
        except (ValueError, KeyError):
            pass
    s = str(v).strip().lower()
    for key, aliases in SPLIT_ALIASES.items():
        if s in aliases:
            return key
    return s


def _classify_file(path: str, root: Optional[str] = None):
    low = os.path.basename(path).lower()
    parts = []
    if root:
        rel = os.path.relpath(os.path.dirname(path), root)
        parts = [p.lower() for p in rel.split(os.sep) if p not in (".", "")]
    if any(p in low for p in MASK_PATTERNS) or any(p in MASK_DIR_NAMES for p in parts):
        return "mask", None
    for phase in PHASES:
        if any(p in low for p in PHASE_PATTERNS[phase]):
            return "phase", phase
    return None, None


def scan_nii_files(root: str) -> List[str]:
    out = []
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith((".nii", ".nii.gz")):
                out.append(os.path.join(dp, fn))
    return sorted(out)


class DatasetIndex:

    def __init__(self, root: str, csv_path: Optional[str] = None,
                 splits_csv: Optional[str] = None):
        self.root = root
        self.cases: Dict[str, Case] = {}
        self.csv_info = {}
        self._build(csv_path)
        sp = splits_csv or self._find_splits_csv()
        if sp:
            self._apply_splits(sp)


    def _build(self, csv_path):
        files = scan_nii_files(self.root)
        csv_path = csv_path or self._find_csv()
        table = self._read_csv(csv_path) if csv_path else {}
        pids_csv = list(table.keys())

        for f in files:
            kind, phase = _classify_file(f, self.root)
            if kind is None:
                continue
            pid = self._match_pid(f, pids_csv)
            case = self.cases.setdefault(pid, Case(pid=pid))
            if kind == "mask":
                case.mask = f
            else:
                case.phases[phase] = f


        by_key = {}
        for f in files:
            by_key.setdefault(_file_key(f), []).append(f)
        for pid, row in table.items():
            refs = row.get("refs")
            if refs:
                c = self.cases.setdefault(pid, Case(pid=pid))
                for ph, ref in refs.items():
                    p = _resolve_ref(ref, self.root, by_key)
                    if p:
                        c.phases[ph] = p
                mp = _resolve_ref(row.get("mask_ref"), self.root, by_key)
                if mp:
                    c.mask = mp
            if pid in self.cases:
                c = self.cases[pid]
                c.label, c.split, c.clinical = row["label"], row["split"], row["clinical"]

    def _list_csvs(self):
        out = []
        for dp, _dn, fns in os.walk(self.root):
            for fn in sorted(fns):
                if fn.lower().endswith(".csv"):
                    out.append(os.path.join(dp, fn))
        return out

    def _find_csv(self):
        import pandas as pd
        best, best_score = None, (-1, -1)
        for p in self._list_csvs():
            if os.path.basename(p).lower() in EXTRA_CSV_NAMES:
                continue
            try:
                cols = list(pd.read_csv(p, nrows=0).columns)
            except Exception:
                continue
            ok = _pick_col(cols, CSV_ID_COLS) is not None and _pick_col(cols, CSV_LABEL_COLS) is not None
            score = (int(ok), len(cols))
            if score > best_score:
                best, best_score = p, score
        return best

    def _find_splits_csv(self):
        for p in self._list_csvs():
            if os.path.basename(p).lower() == SPLITS_CSV_NAME.lower():
                return p
        return None

    def _apply_splits(self, path):
        import pandas as pd
        df = pd.read_csv(path)
        idc = _pick_col(df.columns, CSV_ID_COLS)
        spc = _pick_col(df.columns, ["split"] + CSV_SPLIT_COLS)
        lbc = _pick_col(df.columns, CSV_LABEL_COLS)
        if idc is None or spc is None:
            return
        n = 0
        for _, r in df.iterrows():
            c = self.cases.get(str(r[idc]).strip())
            if c is None or _isnan(r[spc]) or c.split is not None:
                continue
            c.split = _norm_split(r[spc], spc)
            n += 1
            if c.label is None and lbc is not None and not _isnan(r[lbc]):
                c.label = int(float(r[lbc]))
        self.csv_info["splits_csv"] = path
        self.csv_info["splits_applied"] = n

    def _read_csv(self, path):
        import pandas as pd
        df = pd.read_csv(path)
        n_total = len(df)
        dsc = _pick_col(df.columns, [CSV_DATASET_COL]) if CSV_DATASET_COL else None
        if dsc is not None and CSV_DATASET_VALUE:
            keep = df[dsc].astype(str).str.strip().str.lower() == str(CSV_DATASET_VALUE).lower()
            if keep.any():
                df = df[keep]
        idc = _pick_col(df.columns, CSV_ID_COLS)
        lbc = _pick_col(df.columns, CSV_LABEL_COLS)
        spc = _pick_col(df.columns, CSV_SPLIT_COLS)
        phase_cols = {ph: _pick_col(df.columns, [col]) for ph, col in CSV_PHASE_COLS.items()}
        phase_cols = phase_cols if all(phase_cols.values()) else {}
        mkc = _pick_col(df.columns, CSV_MASK_COLS)
        self.csv_info = {"path": path, "id": idc, "label": lbc, "split": spc,
                         "dataset_filter": f"{dsc}=={CSV_DATASET_VALUE}" if dsc else None,
                         "rows_total": n_total, "rows_used": len(df),
                         "phase_cols": phase_cols or None, "mask_col": mkc,
                         "columns": list(df.columns)}
        if idc is None:
            return {}
        skip = {idc, lbc, spc, mkc, *phase_cols.values()}
        table = {}
        for _, r in df.iterrows():
            pid = str(r[idc]).strip()
            label = None
            if lbc is not None and not _isnan(r[lbc]):
                label = int(float(r[lbc]))
            split = _norm_split(r[spc], spc) if spc is not None and not _isnan(r[spc]) else None
            table[pid] = {"label": label, "split": split,
                          "refs": {ph: r[col] for ph, col in phase_cols.items()},
                          "mask_ref": r[mkc] if mkc is not None else None,
                          "clinical": {k: r[k] for k in df.columns if k not in skip}}
        return table

    @staticmethod
    def _match_pid(filepath, pids_csv):
        low = filepath.lower()
        best = None
        for pid in pids_csv:
            if pid.lower() in low and (best is None or len(pid) > len(best)):
                best = pid
        if best:
            return best
        name = os.path.basename(filepath)
        name = re.sub(r"\.nii(\.gz)?$", "", name, flags=re.I)
        for pats in list(PHASE_PATTERNS.values()) + [MASK_PATTERNS]:
            for p in pats:
                name = re.sub(re.escape(p) + r".*$", "", name, flags=re.I)
        return name.strip("_-. ") or os.path.basename(os.path.dirname(filepath))


    def pids(self, split: Optional[str] = None, complete_only=True) -> List[str]:
        out = []
        for pid, c in sorted(self.cases.items()):
            if complete_only and not c.complete():
                continue
            if split and c.split != split:
                continue
            out.append(pid)
        return out

    def class_counts(self, split: Optional[str] = None) -> Dict[int, int]:
        counts = {0: 0, 1: 0}
        for pid in self.pids(split, complete_only=False):
            lab = self.cases[pid].label
            if lab in counts:
                counts[lab] += 1
        return counts

    def summary(self) -> str:
        n = len(self.cases)
        comp = len(self.pids())
        splits = {}
        for c in self.cases.values():
            splits[c.split] = splits.get(c.split, 0) + 1
        return (f"{n} pacientes encontrados, {comp} com 3 fases completas | "
                f"splits: {splits} | csv: {self.csv_info.get('path', 'nenhum')} "
                f"splits.csv: {self.csv_info.get('splits_csv', 'nenhum')} | "
                f"(linhas usadas {self.csv_info.get('rows_used', '?')}/"
                f"{self.csv_info.get('rows_total', '?')}, filtro {self.csv_info.get('dataset_filter')})")


def _file_key(path: str) -> str:
    return re.sub(r"\.nii(\.gz)?$", "", os.path.basename(str(path)).strip(), flags=re.I).lower()


def _resolve_ref(ref, root, by_key):
    if ref is None or _isnan(ref):
        return None
    s = str(ref).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    for cand in (s, os.path.join(root, s)):
        if os.path.isfile(cand):
            return cand
    key = _file_key(s)
    if key in by_key:
        return sorted(by_key[key])[0]
    hits = [p for k, v in by_key.items() if key in k for p in v]
    return hits[0] if len(hits) == 1 else None


def _isnan(v):
    try:
        return v is None or (isinstance(v, float) and np.isnan(v))
    except Exception:
        return False


@dataclass
class LoadedCase:
    pid: str
    label: Optional[int]
    vols: Dict[str, np.ndarray]
    raw: Dict[str, np.ndarray]
    mask: Optional[np.ndarray]
    info: Dict[str, Volume]

    @property
    def n_slices(self):
        return next(iter(self.vols.values())).shape[2]


def load_case(case: Case) -> LoadedCase:
    vols, raw, info = {}, {}, {}
    for ph, p in case.phases.items():
        v = load_nii(p)
        info[ph] = v
        raw[ph] = v.data
        vols[ph] = normalize_volume(v.data)
    mask = None
    if case.mask:
        mask = load_mask(case.mask).data
    return LoadedCase(case.pid, case.label, vols, raw, mask, info)

                                                                               
                     
                                                                               

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
    if strategy in ("central", "max_area"):
        return result["slices"][0]
    per = result["per_slice"]
    if strategy == "max":
        return int(np.argmax(per))
    areas = result["areas"]
    return int(np.argmax(areas)) if areas.max() > 0 else int(np.argmax(per))

                                                                               
                          
                                                                               

NEEDS_MASK = ("max_area", "tumor_wmean")


def classify_folder(spec, folder: str, strategy: str = "mean", csv_path: Optional[str] = None,
                    threshold: Optional[float] = None, split: Optional[str] = None,
                    progress: Optional[Callable[[int, int, str], None]] = None) -> Dict:
                                                                                               
                                                                            
                                                                          
    pass                                    
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
        except Exception as e:
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
                                                                                
    cam = np.asarray(cam, dtype=np.float32)
    mn = cam.min(axis=(-2, -1), keepdims=True)
    mx = cam.max(axis=(-2, -1), keepdims=True)
    return (cam - mn) / np.maximum(mx - mn, 1e-8) * (mx > mn)


def resize_map(cam: np.ndarray, shape) -> np.ndarray:
                                                                          
    if cam.shape == tuple(shape):
        return cam
    from scipy import ndimage
    return ndimage.zoom(cam, (shape[0] / cam.shape[0], shape[1] / cam.shape[1]), order=1)


def heat_in_mask_metrics(cam: np.ndarray, mask: np.ndarray, top_frac: float = 0.10) -> Dict:
                                 
                                                                                 
                                                                                 
                                                                                      
                                                                                               
                                                                           
                                                                      
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
                                                                
    out = {}
    for k in ("heat_fraction", "area_fraction", "enrichment", "top_in_mask", "peak_in_mask"):
        v = np.array([r[k] for r in rows if r.get(k) is not None], dtype=float)
        v = v[~np.isnan(v)]
        out[k] = float(v.mean()) if len(v) else float("nan")
        out[k + "_n"] = int(len(v))
    return out


def overlay_figure(base, cam, mask, title="", orient=("T_lower", False), alpha=0.45, ax=None):
                                                                      
    import matplotlib.pyplot as plt
    pass                                    
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
                                                                          
                                                                               
                                
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    pass                                    
    pass                                    

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


    def _use_max_area(self):
        lc = self.state.case
        if not self.state.need_case():
            return
        if lc.mask is None:
            messagebox.showinfo("Máscara", "Este paciente não tem máscara carregada.")
            return
        pass                                    
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

                                                                               
                           
                                                                               

COLS = ["name", "sensibilidade", "especificidade", "precisao", "acuracia", "f1", "auc", "train_time_s"]


ALIASES = {"sensibilidade": ["sensitivity", "sens", "recall"],
           "especificidade": ["specificity", "spec"],
           "precisao": ["precision"], "acuracia": ["accuracy", "acc"],
           "f1": ["f1_score"], "auc": ["roc_auc", "auc_roc"]}


def get_metric(m: dict, key: str):
    for prefix in ("", "m05_", "mt_"):
        for name in [key] + ALIASES.get(key, []):
            if prefix + name in m:
                return m[prefix + name]
    return None


class ResultsTab(ttk.Frame):
    def __init__(self, parent, state):
        super().__init__(parent)
        self.state = state
        self.runs = []
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", padx=6, pady=4)
        ttk.Button(top, text="Carregar resultados (.json)…", command=self.load).pack(side="left")
        ttk.Button(top, text="Limpar", command=self.clear).pack(side="left", padx=4)
        ttk.Button(top, text="Exportar tabela CSV…", command=self.export).pack(side="left")
        self.tree = ttk.Treeview(self, columns=COLS, show="headings", height=7)
        for c in COLS:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=150 if c == "name" else 95, anchor="center")
        self.tree.pack(side="top", fill="x", padx=6)
        self.fig, self.canvas = make_canvas(self, (9, 3.6))

    def load(self):
        for p in filedialog.askopenfilenames(title="Resultados", filetypes=[("JSON", "*.json")]):
            try:
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                d.setdefault("name", p.split("/")[-1])
                self.runs.append(d)
            except Exception as e:
                messagebox.showerror("Resultados", f"{p}\n{e}")
        self.redraw()

    def clear(self):
        self.runs = []
        self.redraw()

    def redraw(self):
        self.tree.delete(*self.tree.get_children())
        for r in self.runs:
            m = r.get("metrics") or r
            vals = [r.get("name", "?")] + [self._fmt(get_metric(m, c)) for c in COLS[1:7]] + \
                   [self._fmt(r.get("train_time_s", r.get("train_time")), 0)]
            self.tree.insert("", "end", values=vals)
        self.fig.clear()
        ax = self.fig.subplots()
        for r in self.runs:
            if r.get("train_loss"):
                ln, = ax.plot(r["train_loss"], label=f"{r['name']} (treino)")
                if r.get("val_loss"):
                    ax.plot(r["val_loss"], "--", color=ln.get_color(), label=f"{r['name']} (val)")
        ax.set_xlabel("época")
        ax.set_ylabel("loss")
        ax.set_title("Convergência da loss")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    @staticmethod
    def _fmt(v, nd=3):
        return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v))

    def export(self):
        if not self.runs:
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not p:
            return
        import pandas as pd
        rows = [{"name": r.get("name"), **{c: get_metric(r.get("metrics") or r, c) for c in COLS[1:7]},
                 "train_time_s": r.get("train_time_s", r.get("train_time"))} for r in self.runs]
        pd.DataFrame(rows).to_csv(p, index=False)

                                                                               
                              
                                                                               

class StrategiesTab(ttk.Frame):
    def __init__(self, parent, state):
        super().__init__(parent)
        self.state = state
        state.listeners.append(self._on_state)
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", padx=6, pady=4)
        ttk.Label(top, text="Modelo:").pack(side="left")
        self.model_var = tk.StringVar()
        self.model_cb = ttk.Combobox(top, textvariable=self.model_var, width=34, state="readonly")
        self.model_cb.pack(side="left", padx=4)
        ttk.Button(top, text="Comparar estratégias no paciente atual", command=self.run
                   ).pack(side="left", padx=6)
        self.status = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.status).pack(side="left")
        cols = ["estrategia", "P(pCR)", "classe prevista", "cortes usados"]
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=5)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=260 if c == "estrategia" else 120, anchor="center")
        self.tree.pack(side="top", fill="x", padx=6)
        self.fig, self.canvas = make_canvas(self, (9, 3.8))

    def _on_state(self, what):
        if what == "models":
            names = list(self.state.models)
            self.model_cb["values"] = names
            if names and not self.model_var.get():
                self.model_var.set(names[0])

    def run(self):
        if not self.state.need_case():
            return
        name = self.model_var.get()
        if not name:
            messagebox.showinfo("Modelo", "Carregue um modelo (botão 'Carregar pesos…' no topo).")
            return
        spec, lc = self.state.models[name], self.state.case
        missing = [p for p in spec.phases_used if p not in lc.vols]
        if missing:
            messagebox.showerror("Fases", f"O paciente atual não tem as fases {missing} exigidas pelo modelo.")
            return
        self.status.set("Calculando…")
        strategies = [s for s in STRATEGIES if lc.mask is not None or s not in ("max_area", "tumor_wmean")]

        def work():
            pf = make_predict_fn(spec, lc.vols)
            per_slice = np.asarray(pf(list(range(lc.n_slices))))
            lookup = lambda zs: per_slice[list(zs)]
            return {s: predict_volume(lookup, lc.n_slices, lc.mask, s) for s in strategies}, per_slice

        run_async(self.winfo_toplevel(), work, lambda r: self._show(r, spec, lc),
                  lambda e: (self.status.set(""), messagebox.showerror("Estratégias", str(e))))

    def _show(self, result, spec, lc):
        res, per = result
        self.status.set("")
        self.tree.delete(*self.tree.get_children())
        for s, r in res.items():
            sl = r["slices"]
            txt = f"z={sl[0]}" if len(sl) == 1 else f"todos ({len(sl)})"
            self.tree.insert("", "end", values=(STRATEGIES[s], f"{r['prob']:.3f}",
                                                "pCR" if r["prob"] >= spec.threshold else "sem pCR", txt))
        self.fig.clear()
        ax = self.fig.subplots()
        z = np.arange(lc.n_slices)
        ax.plot(z, per, "o-", color="#c44e52", label="P(pCR) por corte")
        ax.axhline(spec.threshold, color="gray", ls=":", label=f"limiar {spec.threshold:.2f}")
        ax.axvline(central_slice(lc.n_slices), color="#4c72b0", ls="--", label="corte central")
        if lc.mask is not None:
            ax.axvline(max_area_slice(lc.mask, lc.n_slices), color="#55a868", ls="--", label="maior área")
            ax2 = ax.twinx()
            ax2.fill_between(z, lc.mask.sum(axis=(0, 1)), alpha=0.15, color="green")
            ax2.set_ylabel("área tumoral (voxels)")
        ax.set_ylim(0, 1)
        ax.set_xlabel("corte z")
        ax.set_ylabel("P(pCR)")
        ax.set_title(f"{lc.pid} - rótulo real: {lc.label}")
        ax.legend(fontsize=7, loc="upper left")
        self.fig.tight_layout()
        self.canvas.draw_idle()

                                                                               
                          
                                                                               

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

                                                                               
                    
 
                                                                                     
 
                                                                                     
 
                                            
                                                                               
                                                                                         
                                                                                       
                                                                                      
                                                                                      
                                                                                     
                                                                            
                                                                          
                                                     
                                                                     
                                                                               

def load_nii_array(path: str) -> np.ndarray:
                                                                                   
    import nibabel as nib
    arr = np.squeeze(np.asarray(nib.load(path).dataobj, dtype=np.float32))
    if arr.ndim != 3:
        raise ValueError(f"Volume com dimensoes inesperadas {arr.shape}: {path}")
    return arr


def plane_axis(shape) -> int:
                                                                                  
    return int(np.argmin(shape))


def plane_axis_last(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    axis = plane_axis(arr.shape) if axis is None else axis
    return np.moveaxis(arr, axis, -1)


def normalize_percentile(vol, p_low=NORM_P_LOW, p_high=NORM_P_HIGH) -> np.ndarray:
                                             
    lo, hi = np.percentile(vol, [p_low, p_high])
    if hi - lo < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


_ID = ["pid", "patient_id", "id", "case", "case_id", "subject", "patient"]
_LB = ["pcr", "pcr_label", "label", "class", "target", "y"]
_SP = ["split", "partition", "set", "subset"]


def _pick(cols, cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    return None


def _pc_norm_split(v):
    s = str(v).strip().lower()
    for key, al in {"train": ["train", "training", "treino", "0"], "val": ["val", "valid", "validation", "validacao", "2"],
                    "test": ["test", "testing", "teste", "1"]}.items():
        if s in al:
            return key
    return s


def read_patient_table(csv_path: str, splits_csv: Optional[str] = None, seed: int = 42):
                                                                         
                                                                                  
                                                                              
    import pandas as pd
    df = pd.read_csv(csv_path)
    idc, lbc, spc = _pick(df.columns, _ID), _pick(df.columns, _LB), _pick(df.columns, _SP)
    if idc is None or lbc is None:
        raise ValueError(f"CSV sem coluna de id/rotulo. Colunas: {list(df.columns)}")
    out = pd.DataFrame({"pid": df[idc].astype(str).str.strip(), "label": df[lbc]})
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(float).astype(int)
    if spc is not None:
        out["split"] = df.loc[out.index, spc].map(_pc_norm_split)
    elif splits_csv:
        sp = pd.read_csv(splits_csv)
        m = dict(zip(sp[_pick(sp.columns, _ID)].astype(str), sp[_pick(sp.columns, _SP)].map(_pc_norm_split)))
        out["split"] = out["pid"].map(m)
        out = out.dropna(subset=["split"])
    else:
        from sklearn.model_selection import train_test_split
        tr, rest = train_test_split(out, test_size=0.3, stratify=out["label"], random_state=seed)
        va, te = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=seed)
        out["split"] = None
        out.loc[tr.index, "split"] = "train"
        out.loc[va.index, "split"] = "val"
        out.loc[te.index, "split"] = "test"
    return out.reset_index(drop=True)


def find_files(data_dir: str, pid: str) -> Dict:
                                                                                
                                                                                    
                                                                      
    files = [f for f in glob.glob(os.path.join(data_dir, "**", "*.nii*"), recursive=True)
             if os.path.basename(f).lower().startswith(pid.lower() + "_")
             or os.path.basename(f).lower().startswith(pid.lower() + ".")]
    out = {"phases": {}, "mask": None}
    for f in sorted(files):
        low = os.path.basename(f).lower()
        if "mask" in low or "seg" in low:
            out["mask"] = f
            continue
        m = re.search(r"aqc_(\d)", low) or re.search(r"_000(\d)\.nii", low)
        if m and int(m.group(1)) < 3:
            out["phases"][PHASES[int(m.group(1))]] = f
    return out


def build_patient_arrays(phase_vols: Sequence[np.ndarray], mask: Optional[np.ndarray]):
                                                                                      
    ax = plane_axis(phase_vols[0].shape)
    vols = [normalize_percentile(plane_axis_last(v, ax)) for v in phase_vols]
    shapes = {v.shape for v in vols}
    if len(shapes) != 1:
        raise ValueError(f"Fases com formatos diferentes: {shapes}")
    vol = np.stack(vols, axis=0)
    vol = np.transpose(vol, (3, 0, 1, 2)).astype(np.float16)
    if mask is None:
        m = np.zeros((vol.shape[0],) + vol.shape[2:], np.uint8)
    else:
        m = (plane_axis_last(mask, ax) > 0.5)
        if m.shape != vols[0].shape:
            raise ValueError(f"Mascara {m.shape} != volume {vols[0].shape}")
        m = np.transpose(m, (2, 0, 1)).astype(np.uint8)
    return vol, m


def build_cache(data_dir, csv_path, cache_dir, splits_csv=None, verbose=True):
                                                                                  
    import pandas as pd
    os.makedirs(cache_dir, exist_ok=True)
    table = read_patient_table(csv_path, splits_csv)
    rows, skipped = [], []
    for i, r in table.iterrows():
        f = find_files(data_dir, r["pid"])
        if len(f["phases"]) < 3:
            skipped.append(r["pid"])
            continue
        try:
            vol, m = build_patient_arrays([load_nii_array(f["phases"][p]) for p in PHASES],
                                          load_nii_array(f["mask"]) if f["mask"] else None)
        except Exception as e:
            skipped.append(f"{r['pid']} ({str(e)[:60]})")
            continue
        np.save(os.path.join(cache_dir, f"{r['pid']}_vol.npy"), vol)
        np.save(os.path.join(cache_dir, f"{r['pid']}_mask.npy"), m)
        areas = m.reshape(m.shape[0], -1).sum(1)
        rows.append({"pid": r["pid"], "label": r["label"], "split": r["split"], "n_slices": vol.shape[0],
                     "n_tumor_slices": int((areas > 0).sum()), "has_mask": int(f["mask"] is not None),
                     "shape_xy": f"{vol.shape[2]}x{vol.shape[3]}"})
        if verbose and (len(rows) % 10 == 0):
            print(f"  cache: {len(rows)} pacientes")
    idx = pd.DataFrame(rows)
    idx.to_csv(os.path.join(cache_dir, "index.csv"), index=False)
    if verbose:
        print(f"Cache pronto: {len(idx)} pacientes | ignorados: {len(skipped)} {skipped[:5]}")
        print(pd.crosstab(idx["split"], idx["label"], margins=True))
    return idx


@dataclass
class PatientData:
    pid: str
    label: int
    split: str
    vol: np.ndarray
    mask: np.ndarray
    areas: np.ndarray

    @property
    def n_slices(self):
        return self.vol.shape[0]


def load_cache(cache_dir: str, split: Optional[str] = None, mmap=True) -> List[PatientData]:
    import pandas as pd
    idx = pd.read_csv(os.path.join(cache_dir, "index.csv"))
    if split:
        idx = idx[idx["split"] == split]
    out = []
    for _, r in idx.iterrows():
        mm = "r" if mmap else None
        vol = np.load(os.path.join(cache_dir, f"{r['pid']}_vol.npy"), mmap_mode=mm)
        mask = np.load(os.path.join(cache_dir, f"{r['pid']}_mask.npy"), mmap_mode=mm)
        areas = np.asarray(mask).reshape(mask.shape[0], -1).sum(1).astype(float)
        out.append(PatientData(r["pid"], int(r["label"]), r["split"], vol, mask, areas))
    return out


def check_no_leakage(patients_by_split: Dict[str, List[PatientData]]):
                                                          
    seen = {}
    for sp, lst in patients_by_split.items():
        for p in lst:
            if p.pid in seen and seen[p.pid] != sp:
                raise AssertionError(f"VAZAMENTO: {p.pid} em {seen[p.pid]} e {sp}")
            seen[p.pid] = sp
    return True


def training_slices(areas: np.ndarray, mode: str) -> List[int]:
                                                                                              
    n = len(areas)
    if mode == "central":
        return [n // 2]
    if mode == "max_area":
        return [int(np.argmax(areas)) if areas.max() > 0 else n // 2]
    tz = np.where(areas > 0)[0].tolist()
    return tz if tz else list(range(n))


def aggregate(per_slice: np.ndarray, areas: np.ndarray, strategy: str) -> float:
                                                                                 
    n = len(per_slice)
    if strategy == "central":
        return float(per_slice[n // 2])
    if strategy == "max_area":
        return float(per_slice[int(np.argmax(areas))]) if areas.max() > 0 else float(per_slice[n // 2])
    if strategy == "mean":
        return float(per_slice.mean())
    if strategy == "max":
        return float(per_slice.max())
    if strategy == "tumor_wmean":
        return float((per_slice * areas / areas.sum()).sum()) if areas.sum() > 0 else float(per_slice.mean())
    raise ValueError(f"Estrategia desconhecida: {strategy}")


def slices_needed(areas: np.ndarray, strategy: str) -> List[int]:
                                                                                      
    n = len(areas)
    if strategy == "central":
        return [n // 2]
    if strategy == "max_area":
        return [int(np.argmax(areas)) if areas.max() > 0 else n // 2]
    if strategy == "tumor_wmean":
        tz = np.where(areas > 0)[0].tolist()
        return tz if tz else list(range(n))
    return list(range(n))


def choose_threshold(y_true, y_prob, method: str = "f1") -> float:
                                                                                         
                                                                              
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    if len(np.unique(y)) < 2:
        return 0.5
    u = np.unique(p)
    cands = np.concatenate([[0.5], (u[:-1] + u[1:]) / 2.0]) if len(u) > 1 else np.array([0.5])
    best, best_s = 0.5, -1.0
    for t in cands:
        pred = (p >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        if method == "youden":
            s = sens + spec - 1.0
        else:
            prec = tp / max(tp + fp, 1)
            s = 2 * prec * sens / max(prec + sens, 1e-12)
        if s > best_s + 1e-12 or (abs(s - best_s) <= 1e-12 and abs(t - 0.5) < abs(best - 0.5)):
            best, best_s = float(t), float(s)
    return best


def make_synthetic_cache(cache_dir, n=24, size=64, nz=10, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    os.makedirs(cache_dir, exist_ok=True)
    yy, xx = np.mgrid[:size, :size]
    rows = []
    for i in range(n):
        pcr = int(i % 3 == 0)
        cx, cy = rng.integers(size // 3, 2 * size // 3, 2)
        r0 = size * (rng.uniform(0.12, 0.15) if pcr else rng.uniform(0.19, 0.24))
        mask = np.zeros((size, size, nz), np.float32)
        for z in range(nz):
            rz = r0 * max(0.0, 1 - abs(z - nz / 2) / (nz / 2 + 0.5))
            mask[:, :, z] = ((xx - cx) ** 2 + (yy - cy) ** 2 <= rz ** 2)
        bg = rng.normal(100, 3, (size, size, nz)).astype(np.float32)
        gain = 0.6 if pcr else 1.6
        vol, m = build_patient_arrays([bg + 20 * mask, bg + 150 * gain * mask, bg + 90 * gain * mask], mask)
        pid = f"S{i + 1:03d}"
        np.save(os.path.join(cache_dir, f"{pid}_vol.npy"), vol)
        np.save(os.path.join(cache_dir, f"{pid}_mask.npy"), m)
        sp = "train" if i < n * 0.6 else ("val" if i < n * 0.8 else "test")
        rows.append({"pid": pid, "label": pcr, "split": sp, "n_slices": nz,
                     "n_tumor_slices": int((m.reshape(nz, -1).sum(1) > 0).sum()), "has_mask": 1,
                     "shape_xy": f"{size}x{size}"})
    pd.DataFrame(rows).to_csv(os.path.join(cache_dir, "index.csv"), index=False)
    return cache_dir



GRAY_MEAN, GRAY_STD = 0.449, 0.226


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters()
               if (p.requires_grad or not trainable_only))


class _Normalize(nn.Module):
                                                                            

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def _make_head(in_features, hidden, dropout):
                                                                          
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, hidden),
                         nn.ReLU(inplace=False), nn.Dropout(dropout), nn.Linear(hidden, 1))


class _FreezeMixin:
                                                                          

    def _backbone_parts(self):
        raise NotImplementedError

    def _last_block_parts(self):
        raise NotImplementedError

    def freeze_backbone(self):
        for m in self._backbone_parts():
            for p in m.parameters():
                p.requires_grad = False
        self.train(self.training)

    def unfreeze_last_block(self):
        for m in self._last_block_parts():
            for p in m.parameters():
                p.requires_grad = True
        self.train(self.training)

    def train(self, mode=True):
                                                                                 
                                                                                   
        super().train(mode)
        if mode:
            for m in self._backbone_parts():
                if not any(p.requires_grad for p in m.parameters()):
                    m.eval()
        return self


CNN_SPEC = [(16, 5, True), (32, 3, True), (64, 3, True),
            (128, 3, True), (128, 3, True), (256, 3, False)]


class PCCNN(nn.Module):
                                                                          

                                                                                       
                                                                                     
                                                                            
       

    def __init__(self, in_channels=1, dropout=0.5, hidden=64, spec=CNN_SPEC):
        super().__init__()
        blocks, c = [], in_channels
        for out_c, k, pool in spec:
            layers = [nn.Conv2d(c, out_c, k, padding=k // 2, bias=False),
                      nn.BatchNorm2d(out_c), nn.ReLU(inplace=False)]
            if pool:
                layers.append(nn.MaxPool2d(2))
            blocks.append(nn.Sequential(*layers))
            c = out_c
        self.blocks = nn.ModuleList(blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = _make_head(c, hidden, dropout)
        self.gradcam_layers = [self.blocks[-1]]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(self.gap(x).flatten(1))


def conv_out(w, k, p=0, s=1):
                                                                  
    return (w - k + 2 * p) // s + 1


def cnn_layer_table(in_channels=1, input_size=256, hidden=64, spec=CNN_SPEC):
                                                                                        
                                                                
    rows, block_shapes = [], []
    w, c = input_size, in_channels
    rows.append(("Entrada", "", (c, w, w), 0))
    for i, (o, k, pool) in enumerate(spec, 1):
        w = conv_out(w, k, k // 2, 1)
        n_par = k * k * c * o + 2 * o
        rows.append((f"Conv{i}+BN+ReLU", f"{k}x{k}, {o} filtros, pad {k // 2}", (o, w, w), n_par))
        if pool:
            w = conv_out(w, 2, 0, 2)
            rows.append(("MaxPool", "2x2, passo 2", (o, w, w), 0))
        block_shapes.append((o, w, w))
        c = o
    rows.append(("GlobalAvgPool", "", (c, 1, 1), 0))
    rows.append(("FC1+ReLU", f"{c}->{hidden}", (hidden,), c * hidden + hidden))
    rows.append(("FC2 (logit)", f"{hidden}->1", (1,), hidden + 1))
    return rows, block_shapes


def verify_cnn_dims(model, in_channels=1, input_size=256, hidden=64):
                                                                                       
    rows, expected = cnn_layer_table(in_channels, input_size, hidden)
    got = []
    hooks = [b.register_forward_hook(lambda m, i, o: got.append(tuple(o.shape[1:])))
             for b in model.blocks]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, in_channels, input_size, input_size))
    model.train(was_training)
    for h in hooks:
        h.remove()
    ok_shape = got == expected
    hand = sum(r[3] for r in rows)
    real = count_params(model)
    return {"ok": ok_shape and hand == real, "shapes_ok": ok_shape,
            "params_hand": hand, "params_real": real, "rows": rows}


def _torchvision_resnet(name, pretrained):
    from torchvision import models
    ctor = getattr(models, name)
    if pretrained:
        try:
            weights = getattr(models, f"{name.capitalize().replace('net', 'Net')}_Weights").IMAGENET1K_V1
        except AttributeError:
            weights = None
        try:
            return ctor(weights=weights)
        except TypeError:
            return ctor(pretrained=True)
    try:
        return ctor(weights=None)
    except TypeError:
        return ctor(pretrained=False)


class PCResNet50(_FreezeMixin, nn.Module):
                                                                               
                                                                             

    def __init__(self, in_channels=3, dropout=0.3, head_hidden=256, pretrained=False, **kw):
        super().__init__()
        if in_channels != 3:
            raise ValueError("PCResNet50 espera 3 canais (replique a fase unica).")
        base = _torchvision_resnet("resnet50", pretrained)
        self.norm = _Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.conv1, self.bn1, self.relu, self.maxpool = base.conv1, base.bn1, base.relu, base.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = _make_head(base.fc.in_features, head_hidden, dropout)
        self.gradcam_layers = [self.layer4]

    def _backbone_parts(self):
        return [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]

    def _last_block_parts(self):
        return [self.layer4]

    def forward(self, x):
        x = self.norm(x)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.avgpool(x).flatten(1))


def _gray_resnet18(pretrained):
                                                                            
                                                                          
    base = _torchvision_resnet("resnet18", pretrained)
    old = base.conv1
    new = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
    if pretrained:
        with torch.no_grad():
            new.weight.copy_(old.weight.sum(dim=1, keepdim=True))
    base.conv1 = new
    base.fc = nn.Identity()
    return base


class PCMultiBranchResNet18(_FreezeMixin, nn.Module):
                                                                                        
                                                                        
                                                                               

    def __init__(self, in_channels=3, dropout=0.4, head_hidden=128, pretrained=False, **kw):
        super().__init__()
        self.n_branches = in_channels
        self.norm = _Normalize((GRAY_MEAN,) * in_channels, (GRAY_STD,) * in_channels)
        self.branches = nn.ModuleList([_gray_resnet18(pretrained) for _ in range(in_channels)])
        self.head = _make_head(512 * in_channels, head_hidden, dropout)
        self.gradcam_layers = [b.layer4 for b in self.branches]

    def _backbone_parts(self):
        return list(self.branches)

    def _last_block_parts(self):
        return [b.layer4 for b in self.branches]

    def forward(self, x):
        x = self.norm(x)
        feats = [b(x[:, i:i + 1]) for i, b in enumerate(self.branches)]
        return self.head(torch.cat(feats, dim=1))


def model_in_channels(arch, mode):
                                                                    
    if arch == "resnet50":
        return 3
    return 3 if mode == "fusion" else 1


def build_cnn(in_channels=1, dropout=0.5, hidden=64, **kw):
    return PCCNN(in_channels=in_channels, dropout=dropout, hidden=hidden)


def build_resnet50(in_channels=3, dropout=0.3, head_hidden=256, **kw):
    return PCResNet50(in_channels, dropout, head_hidden, pretrained=False)


def build_three_branch(in_channels=3, dropout=0.4, head_hidden=128, **kw):
    return PCMultiBranchResNet18(in_channels, dropout, head_hidden, pretrained=False)


register_arch("cnn", build_cnn)
register_arch("resnet50", build_resnet50)
register_arch("three_branch", build_three_branch)

BUILDERS = {"cnn": build_cnn, "resnet50": build_resnet50, "three_branch": build_three_branch}


def make_model(arch, mode, pretrained=True, dropout=None, hidden=None):
                                                                                       
    cin = model_in_channels(arch, mode)
    kw = {}
    if dropout is not None:
        kw["dropout"] = dropout
    if arch == "cnn":
        if hidden is not None:
            kw["hidden"] = hidden
        return PCCNN(in_channels=cin, **kw)
    if hidden is not None:
        kw["head_hidden"] = hidden
    cls = PCResNet50 if arch == "resnet50" else PCMultiBranchResNet18
    return cls(in_channels=cin, pretrained=pretrained, **kw)


def checkpoint_dict(model, arch, mode, threshold=0.5, input_size=256, extra=None):
                                                                                    
    d = {"arch": arch, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
         "mode": mode, "in_channels": model_in_channels(arch, mode), "input_size": input_size,
         "output_kind": "logit", "imagenet_norm": False, "threshold": float(threshold),
         "norm": "percentile"}
    d.update(extra or {})
    return d

                                                                               
                     
 
                                                                            
 
                                                                                     
 
                                          
                                                                                           
                                                                                              
                                                                                            
                                                                                 
                                                                                                     
                                                                                                      
                                                                                                      
 
                                                                                      
                                                                                   
                                                                               

MAIN_STRATEGY = {"central": "central", "max_area": "max_area", "tumor": "tumor_wmean"}
PHASE_IDX = {"pre": 0, "early": 1, "late": 2}
METRIC_KEYS = ["sensibilidade", "especificidade", "precisao", "acuracia", "f1", "auc"]


@dataclass
class Exp:
    arch: str = "cnn"
    mode: str = "fusion"
    train_slices: str = "central"
    seed: int = 0
    batch: int = 16
    epochs: int = 60
    lr: float = 1e-3
    epochs_head: int = 15
    lr_head: float = 1e-3
    epochs_ft: int = 15
    lr_ft: float = 1e-4
    finetune: bool = True
    wd: float = 1e-4
    dropout: Optional[float] = None
    hidden: Optional[int] = None
    patience: int = 10
    loss: str = "wbce"
    augment: bool = True
    slices_per_epoch: int = 6
    select: str = "auc"
    thr_method: str = "f1"
    input_size: int = 256
    pretrained: bool = True
    tag: str = ""

    @property
    def phase(self):
        return "fusion" if self.mode == "fusion" else self.mode.split(":")[1]

    @property
    def name(self):
        return f"{self.arch}_{self.phase}_{self.train_slices}_s{self.seed}" + (f"_{self.tag}" if self.tag else "")

    @property
    def main_strategy(self):
        return MAIN_STRATEGY[self.train_slices]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mode_channels(mode):
    return [0, 1, 2] if mode == "fusion" else [PHASE_IDX[mode.split(":")[1]]]


def load_slices(p: PatientData, zs, chans) -> np.ndarray:
                                                             
    return np.asarray(p.vol[list(zs)])[:, chans].astype(np.float32)


def to_input(x_np, in_ch, size, device):
                                                                                                
    t = torch.from_numpy(x_np)
    if t.shape[1] == 1 and in_ch == 3:
        t = t.repeat(1, 3, 1, 1)
    elif t.shape[1] != in_ch:
        raise ValueError(f"Modelo espera {in_ch} canais, entrada tem {t.shape[1]}")
    if t.shape[-1] != size or t.shape[-2] != size:
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.to(device)


def augment_batch(x, rng):
                                                                                        
                                                                                      
                                                                               
    B = x.shape[0]
    flip = torch.from_numpy(rng.random(B) < 0.5).to(x.device)
    x = torch.where(flip.view(B, 1, 1, 1), x.flip(-1), x)
    ang = torch.from_numpy(rng.choice(AUG_ANGLES, B).astype(np.float32)).to(x.device) * math.pi / 180.0
    c, s = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = c, -s, s, c
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


class FocalLoss(nn.Module):
    def __init__(self, alpha_pos, gamma=2.0):
        super().__init__()
        self.a, self.g = alpha_pos, gamma

    def forward(self, logit, y):
        bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
        pt = torch.exp(-bce)
        a = y * self.a + (1 - y) * (1 - self.a)
        return (a * (1 - pt) ** self.g * bce).mean()


def make_loss(kind, n_pos, n_neg, device):
    if kind == "bce":
        return nn.BCEWithLogitsLoss()
    if kind == "wbce":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=device))
    if kind == "focal":
        return FocalLoss(alpha_pos=n_neg / max(n_pos + n_neg, 1))
    raise ValueError(kind)


def epoch_samples(pats, mode, k, rng):
    s = []
    for pi, p in enumerate(pats):
        zs = training_slices(p.areas, mode)
        if mode == "tumor" and k and len(zs) > k:
            zs = rng.choice(zs, k, replace=False).tolist()
        s += [(pi, int(z)) for z in zs]
    rng.shuffle(s)
    return s


def autocast_ctx(device, amp):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp)


@torch.no_grad()
def predict_logits(model, p, zs, chans, in_ch, size, device, amp, bs=32):
                                                         
    model.eval()
    out = []
    for i in range(0, len(zs), bs):
        x = to_input(load_slices(p, zs[i:i + bs], chans), in_ch, size, device)
        with autocast_ctx(device, amp):
            lg = model(x)
        out.append(lg.float().reshape(len(x), -1)[:, 0].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def evaluate_main(model, pats, exp, device, amp, crit=None):
                                                                                       
                                                                       
    chans, in_ch = mode_channels(exp.mode), model_in_channels(exp.arch, exp.mode)
    strat = exp.main_strategy
    probs, ys, all_lg, all_y = [], [], [], []
    for p in pats:
        zs = slices_needed(p.areas, strat)
        lg = predict_logits(model, p, zs, chans, in_ch, exp.input_size, device, amp)
        per = np.zeros(p.n_slices); per[zs] = 1.0 / (1.0 + np.exp(-lg))
        probs.append(aggregate(per, p.areas, strat)); ys.append(p.label)
        all_lg.append(lg); all_y.append(np.full(len(lg), p.label, np.float32))
    vloss = float("nan")
    if crit is not None and all_lg:
        vloss = float(crit(torch.from_numpy(np.concatenate(all_lg)).float().to(device),
                           torch.from_numpy(np.concatenate(all_y)).to(device)).item())
    auc = binary_metrics(ys, probs)["auc"]
    return vloss, auc, np.array(ys), np.array(probs)


def predict_all_strategies(model, pats, exp, device, amp):
                                                                                             
    chans, in_ch = mode_channels(exp.mode), model_in_channels(exp.arch, exp.mode)
    out = {s: [] for s in STRATEGIES}
    for p in pats:
        per = 1.0 / (1.0 + np.exp(-predict_logits(model, p, list(range(p.n_slices)), chans, in_ch,
                                                  exp.input_size, device, amp)))
        for s in STRATEGIES:
            out[s].append(aggregate(per, p.areas, s))
    return {s: np.array(v) for s, v in out.items()}


def make_scaler(amp):
    try:
        return torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=amp)


def plan_stages(model, exp):
                                                                                             
                               
    if exp.arch == "cnn":
        return [{"name": "tudo", "epochs": exp.epochs, "unfreeze": False, "lr": exp.lr}]
    st = [{"name": "cabeca", "epochs": exp.epochs_head, "unfreeze": False, "lr": exp.lr_head}]
    if exp.finetune and exp.epochs_ft > 0:
        st.append({"name": "layer4", "epochs": exp.epochs_ft, "unfreeze": True, "lr": exp.lr_ft})
    return st


def train_model(exp: Exp, train_pats, val_pats, device, log=print):
    set_seed(exp.seed)
    rng = np.random.default_rng(exp.seed)
    amp = device.type == "cuda"
    model = make_model(exp.arch, exp.mode, pretrained=exp.pretrained, dropout=exp.dropout, hidden=exp.hidden).to(device)
    chans, in_ch = mode_channels(exp.mode), model_in_channels(exp.arch, exp.mode)
    pretrained_arch = exp.arch != "cnn"
    if pretrained_arch:
        model.freeze_backbone()

    s0 = epoch_samples(train_pats, exp.train_slices, exp.slices_per_epoch, np.random.default_rng(exp.seed))
    n_pos = sum(train_pats[pi].label for pi, _ in s0); n_neg = len(s0) - n_pos
    crit = make_loss(exp.loss, n_pos, n_neg, device)
    log(f"  amostras/epoca={len(s0)}  pCR={n_pos}  nao-pCR={n_neg}  perda={exp.loss}"
        f"  pos_weight={n_neg / max(n_pos, 1):.2f}")

    hist = {"train_loss": [], "val_loss": [], "val_auc": [], "stage_start": []}
    best = {"score": -1e9, "vloss": 1e9, "state": None, "epoch": 0}
    scaler = make_scaler(amp)
    t0, ep = time.time(), 0
    for stage in plan_stages(model, exp):
        if stage["unfreeze"]:
            if best["state"] is not None:
                model.load_state_dict(best["state"])
            model.unfreeze_last_block()
        params = [p for p in model.parameters() if p.requires_grad]
        groups = [{"params": params, "lr": stage["lr"]}]
        opt = torch.optim.AdamW(groups, weight_decay=exp.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(stage["epochs"], 1),
                                                           eta_min=stage["lr"] * 0.05)
        hist["stage_start"].append(ep + 1)
        bad = 0
        for _ in range(stage["epochs"]):
            ep += 1
            model.train()
            samples = epoch_samples(train_pats, exp.train_slices, exp.slices_per_epoch, rng)
            tot, cnt = 0.0, 0
            for i in range(0, len(samples), exp.batch):
                chunk = samples[i:i + exp.batch]
                if len(chunk) < 2:
                    continue
                x = np.stack([np.asarray(train_pats[pi].vol[z])[chans] for pi, z in chunk]).astype(np.float32)
                y = torch.tensor([train_pats[pi].label for pi, _ in chunk], dtype=torch.float32, device=device)
                xt = to_input(x, in_ch, exp.input_size, device)
                if exp.augment:
                    xt = augment_batch(xt, rng)
                opt.zero_grad(set_to_none=True)
                with autocast_ctx(device, amp):
                    lg = model(xt)
                loss = crit(lg.float().reshape(-1), y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update()
                tot += loss.item() * len(chunk); cnt += len(chunk)
            sched.step()
            vloss, vauc, _, _ = evaluate_main(model, val_pats, exp, device, amp, crit)
            tl = tot / max(cnt, 1)
            hist["train_loss"].append(tl); hist["val_loss"].append(vloss); hist["val_auc"].append(vauc)
            score = -vloss if exp.select == "loss" else (vauc if not math.isnan(vauc) else 0.5)
            improved = score > best["score"] + 1e-6 or (abs(score - best["score"]) <= 1e-6 and vloss < best["vloss"])
            if improved:
                best.update(score=score, vloss=vloss, epoch=ep,
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                bad = 0
            else:
                bad += 1
            log(f"  [{stage['name']}] ep {ep:03d} loss={tl:.4f} val_loss={vloss:.4f} val_auc={vauc:.3f}"
                f"{' *' if improved else ''}")
            if bad >= exp.patience:
                log(f"  early stopping (paciencia {exp.patience})")
                break
    train_time = time.time() - t0
    model.load_state_dict(best["state"])
    model.eval()
    hist.update(best_epoch=best["epoch"], epochs_run=ep, train_time_s=train_time)
    return model, hist


LOG_COLS = (["exp", "arch", "phase", "train_slices", "eval_strategy", "seed", "loss", "augment", "lr", "lr_ft",
             "wd", "dropout", "hidden", "batch", "epochs_run", "best_epoch", "train_time_s", "n_params",
             "thr_val"] + [f"m05_{k}" for k in METRIC_KEYS] + [f"mt_{k}" for k in METRIC_KEYS] + ["ckpt"])


def append_log(rows, path):
    df = pd.DataFrame(rows).reindex(columns=LOG_COLS)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def run_experiment(exp: Exp, ctx, force=False, log=print):
    names = ctx["paths"]
    res_json = os.path.join(names["results"], f"{exp.name}.json")
    if os.path.exists(res_json) and not force:
        log(f"[pulado] {exp.name} (ja existe)")
        return None
    log(f"\n=== {exp.name} | modelo={exp.arch} fases={exp.phase} cortes={exp.train_slices} ===")
    device = ctx["device"]
    model, hist = train_model(exp, ctx["train"], ctx["val"], device, log)
    amp = device.type == "cuda"
    probs = predict_all_strategies(model, ctx["val"], exp, device, amp)
    y = np.array([p.label for p in ctx["val"]])
    thr_main = choose_threshold(y, probs[exp.main_strategy], exp.thr_method)

    rows, all_metrics = [], {}
    n_params = count_params(model)
    for s in STRATEGIES:
        thr_s = choose_threshold(y, probs[s], exp.thr_method)
        m05, mt = binary_metrics(y, probs[s], 0.5), binary_metrics(y, probs[s], thr_s)
        all_metrics[s] = {"thr": thr_s, "m05": m05, "mt": mt}
        row = {"exp": exp.name, "arch": exp.arch, "phase": exp.phase, "train_slices": exp.train_slices,
               "eval_strategy": s, "seed": exp.seed, "loss": exp.loss, "augment": int(exp.augment),
               "lr": exp.lr if exp.arch == "cnn" else exp.lr_head, "lr_ft": exp.lr_ft if exp.arch != "cnn" else "",
               "wd": exp.wd, "dropout": exp.dropout, "hidden": exp.hidden, "batch": exp.batch,
               "epochs_run": hist["epochs_run"], "best_epoch": hist["best_epoch"],
               "train_time_s": round(hist["train_time_s"], 1), "n_params": n_params, "thr_val": thr_s,
               "ckpt": os.path.join(names["ckpt"], f"{exp.name}.pt")}
        row.update({f"m05_{k}": m05[k] for k in METRIC_KEYS})
        row.update({f"mt_{k}": mt[k] for k in METRIC_KEYS})
        rows.append(row)

    hp = {"dropout": exp.dropout if exp.dropout is not None else
          {"cnn": 0.5, "resnet50": 0.3, "three_branch": 0.4}[exp.arch]}
    default_h = {"cnn": 64, "resnet50": 256, "three_branch": 128}[exp.arch]
    hp["hidden" if exp.arch == "cnn" else "head_hidden"] = exp.hidden or default_h
    os.makedirs(names["ckpt"], exist_ok=True)
    torch.save(checkpoint_dict(model, exp.arch, exp.mode, thr_main, exp.input_size,
                                         extra={**hp, "strategy": exp.main_strategy, "train_slices": exp.train_slices,
                                                "exp_name": exp.name}), os.path.join(names["ckpt"], f"{exp.name}.pt"))
    mm = all_metrics[exp.main_strategy]["mt"]
    with open(res_json, "w", encoding="utf-8") as f:
        json.dump({"name": f"{exp.arch} | {exp.phase} | {exp.train_slices}",
                   "train_loss": hist["train_loss"], "val_loss": hist["val_loss"], "val_auc": hist["val_auc"],
                   "train_time_s": hist["train_time_s"], "metrics": {k: mm[k] for k in METRIC_KEYS},
                   "notes": f"VALIDACAO; limiar={thr_main:.3f} ({exp.thr_method}); estrategia={exp.main_strategy}; "
                            f"melhor epoca={hist['best_epoch']}; params={n_params}",
                   "stage_start": hist["stage_start"], "exp": asdict(exp),
                   "all_strategies": {s: {"thr": v["thr"], "auc": v["mt"]["auc"], "f1_thr05": v["m05"]["f1"],
                                          "f1_thr": v["mt"]["f1"]} for s, v in all_metrics.items()}},
                  f, indent=1, ensure_ascii=False)
    append_log(rows, names["log"])
    log(f"  >> VAL ({exp.main_strategy}): AUC={mm['auc']:.3f} F1={mm['f1']:.3f} sens={mm['sensibilidade']:.3f} "
        f"espec={mm['especificidade']:.3f} thr={thr_main:.2f} | tempo={hist['train_time_s']:.0f}s")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def make_ctx(args, smoke=False):
    cache = args.cache
    train, val = load_cache(cache, "train"), load_cache(cache, "val")
    check_no_leakage({"train": train, "val": val, "test": load_cache(cache, "test")})
    paths = {"results": args.results, "ckpt": args.ckpt_dir, "log": os.path.join(args.results, "log_experimentos.csv")}
    os.makedirs(args.results, exist_ok=True)
    print(f"Dispositivo: {get_torch_device()} | treino={len(train)} val={len(val)} pacientes | "
          f"pCR treino={sum(p.label for p in train)}/{len(train)}")
    return {"train": train, "val": val, "paths": paths, "device": get_torch_device()}


def exp_from_args(a, arch, mode, ts, seed):
    kw = {k: getattr(a, k) for k in ("batch", "wd", "patience", "loss", "slices_per_epoch", "select", "thr_method",
                                     "input_size", "tag") if hasattr(a, k)}
    e = Exp(arch=arch, mode=mode, train_slices=ts, seed=seed, augment=not a.no_aug, finetune=not a.no_finetune,
            dropout=a.dropout, hidden=a.hidden, **kw)
    for k in ("epochs", "lr", "epochs_head", "lr_head", "epochs_ft", "lr_ft"):
        if getattr(a, k, None) is not None:
            setattr(e, k, getattr(a, k))
    return e


def cmd_cache(a):
    build_cache(a.data_dir, a.csv, a.cache, a.splits_csv)


def cmd_check(a):
    r = verify_cnn_dims(PCCNN(1), 1, 256, 64)
    print("CNN propria (entrada 1x256x256) - dimensoes feitas a mao (W=floor((W-K+2P)/S)+1):")
    for name, desc, shp, par in r["rows"]:
        print(f"  {name:<16}{desc:<28}saida={str(shp):<16}parametros={par}")
    print(f"  total a mao={r['params_hand']}  total real={r['params_real']}  formas_ok={r['shapes_ok']}  -> "
          f"{'OK' if r['ok'] else 'DIVERGE'}")
    for arch in ("cnn", "resnet50", "three_branch"):
        for mode in ("single:pre", "fusion"):
            m = make_model(arch, mode, pretrained=False).eval()
            cin = model_in_channels(arch, mode)
            with torch.no_grad():
                o = m(torch.rand(2, cin, 256, 256))
            print(f"  {arch:<13}{mode:<12} in={cin} out={tuple(o.shape)} params={count_params(m):,}")
    print(make_model("cnn", "fusion", False))


def cmd_smoke(a):
    d = "_smoke_cache"
    make_synthetic_cache(d)
    a.cache, a.results, a.ckpt_dir = d, "_smoke_results", "_smoke_ckpt"
    ctx = make_ctx(a)
    for arch in ("cnn", "resnet50", "three_branch"):
        for mode, ts in (("fusion", "tumor"), ("single:late", "central")):
            e = Exp(arch=arch, mode=mode, train_slices=ts, epochs=2, epochs_head=1, epochs_ft=1, batch=8,
                    input_size=64, slices_per_epoch=3, pretrained=False)
            run_experiment(e, ctx, force=True)
    print("\nSMOKE OK: o pipeline completo roda. Apague _smoke_cache/_smoke_results/_smoke_ckpt.")


def cmd_train(a):
    ctx = make_ctx(a)
    for seed in a.seeds:
        run_experiment(exp_from_args(a, a.arch, a.mode, a.train_slices, seed), ctx, force=a.force)


def cmd_battery(a):
    ctx = make_ctx(a)
    todo = [(ar, mo, ts, sd) for ar in a.archs for mo in a.modes for ts in a.train_slices_list for sd in a.seeds]
    print(f"Bateria: {len(todo)} experimentos (retomavel: pula os que ja tem .json)")
    for i, (ar, mo, ts, sd) in enumerate(todo, 1):
        print(f"\n##### {i}/{len(todo)}")
        try:
            run_experiment(exp_from_args(a, ar, mo, ts, sd), ctx, force=a.force)
        except Exception as ex:
            print(f"[ERRO] {ar} {mo} {ts} seed={sd}: {ex}")


def _tex_table(df, cols, caption, label):
    head = " & ".join(cols)
    lines = [r"\begin{table}[h]", r"\centering", r"\small", rf"\begin{{tabular}}{{{'l' * len(cols)}}}", r"\hline",
             head + r" \\", r"\hline"]
    for _, r in df.iterrows():
        lines.append(" & ".join(str(r[c]) for c in cols) + r" \\")
    lines += [r"\hline", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def cmd_report(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    log = pd.read_csv(os.path.join(a.results, "log_experimentos.csv"))
    log["natural"] = log.apply(lambda r: r["eval_strategy"] == MAIN_STRATEGY[r["train_slices"]], axis=1)
    keys = ["arch", "phase", "train_slices", "eval_strategy"]
    agg = log.groupby(keys).agg(n_seeds=("seed", "nunique"), **{f"{k}_mean": (f"mt_{k}", "mean") for k in METRIC_KEYS},
                                auc_std=("mt_auc", "std"), f1_std=("mt_f1", "std"),
                                tempo_s=("train_time_s", "mean")).reset_index()
    agg.to_csv(os.path.join(a.results, "tabela_completa.csv"), index=False)
    nat = agg[agg.apply(lambda r: r["eval_strategy"] == MAIN_STRATEGY[r["train_slices"]], axis=1)].copy()
    nat = nat.sort_values(["arch", "phase", "auc_mean"], ascending=[True, True, False])
    nat.round(3).to_csv(os.path.join(a.results, "tabela_principal.csv"), index=False)
    disp = nat.round(3).astype({"n_seeds": int})
    print("\nTABELA PRINCIPAL (validacao, estrategia natural de cada treino; media entre seeds):")
    print(disp[["arch", "phase", "train_slices", "n_seeds", "auc_mean", "f1_mean", "sensibilidade_mean",
                "especificidade_mean", "tempo_s"]].to_string(index=False))
    with open(os.path.join(a.results, "tabela_principal.tex"), "w", encoding="utf-8") as f:
        t = disp.rename(columns={"auc_mean": "AUC", "f1_mean": "F1", "sensibilidade_mean": "Sens.",
                                 "especificidade_mean": "Espec.", "precisao_mean": "Prec.", "acuracia_mean": "Acur."})
        f.write(_tex_table(t, ["arch", "phase", "train_slices", "AUC", "F1", "Sens.", "Espec.", "Prec.", "Acur."],
                           "Desempenho na validacao (limiar escolhido na validacao).", "tab:principal"))
    print("\nHIPOTESE - 3 fases (fusao) x melhor fase unica (AUC na validacao, melhor corte de cada grupo):")
    for arch, g in nat.groupby("arch"):
        best = g.sort_values("auc_mean", ascending=False).groupby("phase").head(1).set_index("phase")["auc_mean"]
        singles = best.drop("fusion", errors="ignore")
        if "fusion" in best and len(singles):
            print(f"  {arch:<13} fusao={best['fusion']:.3f}  melhor fase unica={singles.max():.3f} "
                  f"({singles.idxmax()})  diferenca={best['fusion'] - singles.max():+.3f}")

    best_rows = log[log.natural].sort_values("mt_auc", ascending=False).groupby("arch").head(1)
    fig, axs = plt.subplots(1, max(len(best_rows), 1), figsize=(5 * max(len(best_rows), 1), 3.6), squeeze=False)
    for ax, (_, r) in zip(axs[0], best_rows.iterrows()):
        j = json.load(open(os.path.join(a.results, f"{r['exp']}.json"), encoding="utf-8"))
        ax.plot(j["train_loss"], label="treino"); ax.plot(j["val_loss"], label="validacao")
        for s in j.get("stage_start", [])[1:]:
            ax.axvline(s - 1, ls="--", c="gray", lw=0.8)
        ax.set_title(r["exp"], fontsize=8); ax.set_xlabel("epoca"); ax.set_ylabel("loss"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(a.results, "curvas_loss.png"), dpi=150)
    print(f"\nSalvos em {a.results}: tabela_principal.csv/.tex, tabela_completa.csv, curvas_loss.png")


def bootstrap_ci(y, p, thr, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    aucs, f1s = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        m = binary_metrics(y[i], p[i], thr)
        aucs.append(m["auc"]); f1s.append(m["f1"])
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (float("nan"),) * 2
    return {"auc_ci95": q(aucs), "f1_ci95": q(f1s)}


def load_ckpt_model(path, device):
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(path, map_location=device)
    model = BUILDERS[ck["arch"]](**{k: v for k, v in ck.items() if k != "state_dict"})
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def cmd_test(a):
                                                                                         
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    device = get_torch_device()
    model, ck = load_ckpt_model(a.ckpt, device)
    strat = a.strategy or ck.get("strategy", "mean")
    key = f"{os.path.basename(a.ckpt)}|{strat}"
    lock_path = os.path.join(a.results, "TESTE_avaliacoes.json")
    lock = json.load(open(lock_path)) if os.path.exists(lock_path) else {}
    if key in lock and not a.force:
        raise SystemExit(f"BLOQUEADO: {key} ja foi avaliado no teste em {lock[key]}. "
                         f"O enunciado permite uma unica avaliacao (use --force so para depurar).")
    test = load_cache(a.cache, "test")
    exp = Exp(arch=ck["arch"], mode=ck["mode"], input_size=ck["input_size"])
    probs = predict_all_strategies(model, test, exp, device, device.type == "cuda")[strat]
    y = np.array([p.label for p in test])
    thr = a.threshold if a.threshold is not None else ck["threshold"]
    m = binary_metrics(y, probs, thr)
    ci = bootstrap_ci(y, probs, thr)
    tag = f"{ck.get('exp_name', os.path.basename(a.ckpt)[:-3])}_{strat}"
    print(f"\nTESTE {tag} (n={len(y)}, limiar={thr:.3f})")
    print(f"  Sens={m['sensibilidade']:.3f} Espec={m['especificidade']:.3f} Prec={m['precisao']:.3f} "
          f"Acur={m['acuracia']:.3f} F1={m['f1']:.3f} AUC={m['auc']:.3f}  (IC95% AUC {ci['auc_ci95'][0]:.2f}-{ci['auc_ci95'][1]:.2f})")
    print(f"  TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")
    errs = pd.DataFrame({"pid": [p.pid for p in test], "label": y, "prob": probs, "pred": (probs >= thr).astype(int)})
    errs["erro"] = np.where(errs.pred == errs.label, "",
                            np.where(errs.pred == 1, "FP (previu pCR, nao houve)", "FN (previu nao-pCR, houve pCR)"))
    errs.sort_values("prob").to_csv(os.path.join(a.results, f"teste_{tag}_predicoes.csv"), index=False)
    print("  Erros:", errs[errs.erro != ""][["pid", "label", "prob", "erro"]].round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["nao-pCR", "pCR"]); ax.set_yticks([0, 1], ["nao-pCR", "pCR"])
    ax.set_xlabel("predito"); ax.set_ylabel("real"); ax.set_title(tag, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.results, f"teste_{tag}_confusao.png"), dpi=150)
    src = os.path.join(a.results, f"{ck.get('exp_name', '')}.json")
    base = json.load(open(src, encoding="utf-8")) if os.path.exists(src) else {}
    out = {"name": f"TESTE {tag}", "train_loss": base.get("train_loss", []), "val_loss": base.get("val_loss", []),
           "train_time_s": base.get("train_time_s"), "metrics": {k: m[k] for k in METRIC_KEYS},
           "notes": f"TESTE; limiar={thr:.3f}; estrategia={strat}; IC95 AUC={ci['auc_ci95']}", "ci": ci,
           "confusao": {k: m[k] for k in ("tp", "tn", "fp", "fn")}}
    json.dump(out, open(os.path.join(a.results, f"teste_{tag}.json"), "w", encoding="utf-8"), indent=1)
    lock[key] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(lock, open(lock_path, "w"), indent=1)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--cache", default="cache_pc")
        sp.add_argument("--results", default="results_pc")
        sp.add_argument("--ckpt_dir", default="ckpt_pc")

    def hp(sp):
        sp.add_argument("--epochs", type=int); sp.add_argument("--lr", type=float)
        sp.add_argument("--epochs_head", type=int); sp.add_argument("--lr_head", type=float)
        sp.add_argument("--epochs_ft", type=int); sp.add_argument("--lr_ft", type=float)
        sp.add_argument("--batch", type=int, default=16); sp.add_argument("--wd", type=float, default=1e-4)
        sp.add_argument("--dropout", type=float); sp.add_argument("--hidden", type=int)
        sp.add_argument("--patience", type=int, default=10)
        sp.add_argument("--loss", default="wbce", choices=["bce", "wbce", "focal"])
        sp.add_argument("--slices_per_epoch", type=int, default=6)
        sp.add_argument("--select", default="auc", choices=["auc", "loss"])
        sp.add_argument("--thr_method", default="f1", choices=["f1", "youden"])
        sp.add_argument("--input_size", type=int, default=256)
        sp.add_argument("--no_aug", action="store_true"); sp.add_argument("--no_finetune", action="store_true")
        sp.add_argument("--tag", default=""); sp.add_argument("--force", action="store_true")
        sp.add_argument("--seeds", type=int, nargs="+", default=[0])

    s = sub.add_parser("cache"); common(s)
    s.add_argument("--data_dir", required=True); s.add_argument("--csv", required=True)
    s.add_argument("--splits_csv"); s.set_defaults(fn=cmd_cache)
    s = sub.add_parser("check"); common(s); s.set_defaults(fn=cmd_check)
    s = sub.add_parser("smoke"); common(s); s.set_defaults(fn=cmd_smoke, no_aug=False, no_finetune=False)
    s = sub.add_parser("train"); common(s); hp(s)
    s.add_argument("--arch", required=True, choices=["cnn", "resnet50", "three_branch"])
    s.add_argument("--mode", default="fusion", choices=["fusion", "single:pre", "single:early", "single:late"])
    s.add_argument("--train_slices", default="central", choices=["central", "max_area", "tumor"])
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("battery"); common(s); hp(s)
    s.add_argument("--archs", nargs="+", default=["cnn", "resnet50", "three_branch"])
    s.add_argument("--modes", nargs="+", default=["single:pre", "single:early", "single:late", "fusion"])
    s.add_argument("--train_slices_list", nargs="+", default=["central", "max_area", "tumor"])
    s.set_defaults(fn=cmd_battery)
    s = sub.add_parser("report"); common(s); s.set_defaults(fn=cmd_report)
    s = sub.add_parser("test"); common(s)
    s.add_argument("--ckpt", required=True); s.add_argument("--strategy", choices=STRATEGIES)
    s.add_argument("--threshold", type=float); s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_test)
    return p

                                                                               
                       
                                                                               

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
        pass                                    
        import_errors = self._import_team_architectures()
        try:
            for p in paths:
                self._register(load_model_spec(p))
        except Exception as e:
            msg = str(e)
            if import_errors:
                msg += "\n\nFalhas ao importar os modulos da equipe:\n" + "\n".join(import_errors)
            messagebox.showerror("Carregar pesos", msg)

    @staticmethod
    def _import_team_architectures():
                                                                                            
                                                                                        
                                                                                           
                                         
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
        pass                                    
        self._register(make_demo_spec("fusion"))
        self._register(make_demo_spec("single:late", seed=1))


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
