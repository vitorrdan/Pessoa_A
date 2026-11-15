# -*- coding: utf-8 -*-
"""
PESSOA A - Leitura de .nii e indexacao do dataset.

Funcoes elementares de biblioteca usadas (permitido pelo enunciado):
  nibabel.load / get_fdata / header.get_zooms / aff2axcodes  -> leitura de .nii
  pandas.read_csv                                            -> leitura da planilha
"""
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from pa_config import (PHASES, PHASE_PATTERNS, MASK_PATTERNS, CSV_ID_COLS,
                       CSV_LABEL_COLS, CSV_SPLIT_COLS, SPLIT_ALIASES,
                       NORM_P_LOW, NORM_P_HIGH)


# --------------------------------------------------------------------------
# Leitura de volumes
# --------------------------------------------------------------------------
@dataclass
class Volume:
    data: np.ndarray                 # (X, Y, Z) float32, ordem de voxels do arquivo
    spacing: tuple = (1.0, 1.0, 1.0)
    axcodes: str = "?"               # orientacao anatomica do arquivo (p/ comparar c/ MRIcro)
    path: str = ""

    @property
    def shape(self):
        return self.data.shape

    @property
    def n_slices(self):
        return self.data.shape[2]


def load_nii(path: str) -> Volume:
    """Le um .nii/.nii.gz como (X, Y, Z). Volumes 4D (X,Y,Z,1) sao achatados;
    2D ganha um eixo z de tamanho 1. NAO reorienta os voxels: o mesmo array
    que o MRIcro mostra e o que entra na rede (consistencia treino/inferencia)."""
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
    return Volume(data=data, spacing=zooms, axcodes=axcodes, path=path)


def load_mask(path: str) -> Volume:
    vol = load_nii(path)
    vol.data = (vol.data > 0.5).astype(np.float32)
    return vol


# --------------------------------------------------------------------------
# Normalizacao e orientacao
# --------------------------------------------------------------------------
def normalize_volume(data: np.ndarray, p_low=NORM_P_LOW, p_high=NORM_P_HIGH) -> np.ndarray:
    """Recorte por percentis + escala para [0,1], por volume."""
    lo, hi = np.percentile(data, [p_low, p_high])
    if hi - lo < 1e-8:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def orient_slice(sl: np.ndarray, orientation: str = "T_lower", flip_lr: bool = False):
    """Prepara um corte 2D (X,Y) para exibicao. Retorna (imagem, origin).
    Usado SO para exibir - as redes recebem o corte cru (X,Y)."""
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
    """Aplica a MESMA transformacao de exibicao a mapas (Grad-CAM, mascara)."""
    return orient_slice(sl, orientation, flip_lr)[0]


# --------------------------------------------------------------------------
# Indexacao do dataset
# --------------------------------------------------------------------------
@dataclass
class Case:
    pid: str
    label: Optional[int] = None
    split: Optional[str] = None
    phases: Dict[str, str] = field(default_factory=dict)   # fase -> caminho
    mask: Optional[str] = None
    clinical: dict = field(default_factory=dict)

    def complete(self, need_mask=False) -> bool:
        ok = all(p in self.phases for p in PHASES)
        return ok and (self.mask is not None or not need_mask)


def _pick_col(columns, candidates):
    low = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def _norm_split(v):
    s = str(v).strip().lower()
    for key, aliases in SPLIT_ALIASES.items():
        if s in aliases:
            return key
    return s


def _classify_file(fname: str):
    """Retorna ('mask', None) | ('phase', nome_da_fase) | (None, None)."""
    low = fname.lower()
    if any(p in low for p in MASK_PATTERNS):
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
    """Associa paciente -> arquivos das 3 fases + mascara (+ rotulo/split do CSV)."""

    def __init__(self, root: str, csv_path: Optional[str] = None):
        self.root = root
        self.cases: Dict[str, Case] = {}
        self.csv_info = {}
        self._build(csv_path)

    # ---- construcao ------------------------------------------------------
    def _build(self, csv_path):
        files = scan_nii_files(self.root)
        csv_path = csv_path or self._find_csv()
        table = self._read_csv(csv_path) if csv_path else {}
        pids_csv = list(table.keys())

        for f in files:
            kind, phase = _classify_file(os.path.basename(f))
            if kind is None:
                continue
            pid = self._match_pid(f, pids_csv)
            case = self.cases.setdefault(pid, Case(pid=pid))
            if kind == "mask":
                case.mask = f
            else:
                case.phases[phase] = f

        for pid, row in table.items():
            if pid in self.cases:
                c = self.cases[pid]
                c.label, c.split, c.clinical = row["label"], row["split"], row["clinical"]

    def _find_csv(self):
        for dp, _dn, fns in os.walk(self.root):
            for fn in sorted(fns):
                if fn.lower().endswith(".csv"):
                    return os.path.join(dp, fn)
        return None

    def _read_csv(self, path):
        import pandas as pd
        df = pd.read_csv(path)
        idc = _pick_col(df.columns, CSV_ID_COLS)
        lbc = _pick_col(df.columns, CSV_LABEL_COLS)
        spc = _pick_col(df.columns, CSV_SPLIT_COLS)
        self.csv_info = {"path": path, "id": idc, "label": lbc, "split": spc,
                         "columns": list(df.columns)}
        if idc is None:
            return {}
        table = {}
        for _, r in df.iterrows():
            pid = str(r[idc]).strip()
            label = None
            if lbc is not None and not _isnan(r[lbc]):
                label = int(float(r[lbc]))
            split = _norm_split(r[spc]) if spc is not None and not _isnan(r[spc]) else None
            table[pid] = {"label": label, "split": split,
                          "clinical": {k: r[k] for k in df.columns if k not in (idc, lbc, spc)}}
        return table

    @staticmethod
    def _match_pid(filepath, pids_csv):
        """Escolhe o ID do paciente: maior ID do CSV contido no caminho; senao
        usa o nome do arquivo sem sufixo de fase/mascara."""
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

    # ---- consultas -------------------------------------------------------
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
                f"splits: {splits} | csv: {self.csv_info.get('path', 'nenhum')}")


def _isnan(v):
    try:
        return v is None or (isinstance(v, float) and np.isnan(v))
    except Exception:
        return False


# --------------------------------------------------------------------------
# Carregamento de um caso completo
# --------------------------------------------------------------------------
@dataclass
class LoadedCase:
    pid: str
    label: Optional[int]
    vols: Dict[str, np.ndarray]        # fase -> (X,Y,Z) normalizado [0,1]
    raw: Dict[str, np.ndarray]         # fase -> (X,Y,Z) intensidade original
    mask: Optional[np.ndarray]         # (X,Y,Z) binaria ou None
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
