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

import pa_config
from pa_config import (PHASES, PHASE_PATTERNS, MASK_PATTERNS, CSV_ID_COLS,
                       CSV_LABEL_COLS, CSV_SPLIT_COLS, SPLIT_ALIASES,
                       NORM_P_LOW, NORM_P_HIGH, CSV_DATASET_COL, CSV_DATASET_VALUE,
                       SPLIT_NUMERIC_COLS, CSV_PHASE_COLS, CSV_MASK_COLS,
                       EXTRA_CSV_NAMES, SPLITS_CSV_NAME, MASK_DIR_NAMES)


# --------------------------------------------------------------------------
# Leitura de volumes
# --------------------------------------------------------------------------
@dataclass
class Volume:
    data: np.ndarray                 # (X, Y, Z) float32, ordem de voxels do arquivo
    spacing: tuple = (1.0, 1.0, 1.0)
    axcodes: str = "?"               # orientacao anatomica (ja permutada p/ a ordem X,Y,Z)
    path: str = ""
    z_axis_original: int = 2         # em qual eixo do ARQUIVO estavam os cortes
    shape_original: tuple = ()

    @property
    def shape(self):
        return self.data.shape

    @property
    def n_slices(self):
        return self.data.shape[2]


def detect_z_axis(shape, forced=None) -> int:
    """Eixo dos cortes de um volume 3D. `forced` (0/1/2) vence; senao o eixo cujo
    tamanho difere dos outros dois (que sao iguais, ex.: 256x256 por plano).
    Se nao der para decidir, usa o MENOR eixo - mesmo criterio da Pessoa C
    (pc_data.plane_axis_last), para treino e interface lerem os volumes igual."""
    if forced is not None:
        return int(forced)
    for a in range(3):
        o = [shape[b] for b in range(3) if b != a]
        if o[0] == o[1] and shape[a] != o[0]:
            return a
    return int(np.argmin(shape))


def load_nii(path: str, z_axis=None) -> Volume:
    """Le um .nii/.nii.gz e devolve SEMPRE (X, Y, Z): o eixo dos cortes e movido
    para o fim (ver detect_z_axis / pa_config.Z_AXIS). Volumes 4D (X,Y,Z,1) sao
    achatados; 2D ganha um eixo z de tamanho 1. Nao reamostra nem inverte voxels."""
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
    za = detect_z_axis(shape0, pa_config.Z_AXIS if z_axis is None else z_axis)
    perm = [a for a in range(3) if a != za] + [za]            # (X, Y, Z)
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

    def mask_range(self):
        """(mask_start, mask_end) do CSV: cortes que contem tumor, se a planilha tiver."""
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
    """Retorna ('mask', None) | ('phase', nome_da_fase) | (None, None).
    E mascara se o NOME tiver um padrao de MASK_PATTERNS ou se estiver numa PASTA
    chamada mask/seg/... (ex.: <raiz>/mask/ISPY1_1001_spy1_vis1.nii.gz)."""
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
    """Associa paciente -> arquivos das 3 fases + mascara (+ rotulo/split do CSV)."""

    def __init__(self, root: str, csv_path: Optional[str] = None,
                 splits_csv: Optional[str] = None):
        self.root = root
        self.cases: Dict[str, Case] = {}
        self.csv_info = {}
        self._build(csv_path)
        sp = splits_csv or self._find_splits_csv()
        if sp:
            self._apply_splits(sp)

    # ---- construcao ------------------------------------------------------
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

        # Colunas do CSV (pre/post_early/post_late) dizem exatamente qual arquivo e cada
        # fase e tem prioridade sobre os padroes de nome.
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
        """Planilha de METADADOS: a que tem coluna de id e de pCR e mais colunas
        (ignora splits.csv e split_x_classe.csv)."""
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
        """Particao do grupo (pid, pCR, split). So PREENCHE quem ainda nao tem particao:
        a coluna de particao do CSV oficial (se existir) tem prioridade - mesma regra da
        Pessoa C (pc_data, --splits_csv)."""
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
            if keep.any():                        # so filtra se o valor existe (nao esvazia a tabela)
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
                f"splits: {splits} | csv: {self.csv_info.get('path', 'nenhum')} "
                f"splits.csv: {self.csv_info.get('splits_csv', 'nenhum')} | "
                f"(linhas usadas {self.csv_info.get('rows_used', '?')}/"
                f"{self.csv_info.get('rows_total', '?')}, filtro {self.csv_info.get('dataset_filter')})")


def _file_key(path: str) -> str:
    """Nome do arquivo sem .nii/.nii.gz, em minusculas (chave de busca)."""
    return re.sub(r"\.nii(\.gz)?$", "", os.path.basename(str(path)).strip(), flags=re.I).lower()


def _resolve_ref(ref, root, by_key):
    """Valor de uma celula do CSV (nome/caminho de arquivo) -> caminho real, ou None."""
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
    hits = [p for k, v in by_key.items() if key in k for p in v]     # busca por trecho do nome
    return hits[0] if len(hits) == 1 else None


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
