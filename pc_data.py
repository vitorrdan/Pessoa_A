"""
PESSOA C - Dados para treino/avaliacao (leitura, cache, cortes, agregacao, limiar).

[Nome, matricula, curso e campus dos componentes do grupo: preencher na integracao]

Decisoes importantes (conferir com A e B):
  * NORMALIZACAO = percentil 1-99 por volume e por fase -> [0,1]  (identica a
    pa_nii_io.normalize_volume, usada pela GUI na inferencia). Assim o que a rede ve no
    treino e o que a interface entrega em "Classificar" sao exatamente a mesma coisa.
  * EIXO DOS CORTES: nos arquivos reais (cache da Pessoa B: fatia 49x256) o eixo dos
    planos e o MENOR dos tres. `plane_axis_last` o move para o fim, entao um corte e
    sempre vol[:, :, z] com 256x256.  (A e B devem aplicar o mesmo - ver README_C.)
  * Cache em disco (fora do ZIP): <cache>/<pid>_vol.npy  (Z,3,X,Y) float16
                                  <cache>/<pid>_mask.npy (Z,X,Y)   uint8
                                  <cache>/index.csv
    abertos com mmap -> a bateria de experimentos nao rele os .nii.
"""
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from pa_config import AUG_ANGLES, NORM_P_HIGH, NORM_P_LOW, PHASES
from pa_slices import STRATEGIES


def load_nii_array(path: str) -> np.ndarray:
    """Le .nii/.nii.gz (float32, sem eixos unitarios). Sem reorientar os voxels."""
    import nibabel as nib
    arr = np.squeeze(np.asarray(nib.load(path).dataobj, dtype=np.float32))
    if arr.ndim != 3:
        raise ValueError(f"Volume com dimensoes inesperadas {arr.shape}: {path}")
    return arr


def plane_axis(shape) -> int:
    """Eixo dos planos (z) = o menor dos tres (cortes 256x256, ~20-100 planos)."""
    return int(np.argmin(shape))


def plane_axis_last(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    axis = plane_axis(arr.shape) if axis is None else axis
    return np.moveaxis(arr, axis, -1)


def normalize_percentile(vol, p_low=NORM_P_LOW, p_high=NORM_P_HIGH) -> np.ndarray:
    """Igual a pa_nii_io.normalize_volume."""
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
    """Retorna DataFrame [pid, label, split]. Prioridade para a particao:
    coluna de split do CSV  >  --splits_csv (ex.: cache/splits.csv da Pessoa B)  >
    divisao estratificada 70/15/15 (seed fixa, a mesma logica da Pessoa B)."""
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
    """Localiza as 3 fases e a mascara de um paciente (nomes do BreastDCEDL real
    '<pid>_spy1_vis1_dce_aqc_<0|1|2>.nii.gz' / '<pid>_..._mask.nii.gz', ou o formato
    '_0000/_0001/_0002' + '_seg' do dataset sintetico da Pessoa A)."""
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
    """3 volumes (eixos quaisquer) + mascara -> ((Z,3,X,Y) float16, (Z,X,Y) uint8)."""
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
    """Le os .nii uma unica vez e grava o cache. Retorna o DataFrame de indice."""
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
    """Nenhum paciente pode aparecer em duas particoes."""
    seen = {}
    for sp, lst in patients_by_split.items():
        for p in lst:
            if p.pid in seen and seen[p.pid] != sp:
                raise AssertionError(f"VAZAMENTO: {p.pid} em {seen[p.pid]} e {sp}")
            seen[p.pid] = sp
    return True


def training_slices(areas: np.ndarray, mode: str) -> List[int]:
    """Cortes de treino de um paciente: 'central' | 'max_area' | 'tumor' (todos com tumor)."""
    n = len(areas)
    if mode == "central":
        return [n // 2]
    if mode == "max_area":
        return [int(np.argmax(areas)) if areas.max() > 0 else n // 2]
    tz = np.where(areas > 0)[0].tolist()
    return tz if tz else list(range(n))


def aggregate(per_slice: np.ndarray, areas: np.ndarray, strategy: str) -> float:
    """Probabilidade do paciente a partir da probabilidade de TODOS os cortes."""
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
    """Cortes que precisam de inferencia para uma estrategia (acelera a validacao)."""
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
    """Varre os pontos medios entre probabilidades consecutivas. method: 'f1' ou 'youden'
    (sensibilidade + especificidade - 1). Empates -> o mais proximo de 0.5."""
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
