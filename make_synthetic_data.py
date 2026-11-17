# -*- coding: utf-8 -*-
"""PESSOA A - Gera um mini-dataset SINTETICO (so p/ testar a interface/pipeline,
nao tem valor clinico).

layout="plain"    : <PID>_0000/_0001/_0002.nii.gz (z no ultimo eixo), <PID>_seg.nii.gz,
                    dataset.csv (patient_id, pcr, split com texto train/val/test).
layout="official": imita o que se sabe do BreastDCEDL real: z no eixo do MEIO
                    (shape X,Z,Y), arquivos <PID>_aqc_0/1/2, CSV com colunas
                    pid, dataset (spy1 + linhas de outro dataset sem arquivos), pcr,
                    test (0=treino, 1=teste, 2=validacao), pre, post_early, post_late,
                    mask_start, mask_end.
Uso: python make_synthetic_data.py ./sintetico [n] [tamanho] [plain|official|meta]"""
import os
import sys

import numpy as np


def _case(rng, size, nz, pcr):
    yy, xx = np.mgrid[:size, :size]
    cx, cy = rng.integers(size // 3, 2 * size // 3, 2)
    r0 = size * (rng.uniform(0.12, 0.15) if pcr else rng.uniform(0.19, 0.24))   # pCR = tumor menor
    mask = np.zeros((size, size, nz), np.float32)
    for z in range(nz):
        rz = r0 * max(0.0, 1 - abs(z - nz / 2) / (nz / 2 + 0.5))
        mask[:, :, z] = ((xx - cx) ** 2 + (yy - cy) ** 2 <= rz ** 2)
    bg = rng.normal(100, 3, (size, size, nz)).astype(np.float32)
    gain = 0.6 if pcr else 1.6                                                   # pCR = realce mais fraco
    return [bg + 20 * mask, bg + 150 * gain * mask, bg + 90 * gain * mask], mask


def make(out, n=16, size=64, nz=8, seed=0, layout="plain", phase_cols=True):
    import nibabel as nib
    import pandas as pd
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    aff = np.diag([0.7, 0.7, 2.0, 1.0])
    rows = []
    for i in range(n):
        pid = f"P{i + 1:03d}" if layout == "plain" else f"ISPY1_{1000 + i}"
        pcr = int(i % 3 == 0)                      # ~1/3 positivos (desbalanceado)
        phases, mask = _case(rng, size, nz, pcr)
        if layout == "plain":
            names = [f"{pid}_000{k}.nii.gz" for k in range(3)]
            mname = f"{pid}_seg.nii.gz"
            save = lambda a: a
        elif layout == "meta":
            os.makedirs(os.path.join(out, "dce"), exist_ok=True)
            os.makedirs(os.path.join(out, "mask"), exist_ok=True)
            names = [os.path.join("dce", f"{pid}_spy1_vis1_dce_aqc_{k}.nii.gz") for k in range(3)]
            mname = os.path.join("mask", f"{pid}_spy1_vis1_mask.nii.gz")
            save = lambda a: np.ascontiguousarray(np.transpose(a, (2, 0, 1)))     # (X,Y,Z)->(Z,X,Y)
        else:
            names = [f"{pid}_aqc_{k}.nii.gz" for k in range(3)]
            mname = f"{pid}_mask.nii.gz"
            save = lambda a: np.ascontiguousarray(np.transpose(a, (0, 2, 1)))     # (X,Y,Z)->(X,Z,Y)
        for nm, vol in zip(names, phases):
            nib.save(nib.Nifti1Image(save(vol).astype(np.float32), aff), os.path.join(out, nm))
        nib.save(nib.Nifti1Image(save(mask), aff), os.path.join(out, mname))
        zs = np.where(mask.sum(axis=(0, 1)) > 0)[0]
        if layout == "plain":
            split = "train" if i < n * 0.6 else ("val" if i < n * 0.8 else "test")
            rows.append((pid, pcr, split))
        elif layout == "meta":
            split = "train" if i < n * 0.6 else ("val" if i < n * 0.8 else "test")
            rows.append({"pid": pid, "age": 40.0 + i, "pCR": float(pcr), "dataset": "duke",
                         "mask_max_plane": float(np.argmax(mask.sum(axis=(0, 1)))),
                         "voi_start_z": float(zs.min()), "voi_end_z": float(zs.max()), "_split": split})
        else:
            code = 0 if i < n * 0.6 else (2 if i < n * 0.8 else 1)      # 0 treino, 2 validacao, 1 teste
            rows.append({"pid": pid, "dataset": "spy1", "pcr": pcr, "test": code,
                         "pre": names[0], "post_early": names[1], "post_late": names[2],
                         "mask_start": int(zs.min()), "mask_end": int(zs.max())})
    if layout == "plain":
        pd.DataFrame(rows, columns=["patient_id", "pcr", "split"]).to_csv(os.path.join(out, "dataset.csv"), index=False)
    elif layout == "meta":
        df = pd.DataFrame(rows)
        os.makedirs(os.path.join(out, "cache"), exist_ok=True)
        df[["pid", "pCR", "_split"]].rename(columns={"_split": "split"}).assign(
            pCR=lambda d: d["pCR"].astype(int)).to_csv(os.path.join(out, "cache", "splits.csv"), index=False)
        pd.crosstab(df["_split"], df["pCR"], margins=True).to_csv(os.path.join(out, "cache", "split_x_classe.csv"))
        extra = pd.DataFrame([{"pid": f"ISPY1_9{j}", "age": 50.0, "pCR": np.nan, "dataset": "duke"} for j in range(2)])
        pd.concat([df.drop(columns=["_split"]), extra]).to_csv(
            os.path.join(out, "BreastDCEDL_spy1_metadata.csv"), index=False)
    else:
        for j in range(5):                         # pacientes de OUTRO dataset: devem ser filtrados
            rows.append({"pid": f"DUKE_{j}", "dataset": "duke", "pcr": j % 2, "test": 0,
                         "pre": f"DUKE_{j}_pre", "post_early": f"DUKE_{j}_e", "post_late": f"DUKE_{j}_l",
                         "mask_start": 0, "mask_end": 1})
        df = pd.DataFrame(rows)
        if not phase_cols:
            df = df.drop(columns=["pre", "post_early", "post_late"])
        df.to_csv(os.path.join(out, "dataset.csv"), index=False)
    return out


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "./sintetico"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    layout = sys.argv[4] if len(sys.argv) > 4 else "plain"
    print("Gerado em", make(out, n, size, layout=layout))
