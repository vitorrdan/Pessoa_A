# -*- coding: utf-8 -*-
"""PESSOA A - Gera um mini-dataset SINTETICO (so p/ testar a interface/pipeline,
nao tem valor clinico). Estrutura: <pasta>/<PID>_0000.nii.gz (pre), _0001 (precoce),
_0002 (tardio), <PID>_seg.nii.gz e dataset.csv (patient_id, pcr, split).
Uso: python make_synthetic_data.py ./sintetico [n_pacientes] [tamanho]"""
import os
import sys

import numpy as np


def make(out, n=16, size=64, nz=8, seed=0):
    import nibabel as nib
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    aff = np.diag([0.7, 0.7, 2.0, 1.0])
    yy, xx = np.mgrid[:size, :size]
    rows = []
    for i in range(n):
        pid = f"P{i + 1:03d}"
        pcr = int(i % 3 == 0)                      # ~1/3 positivos (desbalanceado)
        cx, cy = rng.integers(size // 3, 2 * size // 3, 2)
        r0 = size * (rng.uniform(0.12, 0.15) if pcr else rng.uniform(0.19, 0.24))   # pCR = tumor menor
        mask = np.zeros((size, size, nz), np.float32)
        for z in range(nz):
            rz = r0 * max(0.0, 1 - abs(z - nz / 2) / (nz / 2 + 0.5))
            mask[:, :, z] = ((xx - cx) ** 2 + (yy - cy) ** 2 <= rz ** 2)
        bg = rng.normal(100, 3, (size, size, nz)).astype(np.float32)
        gain = (0.6 if pcr else 1.6)                # pCR = realce mais fraco
        phases = [bg + 20 * mask,
                  bg + 150 * gain * mask,
                  bg + 90 * gain * mask]
        for k, vol in enumerate(phases):
            nib.save(nib.Nifti1Image(vol.astype(np.float32), aff), os.path.join(out, f"{pid}_000{k}.nii.gz"))
        nib.save(nib.Nifti1Image(mask, aff), os.path.join(out, f"{pid}_seg.nii.gz"))
        split = "train" if i < n * 0.6 else ("val" if i < n * 0.8 else "test")
        rows.append((pid, pcr, split))
    import pandas as pd
    pd.DataFrame(rows, columns=["patient_id", "pcr", "split"]).to_csv(os.path.join(out, "dataset.csv"), index=False)
    return out


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "./sintetico"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    print("Gerado em", make(out, n, size))
