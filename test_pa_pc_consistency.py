# -*- coding: utf-8 -*-
"""PESSOA A - Confere se o treino da Pessoa C e a interface (A) leem os dados IGUAL.
Rodar na pasta onde estao pa_*.py E pc_*.py:   python test_pa_pc_consistency.py

ATENCAO: escrito SEM ter visto pc_data.py - assume as funcoes citadas no README_C:
  pc_data.normalize_percentile(vol) e pc_data.plane_axis_last(vol).
Se a assinatura for outra, o teste avisa 'PULADO' e diz o que ajustar (nao falha em silencio)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pa_nii_io import load_nii, normalize_volume


def _try(fn, *args):
    try:
        return fn(*args)
    except TypeError as e:
        return ("PULADO", str(e))


def main():
    import nibabel as nib
    try:
        import pc_data
    except Exception as e:
        print("pc_data nao importou:", type(e).__name__, e)
        print("-> coloque pc_data.py na mesma pasta (e instale as dependencias dele).")
        return 1
    rng = np.random.default_rng(0)
    problemas = 0

    # 1) eixo dos cortes: mesmo resultado que load_nii para volumes (49,256,256)-like
    vol = rng.random((9, 40, 40)).astype(np.float32) * 300
    f = os.path.join(tempfile.mkdtemp(), "v.nii.gz")
    nib.save(nib.Nifti1Image(vol, np.eye(4)), f)
    a = load_nii(f).data
    if hasattr(pc_data, "plane_axis_last"):
        c = _try(pc_data.plane_axis_last, vol)
        if isinstance(c, tuple) and c and c[0] == "PULADO":
            print("PULADO eixo:", c[1])
        elif np.shape(c) == a.shape and np.allclose(c, a):
            print("OK   eixo dos cortes: pc_data.plane_axis_last == pa_nii_io.load_nii", a.shape)
        else:
            problemas += 1
            print("DIFERENTE eixo:", np.shape(c), "vs", a.shape)
    else:
        print("PULADO eixo: pc_data nao tem plane_axis_last")

    # 2) normalizacao de intensidade (por volume/fase), com fundo zero como nos dados reais
    v = rng.gamma(2.0, 40.0, (40, 40, 9)).astype(np.float32)
    v[:6] = 0
    mine = normalize_volume(v)
    if hasattr(pc_data, "normalize_percentile"):
        c = _try(pc_data.normalize_percentile, v)
        if isinstance(c, tuple) and c and c[0] == "PULADO":
            print("PULADO normalizacao:", c[1])
        elif np.shape(c) == mine.shape and np.allclose(c, mine, atol=1e-5):
            print("OK   normalizacao: pc_data.normalize_percentile == pa_nii_io.normalize_volume")
        else:
            problemas += 1
            d = float(np.abs(np.asarray(c) - mine).max()) if np.shape(c) == mine.shape else float("nan")
            print(f"DIFERENTE normalizacao (max |dif| = {d:.4f}).\n"
                  "  Treino (C) e interface (A) veriam entradas diferentes: alinhar as duas funcoes\n"
                  "  (percentis, se o calculo e por volume ou so no primeiro plano, clip em [0,1]).")
    else:
        print("PULADO normalizacao: pc_data nao tem normalize_percentile")

    print("\n" + ("TUDO CONSISTENTE" if problemas == 0 else f"{problemas} DIVERGENCIA(S) - corrigir antes de treinar/apresentar"))
    return problemas


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
