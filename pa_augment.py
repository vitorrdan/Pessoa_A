# -*- coding: utf-8 -*-
"""
PESSOA A - Aumento de dados (item b) e analise de balanceamento.

Estas funcoes servem para a ABA de visualizacao e para o grafico de classes.
A Pessoa B pode importar `augment_variants` no treino (ou usar a propria
implementacao - a assinatura e simples de substituir).

Biblioteca usada: scipy.ndimage.rotate (funcao elementar)
  - reshape=False : mantem 256x256 (cantos ficam preenchidos)
  - order=1       : interpolacao bilinear (suave, sem overshoot do cubico)
  - mode='constant', cval=0 : fundo preto, coerente com o fundo da MRI
"""
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import ndimage

from pa_config import AUG_ANGLES


def rotate_img(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotaciona um corte (H,W) ou (H,W,C) em torno do centro."""
    if angle == 0:
        return img.copy()
    return ndimage.rotate(img, angle, axes=(0, 1), reshape=False, order=1,
                          mode="constant", cval=0.0)


def augment_variants(img: np.ndarray, angles: Sequence[float] = AUG_ANGLES
                     ) -> List[Tuple[str, np.ndarray]]:
    """Gera as 10 variacoes do enunciado: (angulos) x (original, espelhada).
    Retorna lista [(rotulo, imagem)]. A ordem e: originais e depois espelhadas.
    Para imagens multicanal (H,W,C) rotaciona/espelha todos os canais juntos."""
    out = []
    for mirrored in (False, True):
        base = img[:, ::-1].copy() if mirrored else img
        for a in angles:
            tag = f"{a:+d}°" + (" espelhada" if mirrored else " original")
            out.append((tag, rotate_img(base, a)))
    return out


def augment_mask_like(mask: np.ndarray, angle: float, mirrored: bool) -> np.ndarray:
    """Mesma transformacao para a mascara (interpolacao por vizinho mais proximo)."""
    m = mask[:, ::-1] if mirrored else mask
    if angle == 0:
        return m.copy()
    return ndimage.rotate(m, angle, axes=(0, 1), reshape=False, order=0,
                          mode="constant", cval=0.0)


# --------------------------------------------------------------------------
# Balanceamento
# --------------------------------------------------------------------------
def balance_plan(counts: Dict[int, int], n_variants: int = 10,
                 relevant_ratio: float = 1.5) -> Dict:
    """Analisa o desbalanceamento e calcula as estrategias.

    counts: {0: n_sem_pCR, 1: n_com_pCR} nos dados de TREINO/VALIDACAO.
    Retorna dicionario com:
      ratio           : maioria/minoria
      relevant        : ratio >= relevant_ratio  (limiar de decisao, ajustavel)
      before          : contagem original
      after_aug       : contagem apos 10 variacoes por imagem (nao muda a razao)
      after_prop      : contagem apos aumento PROPORCIONAL (minoria recebe mais variacoes)
      variants_per_class : variacoes por imagem em cada classe no aumento proporcional
      class_weights   : pesos p/ a funcao de perda (n_total / (2 * n_classe))
      pos_weight      : n_neg / n_pos (p/ BCEWithLogitsLoss)
    """
    n0, n1 = int(counts.get(0, 0)), int(counts.get(1, 0))
    if n0 == 0 or n1 == 0:
        raise ValueError("Uma das classes nao tem exemplos - verifique o CSV.")
    maj_cls = 0 if n0 >= n1 else 1
    n_maj, n_min = max(n0, n1), min(n0, n1)
    ratio = n_maj / n_min

    v_maj = n_variants
    v_min = int(math.ceil(n_variants * ratio))        # equaliza o total por classe
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
    """Angulos adicionais (alem de -20..20 de 10 em 10) para o aumento
    proporcional da classe minoritaria: pontos intermediarios (+-5, +-15, ...)
    dentro do mesmo intervalo de +-20 graus."""
    pool = [-15, -5, 5, 15, -17.5, -12.5, -7.5, -2.5, 2.5, 7.5, 12.5, 17.5]
    return pool[:max(0, n_extra)]
