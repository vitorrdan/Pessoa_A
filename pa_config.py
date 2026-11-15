# -*- coding: utf-8 -*-
"""
PESSOA A - Configuracao central (interface / explicabilidade).

[Nome, matricula, curso e campus dos componentes do grupo: preencher na integracao]

Tudo que depende de como o dataset BreastDCEDL esta organizado em disco fica
AQUI, para ser ajustado em um unico lugar depois do download.
"""

# --- Fases temporais (ordem = ordem dos canais na fusao) -------------------
PHASES = ["pre", "early", "late"]          # pre-contraste, pos precoce, pos tardio
PHASE_LABELS = {
    "pre": "Pre-contraste",
    "early": "Pos-contraste precoce",
    "late": "Pos-contraste tardio",
}

# --- Padroes para identificar a fase pelo nome do arquivo .nii -------------
# Ajustar apos inspecionar o dataset real (ex.: nnU-Net usa _0000/_0001/_0002).
# A primeira regra que casar vence. Comparacao em minusculas.
PHASE_PATTERNS = {
    "pre":   ["_0000", "_pre", "pre_", "phase0", "ph0", "_t0"],
    "early": ["_0001", "_early", "_post1", "phase1", "ph1", "_t1"],
    "late":  ["_0002", "_late", "_post2", "phase2", "ph2", "_t2"],
}
# Palavras que marcam um arquivo como mascara de segmentacao.
MASK_PATTERNS = ["seg", "mask", "label", "tumor"]

# --- Colunas do CSV (nomes candidatos, em minusculas) ----------------------
CSV_ID_COLS = ["patient_id", "pid", "id", "case", "case_id", "subject", "patient"]
CSV_LABEL_COLS = ["pcr", "pcr_label", "label", "class", "target", "y"]
CSV_SPLIT_COLS = ["split", "partition", "set", "subset", "fold_set", "dataset"]
# Como os valores de split aparecem no CSV (comparacao em minusculas).
SPLIT_ALIASES = {
    "train": ["train", "training", "treino", "tr"],
    "val":   ["val", "valid", "validation", "validacao"],
    "test":  ["test", "testing", "teste", "ts"],
}

# --- Aumento de dados (enunciado, item b) ----------------------------------
AUG_ANGLES = [-20, -10, 0, 10, 20]   # x (original, espelhada) = 10 variacoes

# --- Visualizacao ----------------------------------------------------------
# Orientacao da exibicao do corte. Opcoes: "T_lower" (transposto, origem
# embaixo; padrao nibabel/matplotlib), "rot90", "none".
# Validar contra o MRIcro/ITK-SNAP e ajustar na aba Visualizar.
DEFAULT_ORIENTATION = "T_lower"
DEFAULT_FLIP_LR = False

# --- Modelo / inferencia ---------------------------------------------------
INPUT_SIZE = 256
DEFAULT_THRESHOLD = 0.5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Percentis usados na normalizacao de intensidade por volume.
NORM_P_LOW, NORM_P_HIGH = 1.0, 99.0

# --- Janela ----------------------------------------------------------------
APP_TITLE = "DCE-MRI | Resposta ao tratamento (pCR) - Interface e Grad-CAM"
