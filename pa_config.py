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
    "pre": "Pré-contraste",
    "early": "Pós-contraste precoce",
    "late": "Pós-contraste tardio",
}

# --- Padroes para identificar a fase pelo nome do arquivo .nii -------------
# Ajustar apos inspecionar o dataset real (ex.: nnU-Net usa _0000/_0001/_0002).
# A primeira regra que casar vence. Comparacao em minusculas.
# Usados quando o CSV NAO traz as colunas das fases (ver CSV_PHASE_COLS).
PHASE_PATTERNS = {
    "pre":   ["_aqc_0", "_0000", "_pre", "pre_", "phase0", "ph0", "_t0"],
    "early": ["_aqc_1", "_0001", "_early", "_post1", "phase1", "ph1", "_t1"],
    "late":  ["_aqc_2", "_0002", "_late", "_post2", "phase2", "ph2", "_t2"],
}
# Palavras que marcam um arquivo como mascara de segmentacao.
MASK_PATTERNS = ["seg", "mask", "label", "tumor"]

# --- Colunas do CSV (nomes candidatos, em minusculas) ----------------------
CSV_ID_COLS = ["patient_id", "pid", "id", "case", "case_id", "subject", "patient"]
CSV_LABEL_COLS = ["pcr", "pcr_label", "label", "class", "target", "y"]
CSV_SPLIT_COLS = ["split", "partition", "subset", "set", "test", "fold_set"]
# CSV real do BreastDCEDL: filtrar so o I-SPY1 (172 pacientes).
CSV_DATASET_COL = "dataset"
CSV_DATASET_VALUE = "spy1"          # None = nao filtrar
# Colunas numericas de particao. No CSV real a coluna "test" e: 0=treino, 1=teste, 2=validacao
# (ATENCAO: 1 e teste e 2 e validacao - nao trocar!).
SPLIT_NUMERIC_COLS = {"test": {0: "train", 1: "test", 2: "val"}}
# Colunas do CSV que dizem QUAL arquivo e cada fase (nao assumir acq0/acq1/acq2).
CSV_PHASE_COLS = {"pre": "pre", "early": "post_early", "late": "post_late"}
CSV_MASK_COLS = ["mask", "seg", "segmentation", "mask_file", "mask_path"]

# CSVs auxiliares que NAO sao os metadados (nao escolher como planilha principal).
EXTRA_CSV_NAMES = ("splits.csv", "split_x_classe.csv")
# Particao compartilhada pelo grupo (colunas pid, pCR, split). Se existir em qualquer
# subpasta, tem prioridade sobre a coluna de particao do CSV de metadados.
SPLITS_CSV_NAME = "splits.csv"
# Pastas cujo nome indica que os arquivos dentro sao mascaras de segmentacao.
MASK_DIR_NAMES = ["mask", "masks", "seg", "segmentation", "label", "labels"]

# --- Eixo dos cortes (z) ---------------------------------------------------
# O array do .nii pode vir como (256,256,49), (256,49,256) ou (49,256,256).
# None = detectar: o eixo cujo tamanho difere dos outros dois e o z. Ou fixar 0/1/2.
# Depois da leitura o codigo inteiro trabalha SEMPRE com (X, Y, Z).
Z_AXIS = None
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
