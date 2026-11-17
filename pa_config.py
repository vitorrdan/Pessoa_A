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
