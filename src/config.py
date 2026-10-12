"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = RESULTS / "models"

RAW_FILE = DATA_RAW / "Fe-HCl-317_dataset.xlsx"
RAW_URL = (
    "https://raw.githubusercontent.com/pth-lab/Fe-HCl-317-dataset/"
    "main/Fe-HCl-317_dataset.xlsx"
)
CLEAN_FILE = DATA_PROCESSED / "fe_hcl_clean.csv"

SEED = 42
N_SPLITS = 10
N_REPEATS = 3
TARGET = "IE"  # inhibition efficiency, %

# Light-surface palette: first three slots of a validated categorical order
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
