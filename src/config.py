"""Central configuration: paths and tunable constants.

All paths are relative to the project root (the folder that contains ``src/``),
so the project can be moved to any machine. Environment variables allow
overriding the defaults without editing code:

    ER_LIMIT       number of records processed per source (default 10000)
    ER_DATA_DIR    folder containing ``train/`` and ``test/`` (default <project>/dataset)
    ER_OUTPUT_DIR  folder for the output files
    ER_MODEL_PATH  location of the pickled model (default <project>/model.pkl)
"""
import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

# --- record limit (instruction: work with the FIRST 10,000 records only) ---
LIMIT = int(os.environ.get("ER_LIMIT", "10000"))

_SUBMISSION_DIR = PROJECT_DIR.parent   # e:\Elite_Warriors_Submission

DATA_DIR = Path(os.environ.get("ER_DATA_DIR",   _SUBMISSION_DIR / "dataset"))
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"
MODEL_PATH = Path(os.environ.get("ER_MODEL_PATH", PROJECT_DIR / "model.pkl"))


def _resolve_output_dir() -> Path:
    env = os.environ.get("ER_OUTPUT_DIR")
    if env:
        return Path(env)
    # prefer the submission-level output/ that already has the generated files
    submission_out = _SUBMISSION_DIR / "output"
    if (submission_out / "matching_results.tsv").exists():
        return submission_out
    local = PROJECT_DIR / "output"
    if (local / "matching_results.tsv").exists():
        return local
    return submission_out


OUTPUT_DIR = _resolve_output_dir()
MATCHING_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATES_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"

# --- blocking (candidate generation) ---
CHUNK_SIZE = 500          # Source 1 rows scored per dense block
TOPK_NAME_CHAR = 15       # top-k by name char n-gram cosine
TOPK_NAME_WORD = 10       # top-k by IDF-weighted name token cosine (rare tokens)
TOPK_NAME_SQUASH = 8      # top-k by space-insensitive name cosine
TOPK_ADDRESS = 10         # top-k by address char n-gram cosine
TOPK_COMBINED = 15        # top-k by combined name+address score
MIN_NAME_SIM = 0.15
MIN_ADDRESS_SIM = 0.30
MIN_COMBINED_SIM = 0.20

# --- training ---
VALIDATION_FRACTION = 0.20
RANDOM_STATE = 42
THRESHOLD_GRID_START = 0.05
THRESHOLD_GRID_END = 0.96
THRESHOLD_GRID_STEP = 0.01
F_BETA = 0.5

# Previously recorded baseline (random pair split, 5 features) for comparison only.
BASELINE = {"precision": 0.8571, "recall": 0.5000, "f0_5": 0.7500, "threshold": 0.77,
            "non_empty_matches": 85}
