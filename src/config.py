"""
Central configuration: paths and constants. Small and explicit on purpose so the
whole pipeline's I/O is reviewable in one place.

Design note on storage: the shared netapp home volume is chronically full, so all
HEAVY outputs (graphs, checkpoints, per-run state) go to WORK_DIR on the machine's
LOCAL disk. Only code and small human-readable results live in the repo (REPO_DIR).
"""
from pathlib import Path

# --- environment (graph-tool 3.0 lives on local disk; see README) ---
ENV_PYTHON = "/var/tmp/gt3/bin/python"

# --- inputs: FlyWire codex_2025 ---
DATA_DIR = Path("/home/gamir/maorhury/Projects/Research_Project_2026/Data/network/codex_2025")
CONNECTIONS = DATA_DIR / "connections_princeton.csv.gz"   # pre_root_id, post_root_id, neuropil, syn_count, nt_type
CLASSIFICATION = DATA_DIR / "classification.csv.gz"       # root_id, flow, super_class, class, sub_class, side, ...
CELL_TYPES = DATA_DIR / "consolidated_cell_types.csv.gz"  # root_id, primary_type, additional_type(s)
NEURONS = DATA_DIR / "neurons.csv.gz"                      # root_id, nt_type, ...

# --- outputs ---
REPO_DIR = Path(__file__).resolve().parent.parent          # code + PLAN.md + RESULTS.md (netapp, small)
WORK_DIR = Path("/var/tmp/csbm_work")                       # heavy: graphs, checkpoints, run state (local disk)
RUNS_DIR = WORK_DIR / "runs"                                # one subdir per fit; holds progress.jsonl + state
GRAPH_CACHE = WORK_DIR / "graphs"                           # cached built graphs

# --- label hierarchy (coarse -> fine), evaluation-only, withheld from fitting ---
LABEL_LEVELS = ["super_class", "class", "sub_class", "primary_type"]
EXTRA_LABELS = ["side", "nt_type"]                          # side -> symmetry; nt_type -> orthogonal check

# --- defaults ---
DEFAULT_THRESHOLD = 1        # canonical: >=1 synapse (sensitivity variant: >=5)
DEFAULT_DIRECTED = True
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]

for d in (WORK_DIR, RUNS_DIR, GRAPH_CACHE):
    d.mkdir(parents=True, exist_ok=True)
