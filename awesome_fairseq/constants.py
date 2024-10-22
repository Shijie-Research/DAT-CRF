import os
from pathlib import Path

# set the default cache root for huggingface packages
os.environ["HF_HOME"] = "~/Data/.cache"

# FLAGS
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}

DEBUG_MODE = os.environ.get("DEBUG_MODE", "").upper() in ENV_VARS_TRUE_VALUES
WANDB_DISABLED = os.environ.get("WANDB_DISABLED", "").upper() in ENV_VARS_TRUE_VALUES

# PATHS
PROJECT_DIR = Path(__file__).resolve().parents[1].as_posix()
EXPERIMENT_DIR = Path(PROJECT_DIR, "experiments").as_posix()
PLUGINS_DIR = Path(PROJECT_DIR, "fairseq_plugins").as_posix()
