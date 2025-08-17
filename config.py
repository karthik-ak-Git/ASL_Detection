from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
OUTPUTS_DIR = ROOT / "outputs"
LOGS_DIR = ROOT / "logs"
FRONTEND_DIR = ROOT / "frontend"
MODEL_PATH = OUTPUTS_DIR / "best_model.pth"
CLASS_TO_IDX_PATH = OUTPUTS_DIR / "class_to_idx.json"
IDX_TO_CLASS_PATH = OUTPUTS_DIR / "idx_to_class.json"
API_TITLE = "ASL Detection API"
API_VERSION = "1.0.0"
DEFAULT_DEVICE = "cuda"  # fallback to cpu if not available
