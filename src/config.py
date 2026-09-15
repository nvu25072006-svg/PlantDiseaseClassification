import os
import random
import sys
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

try:
    import torch
except ImportError:
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VENDOR_TIMM_DIR = PROJECT_ROOT / "vendor" / "pytorch-image-models-main"
if VENDOR_TIMM_DIR.exists() and str(VENDOR_TIMM_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_TIMM_DIR))

FULL_PLANTVILLAGE_DIR = (
    PROJECT_ROOT / "data" / "PlantVillage_extracted" / "PlantVillage-Dataset-master" / "raw" / "color"
)
DATA_DIR = FULL_PLANTVILLAGE_DIR if FULL_PLANTVILLAGE_DIR.exists() else (PROJECT_ROOT / "data" / "PlantVillage")
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BATCH_SIZE = 32
NUM_WORKERS = 2
DEFAULT_LR_FROZEN = 1e-3
DEFAULT_LR_FINETUNE = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 4

SAMPLE_CLASSES = [
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
]

VIETNAMESE_LABELS = {
    "Tomato___Early_blight": "Cà chua - Bệnh đốm vòng (Early Blight)",
    "Tomato___Late_blight": "Cà chua - Bệnh mốc sương (Late Blight)",
    "Tomato___healthy": "Cà chua - Lá khỏe mạnh (Healthy)",
    "Tomato___Healthy": "Cà chua - Lá khỏe mạnh (Healthy)",
    "Potato___Early_blight": "Khoai tây - Bệnh đốm vòng (Early Blight)",
    "Potato___Late_blight": "Khoai tây - Bệnh mốc sương (Late Blight)",
    "Potato___healthy": "Khoai tây - Lá khỏe mạnh (Healthy)",
    "Potato___Healthy": "Khoai tây - Lá khỏe mạnh (Healthy)",
    "Pepper,_bell___Bacterial_spot": "Ớt chuông - Bệnh đốm vi khuẩn (Bacterial Spot)",
    "Pepper__bell___Bacterial_spot": "Ớt chuông - Bệnh đốm vi khuẩn (Bacterial Spot)",
    "Pepper,_bell___healthy": "Ớt chuông - Lá khỏe mạnh (Healthy)",
    "Pepper__bell___Healthy": "Ớt chuông - Lá khỏe mạnh (Healthy)",
}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
