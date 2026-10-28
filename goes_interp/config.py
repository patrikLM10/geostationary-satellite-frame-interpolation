"""Central configuration: paths, dataset constants, model hyperparameters."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SOFTMAX_SPLATTING_DIR = PROJECT_ROOT / "softmax-splatting"

ZMAP_CKPT = "zmap_best.pth"
UNET_CKPT = "unet_best.pth"
TRIPLET_INDEX_FILE = "triplets.json"

# --- Data source (NOAA GOES-19 open bucket) --------------------------------
S3_BUCKET_ROOT = "noaa-goes19/ABI-L1b-RadF"
PRODUCT_CHANNEL = "C13_G19"          # ABI band 13, clean thermal IR (10.3 um)
HOURS_BACK = 72                      # how far into the past to search S3
TARGET_TRIPLETS = 32
FALLBACK_DAYS = ("2025/001", "2025/002", "2025/003")
FALLBACK_HOUR = "12"
TRIPLET_STRIDE = 3                   # 10-min cadence -> 30-min gap triplets
VAL_MAX = 6
VAL_DIVISOR = 5

# --- Preprocessing ---------------------------------------------------------
SEED = 42
CROP_SIZE = 512                      # central disk crop per frame
PATCH_SIZE = 256                     # training patch
PATCHES_PER_TRIPLET = 4
NORM_LO = 0.0                        # radiance normalization bounds (W m-2 sr-1 um-1)
NORM_HI = 200.0

# --- Training --------------------------------------------------------------
EPOCHS = 50
BATCH_SIZE = 2
NUM_WORKERS = 2
LEARNING_RATE = 2e-4
GRAD_CLIP = 1.0
L1_WEIGHT = 0.2
SSIM_WEIGHT = 0.8

# --- Inference / reporting -------------------------------------------------
GIF_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)
GIF_MINUTE_LABELS = (0, 15, 30, 45)  # before the 60-min endpoint


def ensure_dirs():
    for d in (DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
