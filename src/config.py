"""Central settings. Everything here must match the Colab fine-tuning notebook."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- paths -----------------------------------------------------------------
WEIGHTS_PATH = ROOT / "weights" / "indicw2v_emotion_v3_last.pt"
RAW_AUDIO_DIR = ROOT / "data" / "raw_audio"                # data/raw_audio/<topic>/<clip>.wav
PREDICTIONS_CSV = ROOT / "data" / "predictions.csv"        # one row per clip
CHUNKS_CSV = ROOT / "data" / "chunk_predictions.csv"       # one row per 4 s window

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}

# ---- audio (same as training) ---------------------------------------------
SAMPLE_RATE = 16_000
CHUNK_SEC = 4.0           # training used 4 s crops, so long clips are scored in 4 s windows
MIN_CHUNK_SEC = 1.0       # a leftover tail shorter than this is dropped
MAX_CHUNKS_PER_CLIP = 30  # safety cap (= 2 minutes per clip)
SILENCE_RMS = 0.003       # windows quieter than this are ignored when averaging

# ---- labels ----------------------------------------------------------------
# EXACT order of `label2id` printed in the notebook (it came from value_counts()).
# If this order is wrong, every prediction is silently mislabelled.
LABELS = [
    "anger", "sadness", "surprise", "fear", "happiness", "disgust", "neutral",  # ids 0-6: core
    "calm", "serious", "excited", "formal",                                      # ids 7-10: rare
]
NUM_LABELS = len(LABELS)
CORE_LABELS = LABELS[:7]  # the 4 rare classes had only 18-35 training clips each

# Languages that were in the fine-tuning data (from DATASET_LANG in the notebook).
# Results for any other language are zero-shot transfer and should be read with care.
TRAINED_LANGUAGES = {"bengali", "malayalam", "marathi", "kannada", "telugu"}

# ---- dashboard colours -----------------------------------------------------
EMOTION_COLORS = {
    "anger": "#D1495B", "sadness": "#3E6FB0", "surprise": "#EDAE49",
    "fear": "#7B5EA7", "happiness": "#4CAF7A", "disgust": "#8A9A5B",
    "neutral": "#9AA5B1", "calm": "#6CC4C4", "serious": "#5B6770",
    "excited": "#F28C38", "formal": "#B0B7C3",
}
