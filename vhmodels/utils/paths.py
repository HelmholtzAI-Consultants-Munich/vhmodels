# For Hyformer, one should choose where to download the model. I didn't want to let the user do this manually,
# so decided to save the model in the standard folder .cache (where all HuggingFace models are usually stored).

from pathlib import Path

# Create ~/.cache/vhmodels
CACHE_ROOT = Path.home() / ".cache" / "vhmodels"
WEIGHTS_DIR = CACHE_ROOT / "weights"

# Ensure the directories exist immediately
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def get_model_cache_dir(model_name: str) -> Path:
    """Returns and creates a specific subfolder for a model."""
    path = WEIGHTS_DIR / model_name.lower()
    path.mkdir(parents=True, exist_ok=True)
    return path
