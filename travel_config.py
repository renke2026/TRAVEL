"""Machine-independent locations; experiment settings remain in their scripts."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = os.environ.get("TRAVEL_MODEL_PATH", str(ROOT / "models/Qwen2.5-7B-Instruct"))
LLAMA_MODEL_PATH = os.environ.get("TRAVEL_LLAMA_MODEL_PATH", str(ROOT / "models/Llama-3.1-8B-Instruct"))
QWEN3B_MODEL_PATH = os.environ.get("TRAVEL_QWEN3B_MODEL_PATH", str(ROOT / "models/Qwen2.5-3B-Instruct"))

# Actual physical transformer blocks, numbered from one. Vectors keep their
# original magnitudes: hidden_state += alpha * vector, without normalization.
TRAIT_SETTINGS = {
    "comfort": dict(layer=18, method="prompt_last", scope="response", context="profile", run="20261018_141422_comfort"),
    "environment": dict(layer=16, method="prompt_last", scope="prompt", context="profile", run="20261018_161705_environment"),
    "safety": dict(layer=14, method="response_mean", scope="all", context="profile", run="20261018_183835_safety"),
    "convenience": dict(layer=14, method="response_mean", scope="system", context="neutral", run="20261018_212758_convenience"),
    "flexibility": dict(layer=18, method="response_mean", scope="all", context="profile", run="20261018_235055_flexibility"),
}
