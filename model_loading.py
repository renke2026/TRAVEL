"""本地模型读取：Windows上用pread避开safetensors内存映射崩溃。

只改变权重文件的读取方式，不改变模型、精度、量化或任何实验文本。
"""

# Portable paths for the public research release.
import os as _TRAVEL_OS
import sys as _TRAVEL_SYS
from pathlib import Path as _TRAVEL_PATH
from travel_config import MODEL_PATH as _TRAVEL_MODEL_PATH, LLAMA_MODEL_PATH as _TRAVEL_LLAMA_PATH, QWEN3B_MODEL_PATH as _TRAVEL_QWEN3B_PATH

from safetensors import safe_open
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, modeling_utils


def load_local_model(path, load_in_8bit=True, weight_read_backend="pread"):
    if weight_read_backend not in ("pread", "mmap"):
        raise ValueError("WEIGHT_READ_BACKEND请选择pread或mmap。")
    original = modeling_utils.safe_open

    def open_weights(*args, **kwargs):
        kwargs["backend"] = weight_read_backend
        return safe_open(*args, **kwargs)

    # 仅在本次加载期间指定读取方式；不修改安装的库文件。
    modeling_utils.safe_open = open_weights
    try:
        return AutoModelForCausalLM.from_pretrained(path, device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None)
    finally:
        modeling_utils.safe_open = original
