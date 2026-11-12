"""第四、第六步共用的注入计算；实验参数仍写在各步骤开头。"""

# Portable paths for the public research release.
import os as _TRAVEL_OS
import sys as _TRAVEL_SYS
from pathlib import Path as _TRAVEL_PATH
from travel_config import MODEL_PATH as _TRAVEL_MODEL_PATH, LLAMA_MODEL_PATH as _TRAVEL_LLAMA_PATH, QWEN3B_MODEL_PATH as _TRAVEL_QWEN3B_PATH

import torch


def encode_steering_batch(tokenizer, texts, system_prompt, positions):
    """system模式仅标记系统消息正文，不标记角色标记、用户题目或补齐位置。"""
    inputs = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt",
                       return_offsets_mapping=positions == "system")
    if positions != "system":
        return inputs, None
    offsets = inputs.pop("offset_mapping")
    masks = []
    for text, offset in zip(texts, offsets):
        start = text.find(system_prompt)
        if start < 0 or not system_prompt:
            raise ValueError("找不到完整系统提示，不能确定注入范围。")
        end = start + len(system_prompt)
        masks.append((offset[:, 0] >= start) & (offset[:, 1] <= end) & (offset[:, 1] > offset[:, 0]))
    mask = torch.stack(masks)
    if not mask.any(dim=1).all():
        raise ValueError("系统提示Token范围为空，不能进行system注入。")
    return inputs, mask


class SteeringHook:
    """原始向量直接乘alpha；system只在读入系统消息时相加。"""
    def __init__(self, model, layer, vector, coeff, positions="response"):
        if positions not in ("response", "prompt", "all", "system"):
            raise ValueError("STEERING_POSITIONS请选择response、prompt、all或system。")
        self.block = model.model.layers[layer - 1]
        self.vector, self.coeff, self.positions = vector, coeff, positions
        self.handle, self.system_mask = None, None

    def __enter__(self):
        if self.coeff != 0:
            self.handle = self.block.register_forward_hook(self.apply)
        return self

    def apply(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.positions in ("prompt", "system") and hidden.shape[1] == 1:
            return output
        steered = hidden.clone()
        vector = self.vector.to(device=hidden.device, dtype=hidden.dtype)
        if self.positions == "system":
            if self.system_mask is None or tuple(self.system_mask.shape) != tuple(hidden.shape[:2]):
                raise ValueError("系统提示Token范围与本批输入不匹配。")
            mask = self.system_mask.to(hidden.device)
            steered[mask] += self.coeff * vector
        elif self.positions in ("all", "prompt"):
            steered += self.coeff * vector
        else:
            steered[:, -1, :] += self.coeff * vector
        return (steered,) + output[1:] if isinstance(output, tuple) else steered

    def __exit__(self, *args):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
