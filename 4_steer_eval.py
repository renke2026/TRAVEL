
# Portable paths for the public research release.
import os as _TRAVEL_OS
import sys as _TRAVEL_SYS
from pathlib import Path as _TRAVEL_PATH
from travel_config import MODEL_PATH as _TRAVEL_MODEL_PATH, LLAMA_MODEL_PATH as _TRAVEL_LLAMA_PATH, QWEN3B_MODEL_PATH as _TRAVEL_QWEN3B_PATH

# ④ steer_eval.py
#    加载向量，在不同 coeff 下用 hook 干预生成
#    对当前维度的20道评测题生成回答
#    ↓ 产出 所选维度的验证回答 JSON

# ==================== 本步骤实验参数 ====================
# 维度跟随第一步的结果；已有结果目录可在run_paths.py开头的RUN_DIR指定。
MODEL_PATH = _TRAVEL_MODEL_PATH  # 与第一、第三步相同
LOAD_IN_8BIT = True
WEIGHT_READ_BACKEND = "pread"  # Windows稳定读取，模型数值不变
VECTOR_METHOD = "response_mean"    # 本轮flexibility：原角色上下文的回答均值差分
STEERING_POSITIONS = "all"       # 本轮flexibility：读题与生成阶段均注入
STEERING_LAYER = 14                 # 实际第14层，从1开始数
COEFFICIENTS = [-1, 0, 0.5, 1, 1.5] # 实际alpha，直接 h += alpha * v
MAX_NEW_TOKENS = 160
DO_SAMPLE = False
BATCH_SIZE = 32                    # 与第一步相同的批量上限；每个alpha只有20题，实际一批20条
# ======================================================

import json
import torch
from steering_utils import SteeringHook, encode_steering_batch
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_paths import TRAIT, current_run, artifact, load_trait_data, record_metadata
from methology.persona.trait_instructions import NEUTRAL_SYSTEM_PROMPT
from persona_judge_utils import scoring_config

print("="*15 + f" 第四阶段：干预验证 ({STEERING_POSITIONS}) " + "="*15)
if STEERING_POSITIONS not in ("response", "prompt", "all", "system"):
    raise ValueError("STEERING_POSITIONS请选择response、prompt、all或system。")

# ==========================================
# 1. 基础配置与模型加载（8-bit，与提取阶段一致）
# ==========================================
run_dir = current_run()
print(f"📁 读取产物目录: {run_dir}")
neutral_prompt_path = artifact(run_dir, "neutral_system_prompt.txt")
if neutral_prompt_path.exists():
    NEUTRAL_SYSTEM_PROMPT = neutral_prompt_path.read_text(encoding="utf-8")

vector_method = VECTOR_METHOD
if vector_method not in ("response_mean", "prompt_last"):
    raise ValueError("VECTOR_METHOD 只能填写 response_mean 或 prompt_last")
suffix = "all_layers" if vector_method == "response_mean" else "prompt_last"
vector_path = artifact(run_dir, f"{TRAIT}_vectors_{suffix}.pt")
if not os.path.exists(vector_path):
    raise FileNotFoundError("未找到向量文件，请先运行 3_extract_vector.py 提取。")

vector_metadata = json.loads(vector_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
if any(vector_metadata.get(key) != value for key, value in record_metadata("extract").items()):
    raise ValueError("向量来自不同框架或数据版本，请重新提取。")
if vector_metadata.get("scoring_config") != scoring_config(load_trait_data("extract")["eval_prompt"]):
    raise ValueError("向量的筛选裁判配置已过期，请从第二、三阶段重新处理后再运行；历史run保持原样。")
response_context = vector_metadata.get("response_context", "profile")
if (vector_method == "response_mean" and response_context == "neutral"
        and vector_metadata.get("neutral_system_prompt") != NEUTRAL_SYSTEM_PROMPT):
    raise ValueError("中性重编码向量与本轮评测作答提示不同，请保持提示一致后重新运行第三步。")


model_path = MODEL_PATH
from model_loading import load_local_model
model = load_local_model(model_path, LOAD_IN_8BIT, WEIGHT_READ_BACKEND)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"
model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def model_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except AttributeError:
        return next(model.parameters()).device


def encode_chat_batch(questions):
    texts = [tokenizer.apply_chat_template([
        {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ], tokenize=False, add_generation_prompt=True) for question in questions]
    inputs, mask = encode_steering_batch(tokenizer, texts, NEUTRAL_SYSTEM_PROMPT, STEERING_POSITIONS)
    return inputs.to(model_input_device(model)), mask

# ==========================================
# 2. 读取全层向量
# ==========================================
all_vectors = torch.load(vector_path, map_location="cpu").float()
if all_vectors.ndim != 2 or list(all_vectors.shape) != [model.config.num_hidden_layers + 1, model.config.hidden_size]:
    raise ValueError("向量层数或隐藏维度与当前模型不匹配。")
physical_layer = STEERING_LAYER
target_layer = physical_layer - 1
vector_index = physical_layer
if target_layer < 0 or target_layer >= model.config.num_hidden_layers or vector_index >= all_vectors.shape[0]:
    raise ValueError(
        f"target_layer={target_layer} 需要 vector_index={vector_index}，"
        f"向量共{all_vectors.shape[0]}行；STEERING_LAYER必须在1到{model.config.num_hidden_layers}之间。"
    )
if physical_layer == model.config.num_hidden_layers and vector_metadata.get("last_layer_state") != "before_final_norm":
    raise ValueError("旧向量的最后一层经过最终RMSNorm，与注入位置不一致。请用更新后的第三步重新提取向量，再选择最后一层。")
steering_vector = all_vectors[vector_index]
print(
    f"已加载 steering vector: physical_layer={physical_layer}, target_layer={target_layer}, method={vector_method}, "
    f"vector_index={vector_index}, norm={steering_vector.norm().item():.4f}"
)
print(f"向量提取上下文：{response_context}")

# ==========================================
# 3. 加载「验证集 (Evaluation Set)」
# ==========================================
eval_data = load_trait_data("eval")
run_metadata = {key: eval_data[key] for key in ("framework_version", "trait", "data_fingerprint")}
evaluation_use = eval_data.get("evaluation_use", "evaluation")
if evaluation_use == "development":
    print("评测题已有历史使用记录，不能当作未接触过的独立验证题。")

# 当前实验只使用20道测试题，不再生成交换题或差价诊断题。
eval_questions = eval_data["questions"]
positive_options = eval_data["positive_options"]
if positive_options is not None and len(positive_options) != len(eval_questions):
    raise ValueError("positive_options 数量必须与验证问题数量一致。")

# ==========================================
# 4. 按文件开头设置的范围比较不同系数
# ==========================================
# 每个维度只有一条高关注减低关注的向量；正alpha增强关注，负alpha降低关注。
# alpha=0保留模型原有偏好，不代表旅客没有偏好或评分必然为50。
test_coeffs = [float(value) for value in COEFFICIENTS]
if 0.0 not in test_coeffs:
    raise ValueError("COEFFICIENTS 必须包含 0 作为无干预对照。")
test_coeffs = sorted(set(test_coeffs))
all_eval_results = []

GENERATION_KWARGS = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": DO_SAMPLE,
    "use_cache": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}


print(f"\n开始在20道评测题上生成回答；注入范围：{STEERING_POSITIONS}...")
print(f"批量上限：{BATCH_SIZE}；每个alpha共{len(eval_questions)}题，首批实际{min(BATCH_SIZE, len(eval_questions))}条。")

for coeff in test_coeffs:
    print(f"\n正在测试干预系数: alpha={coeff}")

    # 4.3 遍历 20 道保留验证情境题
    with SteeringHook(model, physical_layer, steering_vector, coeff, STEERING_POSITIONS) as hook:
        for begin in range(0, len(eval_questions), BATCH_SIZE):
            indices = list(range(begin, min(begin+BATCH_SIZE, len(eval_questions))))
            inputs, system_mask = encode_chat_batch([eval_questions[index] for index in indices])
            hook.system_mask = system_mask
            prompt_len = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                outputs = model.generate(**inputs, **GENERATION_KWARGS)

            for batch_index, (idx, output) in enumerate(zip(indices, outputs)):
                new_ids = output[prompt_len:].tolist()
                ended = tokenizer.eos_token_id in new_ids
                if ended:
                    new_ids = new_ids[:new_ids.index(tokenizer.eos_token_id)]
                all_eval_results.append({
                    **run_metadata, "question_id": idx,
                    "evaluation_use": evaluation_use,
                    "question_source": eval_data.get("question_source"),
                    "question_set_version": eval_data.get("question_set_version"),
                    "question": eval_questions[idx],
                    "positive_option": positive_options[idx] if positive_options is not None else None,
                    "coeff": coeff,
                    "coefficient_definition": "coeff_is_actual_multiplier_of_saved_vector",
                    "intervention_scope": STEERING_POSITIONS,
                    "system_content_tokens": int(system_mask[batch_index].sum()) if system_mask is not None else 0,
                    "target_layer": target_layer, "physical_layer": physical_layer,
                    "vector_method": vector_method, "vector_norm": steering_vector.norm().item(),
                    "response_context": response_context,
                    "vector_normalization": vector_metadata.get("normalization", "none"),
                    "vector_index": vector_index, "system_prompt": NEUTRAL_SYSTEM_PROMPT,
                    "generation_kwargs": GENERATION_KWARGS, "generation_batch_size": BATCH_SIZE,
                    "weight_read_backend": WEIGHT_READ_BACKEND,
                    "generation_actual_batch_size": len(indices),
                    "response_new_tokens": len(new_ids), "response_hit_token_limit": not ended,
                    "response": tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
                })
            print(f"  - 进度: 已生成 {indices[-1] + 1}/{len(eval_questions)} 题...")

    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# 5. 导出结果，供 DeepSeek 最终高精度评测
# ==========================================
output_json_path = artifact(run_dir, f"{TRAIT}_eval_responses.json")
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(all_eval_results, f, ensure_ascii=False, indent=2)

print("\n" + "="*15 + " 第四阶段完成 " + "="*15)
print(
    f"1. 已完成 {len(eval_questions)} 题 × {len(test_coeffs)} 种系数的精细生成"
    f"（共 {len(all_eval_results)} 组对话）"
)
print(f"2. 成果文件已导出至: {output_json_path}")
print(
    "3. 达到回答 token 上限的数量: "
    f"{sum(item['response_hit_token_limit'] for item in all_eval_results)}"
)
print("=========================================\n")
