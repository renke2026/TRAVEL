
# Portable paths for the public research release.
import os as _TRAVEL_OS
import sys as _TRAVEL_SYS
from pathlib import Path as _TRAVEL_PATH
from travel_config import MODEL_PATH as _TRAVEL_MODEL_PATH, LLAMA_MODEL_PATH as _TRAVEL_LLAMA_PATH, QWEN3B_MODEL_PATH as _TRAVEL_QWEN3B_PATH

# ③ extract_vector.py
#    对有效样本对做前向传播，逐层提取激活值
#    vector = pos_mean - neg_mean（全层差分）
#    ↓ 产出 所选维度的全层向量文件

# ==================== 本步骤实验参数 ====================
# 维度跟随第一步的结果；已有结果目录可在run_paths.py开头的RUN_DIR指定。
MODEL_PATH = _TRAVEL_MODEL_PATH  # 与第一、第四步使用相同模型和量化设置
LOAD_IN_8BIT = True
WEIGHT_READ_BACKEND = "pread"  # Windows稳定读取，模型数值不变
MIN_EFFECTIVE_PAIRS = 2
RESPONSE_CONTEXT = "profile"  # neutral：同一评测上下文重编码；profile：保留原高低角色上下文
# 同时保存 response_mean 和 prompt_last，第四步再选择使用哪一种。
# 均值和差分使用FP32，保存原始差分，不做单位归一化。
# ======================================================

import json
import torch
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_paths import TRAIT, current_run, artifact, assert_record_compatibility, record_metadata, load_trait_data
from persona_judge_utils import scoring_config

print("="*15 + " 第三阶段：全层向量提取 " + "="*15)
# ==========================================
# 2. 读取质检通过的「有效样本对」
# ==========================================
run_dir = current_run()
print(f"📁 读取产物目录: {run_dir}")
if RESPONSE_CONTEXT not in ("neutral", "profile"):
    raise ValueError("RESPONSE_CONTEXT请选择neutral或profile。")
neutral_system_prompt = (artifact(run_dir, "neutral_system_prompt.txt").read_text(encoding="utf-8")
                         if RESPONSE_CONTEXT == "neutral" else None)
print(f"回答均值的提取上下文：{RESPONSE_CONTEXT}；prompt_last始终使用原角色提示。")

effective_pairs_path = artifact(run_dir, f"{TRAIT}_effective_pairs.json")

if not os.path.exists(effective_pairs_path):
    raise FileNotFoundError(f"找不到质检过的有效样本文件：{effective_pairs_path}，请先运行 2_efficacy_filter.py！")

with open(effective_pairs_path, "r", encoding="utf-8") as f:
    effective_pairs = json.load(f)

assert_record_compatibility(effective_pairs, "extract")
expected_judge = scoring_config(load_trait_data("extract")["eval_prompt"])
if any(item.get("scoring_config") != expected_judge for item in effective_pairs):
    raise ValueError("筛选结果使用了旧评分/质检配置，请先重新运行第二阶段；不能混用不同裁判协议。")

invalid_pairs = [
    item
    for item in effective_pairs
    if item.get("quality_rejection_reasons")
    or not item.get("quality_pos")
    or not item.get("quality_neg")
    or item.get("pos_hit_token_limit")
    or item.get("neg_hit_token_limit")
]
if invalid_pairs:
    raise ValueError(
        f"有效样本中仍有 {len(invalid_pairs)} 组截断或质量不合格数据。"
        "请先重新运行 1_pt_shengcheng.py 和 2_efficacy_filter.py。"
    )

M = len(effective_pairs)
print(f"\n成功载入 {M} 组通过筛选的对比样本。")

if M < MIN_EFFECTIVE_PAIRS:
    raise ValueError(f"有效样本不足{MIN_EFFECTIVE_PAIRS}组，请检查回答和评分后重新生成。")


model_path = MODEL_PATH
# ==========================================
# 1. 基础配置与模型加载
# ==========================================
# 8-bit 量化加载，兼顾显存占用与推理稳定性
from model_loading import load_local_model
model = load_local_model(model_path, LOAD_IN_8BIT, WEIGHT_READ_BACKEND)
model.config.output_hidden_states = True  # 在这里设置，不会污染 generation_config
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()


# ==========================================
# 3. 动态配置输出文件名与累加器
# ==========================================
# 使用本次实验的维度名称保存向量。
trait_name = TRAIT
output_vector_path = artifact(run_dir, f"{trait_name}_vectors_all_layers.pt")

num_layers = model.config.num_hidden_layers + 1
layer_pos_accum = [[] for _ in range(num_layers)]
layer_neg_accum = [[] for _ in range(num_layers)]
prompt_differences = []


def model_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except AttributeError:
        return next(model.parameters()).device


def response_span(token_ids, prompt_len):
    end = len(token_ids)
    trailing_special_ids = {
        token_id
        for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id)
        if token_id is not None
    }
    while end > prompt_len and token_ids[end - 1] in trailing_special_ids:
        end -= 1
    if end <= prompt_len:
        raise ValueError("回答 token 区间为空，无法提取响应激活。")
    return prompt_len, end


def get_hidden_states(model, token_ids):
    """第0行是嵌入；第k行是第k层输出，全部取最终RMSNorm之前的状态。"""
    # Qwen默认hidden_states的最后一行已过最终RMSNorm。
    # 直接记录最后一个block的输出，让它与其他层、第四/六步的注入位置一致。
    final_block_output = []

    def capture(module, inputs, output):
        final_block_output.append(output[0] if isinstance(output, tuple) else output)

    handle = model.model.layers[-1].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            output = model(token_ids, output_hidden_states=True, use_cache=False)
        states = list(output.hidden_states)
        states[-1] = final_block_output[0]
        return tuple(states)
    finally:
        handle.remove()


def neutral_response_inputs(item, tokenizer, system_prompt):
    """两侧用相同前缀，原始回答Token不重写；只去掉结尾EOS和补齐Token。"""
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": item["question"]},
    ], tokenize=False, add_generation_prompt=True)
    prefix = tokenizer(text, add_special_tokens=False).input_ids
    sequences = []
    for side in ("pos", "neg"):
        response = item[f"outputs_{side}_tokens"][item[f"prompt_{side}_len"]:]
        while response and response[-1] in {tokenizer.eos_token_id, tokenizer.pad_token_id}:
            response = response[:-1]
        if not response:
            raise ValueError("回答Token为空，不能进行中性上下文重编码。")
        sequences.append(prefix + response)
    return sequences[0], sequences[1], len(prefix)


# ==========================================
# 4. 循环前向传播：拦截隐状态并精准切片
# ==========================================
print(f"\n开始对 {M} 对样本分别提取回答均值与提示末位置（每对四次前向传播）...")

for idx, item in enumerate(effective_pairs):
    # 4.1 直接从 JSON 中重构 Token 张量并送入 GPU，完美避开二次 Tokenizer 错位问题
    # Token 列表转为 [1, seq_len] 的 2D Tensor
    outputs_pos_tokens = item["outputs_pos_tokens"]
    outputs_neg_tokens = item["outputs_neg_tokens"]
    device = model_input_device(model)
    prompt_pos_len = item["prompt_pos_len"]
    prompt_neg_len = item["prompt_neg_len"]
    response_pos_tokens, response_neg_tokens = outputs_pos_tokens, outputs_neg_tokens
    response_pos_prefix, response_neg_prefix = prompt_pos_len, prompt_neg_len
    if RESPONSE_CONTEXT == "neutral":
        response_pos_tokens, response_neg_tokens, prefix_length = neutral_response_inputs(
            item, tokenizer, neutral_system_prompt)
        response_pos_prefix = response_neg_prefix = prefix_length
    outputs_pos = torch.tensor([response_pos_tokens], device=device)
    outputs_neg = torch.tensor([response_neg_tokens], device=device)
    pos_start, pos_end = response_span(response_pos_tokens, response_pos_prefix)
    neg_start, neg_end = response_span(response_neg_tokens, response_neg_prefix)
    
    # 4.2 单次前向传播（只读、不计算梯度）
    states_pos = get_hidden_states(model, outputs_pos)
    states_neg = get_hidden_states(model, outputs_neg)
    
    for l in range(num_layers):
        # 切片提取回答区间，并在序列长度维度求均值降维，剥离梯度并移到 CPU
        pos_mean_l = states_pos[l][:, pos_start:pos_end, :].float().mean(dim=1).cpu()
        neg_mean_l = states_neg[l][:, neg_start:neg_end, :].float().mean(dim=1).cpu()
        
        layer_pos_accum[l].append(pos_mean_l)
        layer_neg_accum[l].append(neg_mean_l)
        
    # 4.4 即时释放庞大显存，确保显存处于绝对安全线
    del states_pos, states_neg
    # 提示向量只前向传播提示，不需要回答；避免量化实现受整段回答的数值范围影响。
    prefix_pos = get_hidden_states(model, torch.tensor([outputs_pos_tokens[:prompt_pos_len]], device=device))
    prefix_neg = get_hidden_states(model, torch.tensor([outputs_neg_tokens[:prompt_neg_len]], device=device))
    prompt_differences.append(torch.stack([
        prefix_pos[l][0, -1].float().cpu()
        - prefix_neg[l][0, -1].float().cpu()
        for l in range(num_layers)
    ]))
    del prefix_pos, prefix_neg
    gc.collect()
    torch.cuda.empty_cache()
    
    if (idx + 1) % 5 == 0 or idx == M - 1:
        print(f"  - 提取进度: {idx + 1}/{M}...")

# ==========================================
# 5. 全样本均值化、差分计算与全层合并
# ==========================================
print("\n" + "="*15 + " 开始计算高精度全层差分 " + "="*15)

all_layer_vectors = []

for l in range(num_layers):
    # 5.1 将多组有效样本的 [1, hidden_size] 拼接为 [M, hidden_size] 矩阵
    pos_samples_l = torch.cat(layer_pos_accum[l], dim=0)
    neg_samples_l = torch.cat(layer_neg_accum[l], dim=0)
    
    # 5.2 样本轴求均值：按相同聚合口径计算样本平均；不保证消除全部情境噪音
    pos_mean_l = pos_samples_l.mean(dim=0, keepdim=True)  # 形状为 [1, hidden_size]
    neg_mean_l = neg_samples_l.mean(dim=0, keepdim=True)  # 形状为 [1, hidden_size]
    
    # 正向均值减反向均值；与参考实现一样不做L2单位归一化。
    layer_vector = pos_mean_l - neg_mean_l
    all_layer_vectors.append(layer_vector)

# 合并为 [num_layers, hidden_size]，均值与做差均在FP32中计算。
all_persona_vectors = torch.cat(all_layer_vectors, dim=0)

# 5.5 持久化保存
torch.save(all_persona_vectors, output_vector_path)
prompt_vector_path = artifact(run_dir, f"{trait_name}_vectors_prompt_last.pt")
prompt_vectors = torch.stack(prompt_differences).mean(0)
torch.save(prompt_vectors, prompt_vector_path)
vector_metadata = record_metadata("extract") | {
    "model_path": model_path, "weight_read_backend": WEIGHT_READ_BACKEND, "sample_count": M,
    "shape": list(all_persona_vectors.shape),
    "layer_layout": "row 0: embedding; row k: raw output of physical block k (1-based); final RMSNorm excluded",
    "last_layer_state": "before_final_norm",
    "extraction_position": "response_mean", "aggregation_dtype": "float32", "normalization": "none",
    "response_context": RESPONSE_CONTEXT, "neutral_system_prompt": neutral_system_prompt,
    "response_tokens": "original_generated_tokens_excluding_trailing_special_tokens",
    "layer_norms": all_persona_vectors.norm(dim=1).tolist(),
    "scoring_config": expected_judge,
}
output_vector_path.with_suffix(".metadata.json").write_text(
    json.dumps(vector_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
)
prompt_vector_path.with_suffix(".metadata.json").write_text(json.dumps(
    vector_metadata | {"extraction_position": "prompt_last", "forward_context": "prompt_only",
                       "response_context": "profile", "neutral_system_prompt": None,
                       "layer_norms": prompt_vectors.norm(dim=1).tolist()},
    ensure_ascii=False, indent=2), encoding="utf-8")

print("\n" + "="*15 + " 向量提取完毕 " + "="*15)
print(f"1. 参与计算的有效样本数 M = {M} 组。")
print(f"2. 生成的全层激活特征向量维度: {all_persona_vectors.shape}")
print(f"3. 出行偏好向量文件已成功保存至本地：{output_vector_path}")
print(f"4. 提示末位置向量：{prompt_vector_path}")
print("=========================================\n")
