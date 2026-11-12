
# Portable paths for the public research release.
import os as _TRAVEL_OS
import sys as _TRAVEL_SYS
from pathlib import Path as _TRAVEL_PATH
from travel_config import MODEL_PATH as _TRAVEL_MODEL_PATH, LLAMA_MODEL_PATH as _TRAVEL_LLAMA_PATH, QWEN3B_MODEL_PATH as _TRAVEL_QWEN3B_PATH

# ② efficacy_filter.py
#    用 DeepSeek API 对 100 对回答做 logprobs 连续打分
#    筛选出正向≥50、反向<50的有效样本对
#    ↓ 产出 {TRAIT}_effective_pairs.json

# ==================== 本步骤实验参数 ====================
# 维度跟随第一步的结果；裁判参数：persona_judge_utils.py；指定结果目录：run_paths.py。
MIN_POS_SCORE = 50.0    # 促进回答至少达到此分数
MAX_NEG_SCORE = 50.0    # 抑制回答必须低于此分数
# ======================================================

import json
import os
import hashlib
from collections import Counter

from run_paths import TRAIT, current_run, artifact, load_trait_data, assert_record_compatibility
from persona_judge_utils import create_deepseek_client, score_answer, scoring_config, assess_answer_quality

print("="*15 + " 第二阶段：特质评分与一致性筛选 (Logprobs 版) " + "="*15)

# ==========================================
# 1. 配置 DeepSeek API 客户端
# ==========================================
client = create_deepseek_client()

# ==========================================
# 2. 读取第一阶段生成的原始数据与评分量表
# ==========================================
run_dir = current_run()
print(f"📁 读取产物目录: {run_dir}")

raw_pairs_path = artifact(run_dir, f"{TRAIT}_raw_pairs_100.json")


if not os.path.exists(raw_pairs_path):
    raise FileNotFoundError(f"找不到原始文本文件，请先运行 1_pt_shengcheng.py 生产数据！")

with open(raw_pairs_path, "r", encoding="utf-8") as f:
    raw_pairs = json.load(f)

assert_record_compatibility(raw_pairs, "extract")
extract_data = load_trait_data("extract")

# 获取打分量表模板
rubric_template = extract_data["eval_prompt"]

# ==========================================
# 评分配置：这些字段变化时，缓存必须自动失效
# ==========================================
# 与第五阶段使用同一配置；参数、重试提示或解析版本变化会使旧缓存失效。
SCORING_CONFIG = scoring_config(rubric_template)
print(f"特质裁判请求模型: {SCORING_CONFIG['judge_model']} | 温度: {SCORING_CONFIG['temperature']} | Token上限: {SCORING_CONFIG['max_tokens']} | Top Logprobs: {SCORING_CONFIG['top_logprobs']} | 固定重复: {SCORING_CONFIG['score_repeats']}；独立质检温度: {SCORING_CONFIG['quality_temperature']}")


def pair_cache_key(item):
    payload = {
        "question": item.get("question", ""),
        "sys_pos": item.get("sys_pos", ""),
        "sys_neg": item.get("sys_neg", ""),
        "response_pos": item.get("response_pos", ""),
        "response_neg": item.get("response_neg", ""),
        "scoring_config": SCORING_CONFIG,
        "framework_version": item.get("framework_version"),
        "trait": item.get("trait"),
        "data_fingerprint": item.get("data_fingerprint"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


raw_pair_keys = {item["pair_id"]: pair_cache_key(item) for item in raw_pairs}

# ==========================================
# 3. 断点续传缓存加载
# ==========================================
temp_scored_path = artifact(run_dir, f"{TRAIT}_scored_pairs_temp_pv.json")
scored_pairs = []


def cached_pair_completed(item):
    """真实不合格可缓存；API或解析失败不能永久当成已完成。"""
    for side in ("pos", "neg"):
        quality = item.get(f"quality_diagnostics_{side}", [])
        if not quality:
            return False
        if not item.get(f"quality_{side}"):
            if quality[-1].get("valid") and quality[-1].get("status") in ("fail", "uncertain"):
                continue
            if quality[-1].get("reason") == "missing_explicit_choice":
                continue
            if quality[-1].get("valid") and quality[-1].get("consistent") is False:
                continue
            return False
        if item.get(f"score_{side}") is None:
            score_attempts = item.get(f"score_diagnostics_{side}", [])
            if not score_attempts or score_attempts[-1].get("reason") != "refusal":
                return False
    return True


if os.path.exists(temp_scored_path):
    with open(temp_scored_path, "r", encoding="utf-8") as f:
        cached_pairs = json.load(f)
    scored_pairs = [
        item for item in cached_pairs
        if item.get("cache_key") == raw_pair_keys.get(item.get("pair_id")) and cached_pair_completed(item)
    ]
    dropped_count = len(cached_pairs) - len(scored_pairs)
    print(f"📦 已加载本地已完成的打分缓存（当前有效进度：{len(scored_pairs)}/{len(raw_pairs)}）。")
    if dropped_count:
        print(f"🧹 已忽略 {dropped_count} 条配置不匹配或未成功完成的缓存。")

scored_ids = {item["pair_id"] for item in scored_pairs}

# ==========================================
# 4. 循环调用连续打分接口
# ==========================================
remaining_pairs = len(raw_pairs) - len(scored_pairs)
print(
    f"剩余 {remaining_pairs} 组样本需要正反向打分；"
    f"每条回答先独立检查一致性，通过后做{SCORING_CONFIG['score_repeats']}次固定评分；每次最多尝试 {SCORING_CONFIG['max_score_attempts']} 次 API。\n"
)

for idx, item in enumerate(raw_pairs):
    pair_id = item["pair_id"]
    
    if pair_id in scored_ids:
        continue
        
    question = item["question"]
    resp_pos = item["response_pos"]
    resp_neg = item["response_neg"]
    
    # 独立检查事实/选择理由，不能用高偏好评分掩盖错误回答。
    quality_pos, quality_diag_pos = assess_answer_quality(client, question, resp_pos)
    quality_neg, quality_diag_neg = assess_answer_quality(client, question, resp_neg)
    item["quality_pos"] = quality_pos
    item["quality_neg"] = quality_neg
    item["quality_diagnostics_pos"] = quality_diag_pos
    item["quality_diagnostics_neg"] = quality_diag_neg
    item["quality_rejection_reasons"] = [
        side for side, valid in (("pos", quality_pos), ("neg", quality_neg)) if not valid
    ]
    if item.get("pos_hit_token_limit") or item.get("neg_hit_token_limit"):
        item["quality_rejection_reasons"].append("truncated_response")
    score_pos, diagnostics_pos = (score_answer(client, rubric_template, question, resp_pos)
        if quality_pos else (None, [{"reason": "quality_check_failed"}]))
    score_neg, diagnostics_neg = (score_answer(client, rubric_template, question, resp_neg)
        if quality_neg else (None, [{"reason": "quality_check_failed"}]))
    
    item["score_pos"] = score_pos
    item["score_neg"] = score_neg
    item["score_diagnostics_pos"] = diagnostics_pos
    item["score_diagnostics_neg"] = diagnostics_neg
    item["scoring_config"] = SCORING_CONFIG
    item["cache_key"] = raw_pair_keys[pair_id]
    
    scored_pairs.append(item)
    
    # 即时保存进度，防止中途断网崩溃
    with open(temp_scored_path, "w", encoding="utf-8") as f:
        json.dump(scored_pairs, f, ensure_ascii=False, indent=2)
        
    print(f"进度: {len(scored_pairs)}/{len(raw_pairs)} | 正向期望分: {score_pos} | 反向期望分: {score_neg}")

# ==========================================
# 5. 黄金质检硬性过滤（The Efficacy Filter）
# ==========================================
print("\n" + "="*15 + " 对比样本筛选结果 " + "="*15)

filtered_pairs = []

# 同一维度统一按高关注表达评分：高关注回答>=50，低关注回答<50。
# 不对负向回答切换量表或取反；事实质检和截断检查仍须同时通过。
for item in scored_pairs:
    s_pos = item.get("score_pos")
    s_neg = item.get("score_neg")
    
    if s_pos is not None and s_neg is not None and not item.get("quality_rejection_reasons"):
        if s_pos >= MIN_POS_SCORE and s_neg < MAX_NEG_SCORE:
            filtered_pairs.append(item)

# 保存满足一致性和特质分数阈值的对比数据集。
output_effective_path = artifact(run_dir, f"{TRAIT}_effective_pairs.json")
with open(output_effective_path, "w", encoding="utf-8") as f:
    json.dump(filtered_pairs, f, ensure_ascii=False, indent=2)

total_pairs = len(raw_pairs)
print(f"1. {total_pairs} 组原始数据中，共检测完 {len(scored_pairs)} 组。")
print(f"2. 同时满足一致性与促进/抑制分数阈值的有效样本对: {len(filtered_pairs)} 组。")
print(f"3. 淘汰率: {((total_pairs - len(filtered_pairs)) / total_pairs) * 100:.1f}%")
print(f"4. 有效数据集已保存至: {output_effective_path}")
rejections = Counter()
for item in scored_pairs:
    if item.get("quality_rejection_reasons"):
        rejections["quality_or_truncation"] += 1
    elif item.get("score_pos") is None or item.get("score_neg") is None:
        rejections["score_failure"] += 1
    elif item["score_pos"] < MIN_POS_SCORE:
        rejections["positive_below_threshold"] += 1
    elif item["score_neg"] >= MAX_NEG_SCORE:
        rejections["negative_above_threshold"] += 1
facets = extract_data.get("question_facets")
selection_summary = {
    "total": total_pairs, "retained": len(filtered_pairs), "rejection_counts": dict(rejections),
    "retained_by_instruction": dict(Counter(r["instruction_idx"] for r in filtered_pairs if "instruction_idx" in r)),
    "retained_by_question": {str(q): sum(r.get("question_idx") == q for r in filtered_pairs) for q in range(len(extract_data.get("questions", [])))},
    "retained_by_facet": dict(Counter(facets[r["question_idx"]] for r in filtered_pairs if "question_idx" in r)) if facets else None,
    "thresholds": {"positive_at_least": MIN_POS_SCORE, "negative_below": MAX_NEG_SCORE},
}
artifact(run_dir, f"{TRAIT}_selection_summary.json").write_text(
    json.dumps(selection_summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"淘汰原因（按筛选顺序）：{dict(rejections)}")
print(f"五对提示各保留：{selection_summary['retained_by_instruction']}")
if facets:
    print(f"各侧面保留：{selection_summary['retained_by_facet']}")
pending_quality = [{"pair_id": item["pair_id"], "side": side, "question": item["question"],
    "response": item[f"response_{side}"], "diagnostics": item[f"quality_diagnostics_{side}"]}
    for item in scored_pairs for side in ("pos", "neg")
    if item[f"quality_diagnostics_{side}"][-1].get("status") == "uncertain"]
artifact(run_dir, f"{TRAIT}_extract_quality_review.json").write_text(
    json.dumps(pending_quality, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"其中待核验的回答侧数: {len(pending_quality)}；未自动视为通过。")
quality_errors = sum(item[f"quality_diagnostics_{side}"][-1].get("status") == "error"
    for item in scored_pairs for side in ("pos", "neg"))
print(f"质检接口/格式错误侧数: {quality_errors}；不缓存成永久不合格，可恢复后续跑。")
quality_rejected = score_invalid = pos_low = neg_high = 0
for item in scored_pairs:
    if item.get("quality_rejection_reasons"):
        quality_rejected += 1
    elif item["score_pos"] is None or item["score_neg"] is None:
        score_invalid += 1
    elif item["score_pos"] < MIN_POS_SCORE:
        pos_low += 1
    elif item["score_neg"] >= MAX_NEG_SCORE:
        neg_high += 1
print(f"淘汰原因（依次归类、不重复计数）：一致性/截断 {quality_rejected}，评分无效 {score_invalid}，促进分不足 {pos_low}，抑制分过高 {neg_high}。")
for index in sorted({item["instruction_idx"] for item in scored_pairs if "instruction_idx" in item}):
    total = sum(item.get("instruction_idx") == index for item in scored_pairs)
    kept = sum(item.get("instruction_idx") == index for item in filtered_pairs)
    print(f"提示组 {index + 1}: 保留 {kept}/{total} 对")
valid_diagnostics = [d for item in scored_pairs for side in ("pos", "neg")
                     for d in item.get(f"score_diagnostics_{side}", []) if d.get("valid") and "numeric_mass" in d]
point_count = sum(d.get("near_point_mass", False) for d in valid_diagnostics)
print(f"特质评分候选概率近单点: {point_count}/{len(valid_diagnostics)} 次；近单点时加权结果基本等于一个整数。")
print("=========================================\n")
