
# Portable paths for the public research release.
import os as _TRAVEL_OS
import sys as _TRAVEL_SYS
from pathlib import Path as _TRAVEL_PATH
from travel_config import MODEL_PATH as _TRAVEL_MODEL_PATH, LLAMA_MODEL_PATH as _TRAVEL_LLAMA_PATH, QWEN3B_MODEL_PATH as _TRAVEL_QWEN3B_PATH

# ⑤ deepseek_judge.py
#    对干预生成的回答做 logprobs 连续打分
#    统计各 coeff 的目标出行人格特质表现分，判断向量是否有效
#    ↓ 产出 所选维度的评分结果 JSON

# ==================== 本步骤实验参数 ====================
# 维度跟随第一步的结果；裁判参数：persona_judge_utils.py；指定结果目录：run_paths.py。
MONOTONIC_TOLERANCE = 2.0   # 相邻alpha分数允许的小幅波动
MIN_RESPONSIVE_DELTA = 10.0
MIN_STRONG_DELTA = 20.0
BOOTSTRAP_SAMPLES = 10000
RANDOM_SEED = 20261016
LOW_SCORE_THRESHOLD = 25.0   # 仅用于汇总分数区间，不是裁判评分锚点
HIGH_SCORE_THRESHOLD = 75.0
SAVE_EFFECT_FIGURE = True
FIGURE_DPI = 300
# ======================================================

import json
import os
from collections import defaultdict
from statistics import mean, median
from math import comb
import numpy as np

from run_paths import TRAIT, current_run, artifact, load_trait_data, assert_record_compatibility

from persona_judge_utils import (
    create_deepseek_client,
    extract_selected_option,
    score_answer,
    scoring_config,
    assess_answer_quality,
)


print("=" * 15 + " 启动 DeepSeek Persona Vectors 风格裁判引擎 " + "=" * 15)

run_dir = current_run()
print(f"📁 读取产物目录: {run_dir}")

eval_responses_path = artifact(run_dir, f"{TRAIT}_eval_responses.json")

if not os.path.exists(eval_responses_path):
    raise FileNotFoundError(f"找不到生成的回答文件：{eval_responses_path}")

with open(eval_responses_path, "r", encoding="utf-8") as f:
    eval_results = json.load(f)

assert_record_compatibility(eval_results, "eval")
relabelled = 0
for item in eval_results:
    if item.get("coefficient_definition") != "coeff_is_actual_multiplier_of_saved_vector":
        # 旧第四阶段将输入系数记为coeff；只在内存中恢复实际系数，不回写原始回答。
        requested = float(item["coeff"])
        scale = float(item.get("vector_scale", 1))
        if not 0 < scale < float("inf"):
            raise ValueError("旧回答记录中的vector_scale必须为有限正数")
        item.update(requested_coeff=requested, coeff=requested*scale,
            coefficient_definition="coeff_is_actual_multiplier_of_saved_vector",
            coefficient_relabelled_from_legacy=True)
        relabelled += scale != 1
if relabelled:
    print(f"已按旧记录的vector_scale还原{relabelled}条回答的实际α；原始回答文件保持不变。")
rubric_data = load_trait_data("eval")
if rubric_data.get("evaluation_use") == "development":
    print("评测题已有历史使用记录，不能当作未接触过的独立验证题。")

rubric_template = rubric_data["eval_prompt"]
client = create_deepseek_client()
# 与第二阶段共用参数、解析和重试逻辑，完整配置随评分结果保存。
SCORING_CONFIG = scoring_config(rubric_template)
print(f"特质裁判请求模型: {SCORING_CONFIG['judge_model']} | 温度: {SCORING_CONFIG['temperature']} | Token上限: {SCORING_CONFIG['max_tokens']} | Top Logprobs: {SCORING_CONFIG['top_logprobs']} | 固定重复: {SCORING_CONFIG['score_repeats']}；独立质检温度: {SCORING_CONFIG['quality_temperature']}")


print(f"共加载 {len(eval_results)} 条回答，开始调用 Logprobs 特质评分接口...\n")

scored_results = []
for idx, item in enumerate(eval_results):
    quality_ok, quality_diagnostics = assess_answer_quality(client, item["question"], item["response"])
    quality_ok = quality_ok and not item.get("response_hit_token_limit", False)
    # 与参考实现一致，特质表达和质量分别测量；错误回答仍不能作为有效对齐证据。
    score, diagnostics = score_answer(client, rubric_template, item["question"], item["response"])

    selected_option = extract_selected_option(item["response"], item["question"])
    positive_option = item.get("positive_option")

    result = dict(item)
    result.update(
        {
            "score": score,
            "quality_valid_score": score if quality_ok else None,
            "selected_option": selected_option,
            "positive_choice": (
                selected_option == positive_option
                if quality_ok and selected_option is not None and positive_option is not None
                else None
            ),
            "score_diagnostics": diagnostics,
            "quality_ok": quality_ok,
            "quality_diagnostics": quality_diagnostics,
            "scoring_config": SCORING_CONFIG,
        }
    )
    scored_results.append(result)

    print(
        f"进度: {idx + 1}/{len(eval_results)} | "
        f"题号: {item['question_id']} | Coeff: {item['coeff']:4.1f} | "
        f"期望打分: {score} | 质检: {quality_diagnostics[-1].get('status', 'fail')} | 选择: {selected_option or '未识别'}"
    )

output_path = artifact(run_dir, f"{TRAIT}_scored_results_pv.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(scored_results, f, ensure_ascii=False, indent=2)

coefficients = sorted({float(item["coeff"]) for item in scored_results})
scores_by_coeff = defaultdict(list)
choices_by_coeff = defaultdict(list)
by_question = defaultdict(dict)

for item in scored_results:
    coeff = float(item["coeff"])
    if coeff in by_question[item["question_id"]]:
        raise ValueError("检测到同题同系数多条回答；需先按题聚合重复样本，不能静默覆盖。")
    by_question[item["question_id"]][coeff] = item
    if item["score"] is not None:
        scores_by_coeff[coeff].append(float(item["score"]))
    if item["positive_choice"] is not None:
        choices_by_coeff[coeff].append(bool(item["positive_choice"]))

print("\n" + "=" * 15 + " 全部回答的特质表达（与质量分开） " + "=" * 15)
for coeff in coefficients:
    scores = scores_by_coeff[coeff]
    if not scores:
        print(f"Coeff = {coeff:4.1f} | 无有效分数")
        continue
    low = sum(score < LOW_SCORE_THRESHOLD for score in scores)
    mid = sum(LOW_SCORE_THRESHOLD <= score <= HIGH_SCORE_THRESHOLD for score in scores)
    high = sum(score > HIGH_SCORE_THRESHOLD for score in scores)
    print(
        f"Coeff = {coeff:4.1f} | 平均: {mean(scores):.2f} | "
        f"中位数: {median(scores):.2f} | <{LOW_SCORE_THRESHOLD:g}: {low} | "
        f"{LOW_SCORE_THRESHOLD:g}-{HIGH_SCORE_THRESHOLD:g}: {mid} | "
        f">{HIGH_SCORE_THRESHOLD:g}: {high} | N={len(scores)}"
    )

complete_question_ids = [
    question_id
    for question_id, item_map in sorted(by_question.items())
    if all(
        coeff in item_map and item_map[coeff]["score"] is not None
        for coeff in coefficients
    )
]

paired_means = {}
print("\n" + "=" * 15 + " 完整配对主分析 " + "=" * 15)
print(
    f"所有 {len(coefficients)} 个系数均有有效分数的题目: "
    f"{len(complete_question_ids)}/{len(by_question)}"
)
for coeff in coefficients:
    paired_scores = [
        float(by_question[question_id][coeff]["score"])
        for question_id in complete_question_ids
    ]
    paired_means[coeff] = mean(paired_scores) if paired_scores else None
    if paired_scores:
        print(f"Coeff = {coeff:4.1f} | 配对均值: {paired_means[coeff]:.2f}")

trajectory_diagnostics = []
if complete_question_ids:
    min_coeff = coefficients[0]
    max_coeff = coefficients[-1]

    for question_id in complete_question_ids:
        trajectory = [
            float(by_question[question_id][coeff]["score"])
            for coeff in coefficients
        ]
        endpoint_delta = trajectory[-1] - trajectory[0]
        ordered = all(
            trajectory[index + 1] + MONOTONIC_TOLERANCE >= trajectory[index]
            for index in range(len(trajectory) - 1)
        )
        responsive = endpoint_delta >= MIN_RESPONSIVE_DELTA
        strong = endpoint_delta >= MIN_STRONG_DELTA
        trajectory_diagnostics.append(
            {
                "question_id": question_id,
                "scores": dict(zip(map(str, coefficients), trajectory)),
                "endpoint_delta": round(endpoint_delta, 2),
                "ordered_with_tolerance": ordered,
                "responsive": responsive,
                "strong_response": strong,
                "robust_monotonic": ordered and responsive,
                "ceiling": min(trajectory) > HIGH_SCORE_THRESHOLD,
                "floor": max(trajectory) < LOW_SCORE_THRESHOLD,
                "reversed": endpoint_delta < -MIN_RESPONSIVE_DELTA,
            }
        )

    endpoint_deltas = [
        item["endpoint_delta"] for item in trajectory_diagnostics
    ]
    print("\n" + "=" * 15 + " 剂量轨迹诊断 " + "=" * 15)
    print(
        f"平均 delta({max_coeff:g} - {min_coeff:g}): "
        f"{mean(endpoint_deltas):.2f}"
    )
    for key, label in (
        ("ordered_with_tolerance", "顺序单调题"),
        ("responsive", f"有效响应题（delta>={MIN_RESPONSIVE_DELTA:g}）"),
        ("strong_response", f"强响应题（delta>={MIN_STRONG_DELTA:g}）"),
        ("robust_monotonic", "稳健单调题（顺序单调且有效响应）"),
        ("ceiling", "天花板题"),
        ("floor", "地板题"),
        ("reversed", "反向异常题"),
    ):
        question_ids = [
            item["question_id"] for item in trajectory_diagnostics if item[key]
        ]
        print(f"{label}: {len(question_ids)} | 题号: {question_ids}")

print("\n" + "=" * 15 + " 明确选择行为分析 " + "=" * 15)
choice_rates = {}
for coeff in coefficients:
    choices = choices_by_coeff[coeff]
    rate = mean(choices) if choices else None
    choice_rates[coeff] = rate
    if rate is None:
        print(f"Coeff = {coeff:4.1f} | 无法识别明确选择")
    else:
        print(
            f"Coeff = {coeff:4.1f} | 改善端选择率（各组可用题，描述性）: "
            f"{rate * 100:.1f}% | N={len(choices)}"
        )

# 比较系数时，必须用相同题目；不能把不同质检分母的比例当成趋势。
choice_common_ids = [qid for qid, rows in sorted(by_question.items())
    if all(c in rows and rows[c]["positive_choice"] is not None for c in coefficients)]
paired_choice_rates = {str(c): mean(by_question[qid][c]["positive_choice"] for qid in choice_common_ids)
    if choice_common_ids else None for c in coefficients}
print(f"\n配对选择行为：所有系数共同通过质检且明确选择 {len(choice_common_ids)}/{len(by_question)}题")
for coeff in coefficients:
    rate = paired_choice_rates[str(coeff)]
    print(f"Coeff = {coeff:g} | 相同题目的改善端选择率：{rate:.1%}" if rate is not None else f"Coeff = {coeff:g} | 无共同题")
quality_score_common_ids = [qid for qid in complete_question_ids
    if all(by_question[qid][c]["quality_ok"] for c in coefficients)]
quality_paired_means = {str(c): mean(by_question[qid][c]["score"] for qid in quality_score_common_ids)
    if quality_score_common_ids else None for c in coefficients}
print(f"共同质检合格题的表达分：N={len(quality_score_common_ids)}")
for coeff in coefficients:
    value = quality_paired_means[str(coeff)]
    print(f"Coeff = {coeff:g} | 共同合格题均分：{value:.2f}" if value is not None else f"Coeff = {coeff:g} | 无共同题")
choice_changes = []
choice_comparisons = sorted(set([(0.0, c) for c in coefficients if c != 0 and 0 in coefficients]
    + [(coefficients[0], coefficients[-1])]))
for left, coeff in choice_comparisons:
    ids = [qid for qid, rows in sorted(by_question.items())
        if rows.get(left, {}).get("positive_choice") is not None and rows.get(coeff, {}).get("positive_choice") is not None]
    forward = [qid for qid in ids if not by_question[qid][left]["positive_choice"] and by_question[qid][coeff]["positive_choice"]]
    reverse = [qid for qid in ids if by_question[qid][left]["positive_choice"] and not by_question[qid][coeff]["positive_choice"]]
    discordant = len(forward) + len(reverse)
    pvalue = min(1., 2*sum(comb(discordant, k) for k in range(min(len(forward), len(reverse))+1))/2**discordant) if discordant else 1.
    choice_changes.append({"left": left, "coeff": coeff, "question_ids": ids, "n": len(ids),
        "ordinary_to_improved": forward, "improved_to_ordinary": reverse,
        "exact_mcnemar_two_sided_p": pvalue})

invalid_scores = [
    {
        "question_id": item["question_id"],
        "coeff": item["coeff"],
        "attempt_reasons": [
            attempt.get("reason") for attempt in item["score_diagnostics"]
        ],
    }
    for item in scored_results
    if item["score"] is None
]
unrecognized_choices = [
    {"question_id": item["question_id"], "coeff": item["coeff"]}
    for item in scored_results
    if item["selected_option"] is None
]

# 固定题目内配对，分别报告全部表达与质量合格表达，不把缺失当作低分或删除整道题的所有轨迹。
def paired_change(left, right, require_quality=False):
    # 0→负alpha应当降分；异常方向随系数变化，不能统一把降分视为异常。
    expected_direction = 1 if right > left else -1
    deltas, ids = [], []
    for question_id, rows in sorted(by_question.items()):
        a, b = rows.get(left), rows.get(right)
        if not a or not b or a["score"] is None or b["score"] is None:
            continue
        if require_quality and not (a["quality_ok"] and b["quality_ok"]):
            continue
        ids.append(question_id)
        deltas.append(b["score"] - a["score"])
    interval = None
    if len(deltas) >= 2:
        rng = np.random.default_rng(RANDOM_SEED)
        boot = rng.choice(deltas, size=(BOOTSTRAP_SAMPLES, len(deltas)), replace=True).mean(axis=1)
        interval = np.quantile(boot, [0.025, 0.975]).tolist()
    return {"left": left, "right": right, "n": len(deltas), "question_ids": ids,
        "mean_delta": mean(deltas) if deltas else None, "median_delta": median(deltas) if deltas else None,
        "question_bootstrap_95_ci": interval,
        "reverse_question_ids": [qi for qi, delta in zip(ids, deltas)
                                 if delta * expected_direction < -MIN_RESPONSIVE_DELTA]}


comparisons = sorted(set([(0.0, coeff) for coeff in coefficients if coeff != 0]
    + list(zip(coefficients[:-1], coefficients[1:]))))
pairwise = [paired_change(a, b) for a, b in comparisons]
quality_pairwise = [paired_change(a, b, True) for a, b in comparisons]
quality_summary = {}
print("\n" + "=" * 15 + " 质量合格率与质量合格回答得分 " + "=" * 15)
for coeff in coefficients:
    group = [item for item in scored_results if float(item["coeff"]) == coeff]
    good = [item["score"] for item in group if item["quality_ok"] and item["score"] is not None]
    quality_summary[str(coeff)] = {"total": len(group), "passed": sum(item["quality_ok"] for item in group),
        "uncertain": sum(item["quality_diagnostics"][-1].get("status") == "uncertain" for item in group),
        "quality_errors": sum(item["quality_diagnostics"][-1].get("status") == "error" for item in group),
        "truncated": sum(item.get("response_hit_token_limit", False) for item in group),
        "valid_mean": mean(good) if good else None, "valid_score_n": len(good)}
    print(f"Coeff = {coeff:g} | 质检通过 {quality_summary[str(coeff)]['passed']}/{len(group)} | 合格回答均分: {mean(good) if good else None}")
print("\n固定题目与中立组的配对变化（95%区间按题目bootstrap；单次生成时不涵盖生成随机性）：")
for comparison in pairwise:
    if comparison["left"] == 0:
        print(f"Coeff={comparison['right']:g} | N={comparison['n']} | delta={comparison['mean_delta']} | CI={comparison['question_bootstrap_95_ci']}")
local_reversals = [item for item in pairwise if item["reverse_question_ids"]]
print(f"各相邻系数/中立组比较中的反向异常: {local_reversals}")
print("全部回答分数描述特质表达；事实错误或待核验回答的高分不构成有效对齐证据。")

valid_diagnostics = [d for item in scored_results for d in item["score_diagnostics"] if d.get("valid") and "numeric_mass" in d]
point_count = sum(d.get("near_point_mass", False) for d in valid_diagnostics)
summary = {
    "evaluation_use": rubric_data.get("evaluation_use", "evaluation"),
    "question_source": rubric_data.get("question_source"),
    "question_set_versions": sorted({item["question_set_version"] for item in scored_results if item.get("question_set_version")}),
    "scoring_config": SCORING_CONFIG,
    "coefficients": coefficients,
    "total_questions": len(by_question),
    "complete_question_ids": complete_question_ids,
    "paired_means": {str(key): value for key, value in paired_means.items()},
    "choice_positive_rates": {
        str(key): value for key, value in choice_rates.items()
    },
    "choice_common_question_ids": choice_common_ids,
    "paired_choice_positive_rates": paired_choice_rates,
    "quality_score_common_question_ids": quality_score_common_ids,
    "quality_paired_means": quality_paired_means,
    "paired_choice_changes_from_zero": [row for row in choice_changes if row["left"] == 0],
    "paired_choice_changes": choice_changes,
    "trajectory_diagnostics": trajectory_diagnostics,
    "pairwise_changes": pairwise,
    "quality_valid_pairwise_changes": quality_pairwise,
    "quality_by_coeff": quality_summary,
    "invalid_scores": invalid_scores,
    "unrecognized_choices": unrecognized_choices,
    "score_probability_diagnostics": {
        "valid_calls": len(valid_diagnostics),
        "near_point_mass_calls": point_count,
    },
    "diagnostic_thresholds": {
        "monotonic_tolerance": MONOTONIC_TOLERANCE,
        "responsive_delta": MIN_RESPONSIVE_DELTA,
        "strong_delta": MIN_STRONG_DELTA,
    },
}

summary_path = artifact(run_dir, f"{TRAIT}_judge_summary_pv.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
pending_quality = [item for item in scored_results if item["quality_diagnostics"][-1].get("status") == "uncertain"]
artifact(run_dir, f"{TRAIT}_eval_quality_review.json").write_text(
    json.dumps(pending_quality, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n" + "=" * 15 + " 质量诊断 " + "=" * 15)
print(
    f"无效分数: {len(invalid_scores)} | "
    f"未识别明确选择: {len(unrecognized_choices)} | "
    f"每条最多尝试（含首次调用）: {SCORING_CONFIG['max_score_attempts']}"
)
print(f"特质评分候选概率近单点: {point_count}/{len(valid_diagnostics)} 次；近单点时加权结果基本等于一个整数。")
print("=" * 45)
print(f"完整评分结果：{output_path}")
print(f"汇总诊断结果：{summary_path}")

if SAVE_EFFECT_FIGURE:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    axes[0].plot(coefficients, [paired_means[c] for c in coefficients], "o--", color="#888888",
        label=f"All scored questions (N={len(complete_question_ids)})")
    axes[0].plot(coefficients, [quality_paired_means[str(c)] for c in coefficients], "o-", color="#2878B5",
        label=f"Same quality-passed questions (N={len(quality_score_common_ids)})")
    axes[0].set(title="Trait expression", ylabel="Score (0–100)", ylim=(0, 100))
    axes[0].legend(fontsize=8, loc="best")
    axes[1].plot(coefficients, [paired_choice_rates[str(c)] for c in coefficients], "o-", color="#E68A2E")
    axes[1].set(title=f"Improvement-side choices (same {len(choice_common_ids)} questions)",
        ylabel="Choice rate", ylim=(0, 1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.set(xlabel="Actual steering coefficient α", xticks=coefficients)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(TRAIT)
    for extension in ("png", "pdf"):
        fig.savefig(artifact(run_dir, f"{TRAIT}_coefficient_effects.{extension}"), dpi=FIGURE_DPI)
    plt.close(fig)
