"""⑥ 比较不同注入层和alpha，画出每层的平均特质得分。

先完成第三步的向量提取，即可单独运行本文件；不要求先运行第四、第五步。
使用所选运行目录保存的评测题、Rubric和第五步相同的裁判函数。
"""

# Portable paths for the public research release.
import os as _TRAVEL_OS
import sys as _TRAVEL_SYS
from pathlib import Path as _TRAVEL_PATH
from travel_config import MODEL_PATH as _TRAVEL_MODEL_PATH, LLAMA_MODEL_PATH as _TRAVEL_LLAMA_PATH, QWEN3B_MODEL_PATH as _TRAVEL_QWEN3B_PATH


# ==================== 在这里修改实验参数 ====================
RUN_DIR = ""  # 留空：跟随run_paths.py；指定旧实验，例如 "20261018_155041_environment"
# 维度、题目和Rubric读取所选目录的归档；下面的层数、强度等是本次扫描参数。
# 裁判模型、评分温度等：persona_judge_utils.py 开头。
MODEL_PATH = _TRAVEL_MODEL_PATH  # 与第一、第三步一致
LOAD_IN_8BIT = True
WEIGHT_READ_BACKEND = "pread"     # Windows稳定读取；只改读取方式，不改权重和量化
VECTOR_METHOD = "response_mean"     # 提取上下文见第三步及向量metadata
STEERING_POSITIONS = "all"       # 本轮flexibility：读题与生成阶段均注入
LAYERS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]  # 五维统一的实际层号
COEFFICIENTS = [-1, 0, 1, 2, 3] # 五维统一；实际alpha，无额外倍率
QUESTION_IDS = None                 # None：全部评测题；或 [0, 1, 2, 3]，题号从0开始
MAX_NEW_TOKENS = 160
DO_SAMPLE = False
TEMPERATURE = 1.0                   # 仅 DO_SAMPLE=True 时使用
TOP_P = 1.0                         # 仅 DO_SAMPLE=True 时使用
RANDOM_SEED = 20261017
BATCH_SIZE = 32                     # 与第一步相同的批量上限；每层每个alpha的20题一起生成
OUTPUT_FOLDER = "layer_scan_response_mean_all_batch32"
GENERATE_ANSWERS = True             # 已有回答、只想重新评分时可改为False
SCORE_ANSWERS = True                # 已有评分、只想重新画图时两项都改为False
REUSE_COMPLETE_SCORES = True        # 完整旧评分直接复用；确实要按当前裁判重评时改为False
PAIRED_ONLY = False                # True：两张图分别仅使用所有层×alpha均有效的共同题
FIGURE_SIZE = (9, 5.5)
FIGURE_DPI = 300
FIGURE_TITLE = ""                  # 留空用维度名称；也可填写中文标题
SHOW_FIGURE = False                # True：保存后打开绘图窗口
# ==========================================================

import csv
import gc
import hashlib
import json
import math
import time
from pathlib import Path
from statistics import mean

import torch
from steering_utils import SteeringHook, encode_steering_batch

from run_paths import TRAIT, RUNS_ROOT, LATEST_POINTER, RUN_DIR as SHARED_RUN_DIR
from methology.persona.build_trait_artifacts import fingerprint
from persona_judge_utils import (
    create_deepseek_client, scoring_config, score_answer, assess_answer_quality,
    extract_selected_option,
)


def save_json(path, data):
    """及时保存；中途停止后可再次运行，接着处理。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    # Windows上的读取/索引可能短暂占用目标文件，等待释放后再原子替换。
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1)


def load_run_materials():
    """选择一次实验，读取它自己的材料，不与后来修改的题库混用。"""
    selected = RUN_DIR or SHARED_RUN_DIR
    if not selected:
        if not LATEST_POINTER.exists():
            raise FileNotFoundError("找不到runs/latest.txt，请先运行第一步，或在本文件开头填写RUN_DIR。")
        selected = LATEST_POINTER.read_text(encoding="utf-8").strip()
    run_dir = Path(selected)
    if not run_dir.is_absolute():
        run_dir = RUNS_ROOT / run_dir
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    materials = {}
    for split in ("extract", "eval"):
        path = run_dir / f"trait_data_{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"缺少本轮归档材料：{path}；不能用当前题库代替旧题库。")
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data.get("trait") != manifest.get("trait")
                or data.get("framework_version") != manifest.get("framework_version")
                or data.get("data_fingerprint") != manifest.get(f"{split}_data_fingerprint")
                or data.get("data_fingerprint") != fingerprint(data)):
            raise ValueError(f"本轮归档材料与清单不一致：{path}")
        materials[split] = data
    return run_dir, materials["extract"], materials["eval"]


def material_metadata(data):
    return {key: data[key] for key in ("framework_version", "trait", "data_fingerprint")}


def complete_saved_scores(path, answers, rubric):
    """只复用与全部实际回答对应的完整旧评分，不因裁判后来修改而偷偷重打。"""
    if not path.exists():
        return None
    scores = json.loads(path.read_text(encoding="utf-8"))
    key = lambda r: (r["physical_layer"], r["coeff"], r["question_id"])
    lookup = {key(r): r for r in scores}
    if not scores or len(lookup) != len(scores) or len(scores) != len(answers):
        return None
    config = scores[0].get("scoring_config", {})
    if config.get("rubric_template") != rubric or any(r.get("scoring_config") != config for r in scores):
        return None
    if any(key(r) not in lookup or any(lookup[key(r)].get(field) != r.get(field)
           for field in ("question", "response")) for r in answers):
        return None
    return scores




def generate_answers(path, settings, vectors, eval_data):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    existing = {(r["physical_layer"], r["coeff"], r["question_id"]): r for r in rows}
    expected = len(settings["layers"]) * len(settings["coefficients"]) * len(settings["question_ids"])
    if len(existing) == expected:
        print("所有层和alpha的回答已保存，直接进入评分。")
        return rows
    from model_loading import load_local_model
    model = load_local_model(MODEL_PATH, LOAD_IN_8BIT, WEIGHT_READ_BACKEND)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if list(vectors.shape) != [model.config.num_hidden_layers + 1, model.config.hidden_size]:
        raise ValueError("向量形状与模型不匹配，请检查MODEL_PATH。")
    device = model.get_input_embeddings().weight.device
    kwargs = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": DO_SAMPLE, "use_cache": True,
              "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id}
    if DO_SAMPLE:
        kwargs.update(temperature=TEMPERATURE, top_p=TOP_P)

    for layer in settings["layers"]:
        for coeff in settings["coefficients"]:
            print(f"生成：第{layer}层 | alpha={coeff:g}")
            # 无干预不存在层的区别；各层共用同一批alpha=0回答，不重复采样。
            baseline = {r["question_id"]: r for r in rows if r["coeff"] == 0}
            if coeff == 0 and baseline:
                for qid in settings["question_ids"]:
                    if (layer, coeff, qid) not in existing and qid in baseline:
                        row = dict(baseline[qid], physical_layer=layer, target_layer=layer - 1, vector_index=layer,
                                   vector_norm=vectors[layer].norm().item())
                        rows.append(row)
                        existing[(layer, coeff, qid)] = row
            missing = [qid for qid in settings["question_ids"] if (layer, coeff, qid) not in existing]
            with SteeringHook(model, layer, vectors[layer], coeff, STEERING_POSITIONS) as hook:
                for begin in range(0, len(missing), BATCH_SIZE):
                    ids = missing[begin:begin + BATCH_SIZE]
                    # 每批使用固定种子，再次运行时可接着生成；贪心生成不受种子影响。
                    set_seed(RANDOM_SEED + ids[0])
                    texts = [tokenizer.apply_chat_template([
                        {"role": "system", "content": settings["system_prompt"]},
                        {"role": "user", "content": eval_data["questions"][qid]},
                    ], tokenize=False, add_generation_prompt=True) for qid in ids]
                    inputs, system_mask = encode_steering_batch(tokenizer, texts, settings["system_prompt"], STEERING_POSITIONS)
                    inputs = inputs.to(device)
                    hook.system_mask = system_mask
                    prompt_length = inputs["input_ids"].shape[1]
                    with torch.inference_mode():
                        outputs = model.generate(**inputs, **kwargs)
                    for batch_index, (qid, output) in enumerate(zip(ids, outputs)):
                        new_ids = output[prompt_length:].tolist()
                        ended = tokenizer.eos_token_id in new_ids
                        if ended:
                            new_ids = new_ids[:new_ids.index(tokenizer.eos_token_id)]
                        row = {**material_metadata(eval_data), "question_id": qid,
                               "question": eval_data["questions"][qid],
                               "positive_option": eval_data.get("positive_options", [None]*len(eval_data["questions"]))[qid],
                               "physical_layer": layer,
                               "target_layer": layer - 1, "vector_index": layer,
                               "vector_method": VECTOR_METHOD, "vector_norm": vectors[layer].norm().item(),
                               "coeff": coeff, "coefficient_definition": "coeff_is_actual_multiplier_of_saved_vector",
                               "intervention_scope": STEERING_POSITIONS,
                               "system_content_tokens": int(system_mask[batch_index].sum()) if system_mask is not None else 0,
                               "generation_kwargs": kwargs, "generation_batch_size": BATCH_SIZE,
                               "weight_read_backend": WEIGHT_READ_BACKEND,
                               "generation_actual_batch_size": len(ids),
                               "response": tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
                               "response_new_tokens": len(new_ids), "response_hit_token_limit": not ended}
                        rows.append(row)
                        existing[(layer, coeff, qid)] = row
                    save_json(path, rows)
                    print(f"  已保存 {len(rows)}/{expected} 条回答")
            save_json(path, rows)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def judge_answers(path, rows, rubric):
    """和第五步一样：特质评分与事实质检分别进行。"""
    config = scoring_config(rubric)
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    # 同一道题、同一回答只请求一次；不同层出现相同回答时共用结果。
    judged = {(r["question"], r["response"]): r for r in previous if r.get("scoring_config") == config
              and r.get("score") is not None}
    client = None
    results = []
    for index, row in enumerate(rows):
        key = (row["question"], row["response"])
        cached = judged.get(key)
        if cached is None:
            if client is None:
                client = create_deepseek_client()
            quality, quality_details = assess_answer_quality(client, *key)
            score, details = score_answer(client, rubric, *key)
            cached = {"score": score, "quality_consistent": quality,
                      "quality_diagnostics": quality_details, "score_diagnostics": details, "scoring_config": config}
            judged[key] = cached
        quality_ok = cached["quality_consistent"] and not row.get("response_hit_token_limit", False)
        result = dict(row)
        result.update({key: cached[key] for key in ("score", "quality_consistent", "quality_diagnostics",
                                                   "score_diagnostics", "scoring_config")})
        result.update(quality_ok=quality_ok, quality_valid_score=cached["score"] if quality_ok else None)
        selected = extract_selected_option(row["response"], row["question"])
        result.update(selected_option=selected, positive_choice=(selected == row.get("positive_option")
            if quality_ok and selected and row.get("positive_option") else None))
        results.append(result)
        print(f"评分 {index + 1}/{len(rows)} | 层{row['physical_layer']} | alpha={row['coeff']:g} | "
              f"分数={result['score']} | 质检={'通过' if quality_ok else '未通过'}")
        if (index + 1) % 20 == 0 or index + 1 == len(rows):
            save_json(path, results)
    return results


def summarize(rows, layers, coefficients, question_ids, score_field):
    cells = {(layer, coeff): {} for layer in layers for coeff in coefficients}
    for row in rows:
        cell = cells[(row["physical_layer"], row["coeff"])]
        if row["question_id"] in cell:
            raise ValueError("同一层、alpha和题号存在重复记录，不能重复计入平均分。")
        cell[row["question_id"]] = row.get(score_field)
    common = set(question_ids)
    for cell in cells.values():
        common &= {qid for qid, score in cell.items() if score is not None}
    summary = []
    for (layer, coeff), cell in cells.items():
        used = [qid for qid in question_ids if cell.get(qid) is not None and (not PAIRED_ONLY or qid in common)]
        scores = [cell[qid] for qid in used]
        summary.append({"physical_layer": layer, "coeff": coeff, "score_field": score_field,
                        "mean_score": mean(scores) if scores else None, "n": len(scores),
                        "total_questions": len(question_ids), "common_questions": len(common),
                        "question_ids": used})
    return summary


def summarize_choices(rows, layers, coefficients, question_ids):
    """每层内固定相同题目比较选择率，同时保留各组可用比例。"""
    lookup = {(r["physical_layer"], r["coeff"], r["question_id"]): r for r in rows}
    results = []
    for layer in layers:
        common = [qid for qid in question_ids if all(
            lookup.get((layer, coeff, qid), {}).get("positive_choice") is not None for coeff in coefficients)]
        for coeff in coefficients:
            cell = [lookup.get((layer, coeff, qid), {}) for qid in question_ids]
            selected = [r.get("selected_option") or extract_selected_option(r.get("response", ""), r.get("question")) for r in cell]
            raw = [s == r.get("positive_option") for s, r in zip(selected, cell)
                   if s in ("A", "B") and r.get("positive_option") in ("A", "B")]
            available = [lookup[layer, coeff, qid]["positive_choice"] for qid in question_ids
                         if lookup.get((layer, coeff, qid), {}).get("positive_choice") is not None]
            choices = [lookup[layer, coeff, qid]["positive_choice"] for qid in common]
            results.append({"physical_layer": layer, "coeff": coeff, "score_field": "positive_choice",
                "mean_score": mean(choices) if choices else None, "n": len(common), "question_ids": common,
                "available_choice_rate": mean(available) if available else None, "available_n": len(available),
                "total_n": len(question_ids), "a_count": selected.count("A"), "b_count": selected.count("B"),
                "unrecognized_count": sum(s not in ("A", "B") for s in selected),
                "raw_high_count": sum(raw), "raw_low_count": len(raw)-sum(raw), "raw_n": len(raw),
                "raw_choice_rate": mean(raw) if raw else None,
                "raw_high_rate_all_questions": sum(raw)/len(question_ids) if question_ids else None,
                "quality_pass_n": sum(r.get("quality_ok", False) for r in cell),
                "available_high_count": sum(available), "paired_high_count": sum(choices),
                "paired_low_count": len(choices)-sum(choices)})
    return results


def draw_figure(summary, settings, output, score_field):
    import matplotlib
    if not SHOW_FIGURE:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import Normalize

    fonts = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if name in fonts:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, layout="constrained")
    coefficients = settings["coefficients"]
    color_scale = Normalize(min(coefficients), max(coefficients))
    for coeff in coefficients:
        line = [r for r in summary if r["coeff"] == coeff]
        ys = [r["mean_score"] if r["mean_score"] is not None else float("nan") for r in line]
        color = "#777777" if coeff == 0 else plt.colormaps["viridis"](color_scale(coeff))
        ax.plot([r["physical_layer"] for r in line], ys, marker="o", linewidth=2,
                color=color, linestyle="--" if coeff == 0 else "-", label=f"α = {coeff:g}")
    title = FIGURE_TITLE or TRAIT
    is_choice = score_field in ("positive_choice", "positive_choice_raw")
    subtitle = ("Improvement-side choices · before quality filtering" if score_field == "positive_choice_raw"
                else "Improvement-side choices · same questions within layer") if is_choice else ("Trait expression" if score_field == "score" else "Trait expression · quality passed")
    ax.set(title=f"{title}\n{subtitle}", xlabel="Steering layer",
           ylabel="Improvement-side choice rate" if is_choice else "Mean trait expression score (0-100)",
           ylim=(-0.02, 1.02) if is_choice else (-2, 102), xticks=settings["layers"])
    if is_choice:
        from matplotlib.ticker import PercentFormatter
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Steering coefficient", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    note = ("All explicit choices; denominators and unrecognized answers in choices_summary.csv" if score_field == "positive_choice_raw"
            else "Same valid questions across coefficients within each layer") if is_choice else ("Same valid questions in every cell" if PAIRED_ONLY else "Available scores per cell; sample counts are in summary.csv")
    fig.supxlabel(note, fontsize=8, color="#666666")
    stem = ("layer_choices_raw" if score_field == "positive_choice_raw" else "layer_choices_paired") if is_choice else ("layer_scores" if score_field == "score" else "layer_scores_quality_passed")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=FIGURE_DPI)
    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)


def main():
    global TRAIT
    run_dir, extract_data, eval_data = load_run_materials()
    TRAIT = eval_data["trait"]
    suffix = {"prompt_last": "prompt_last", "response_mean": "all_layers"}.get(VECTOR_METHOD)
    if suffix is None:
        raise ValueError("VECTOR_METHOD只能填prompt_last或response_mean。")
    vector_path = run_dir / f"{TRAIT}_vectors_{suffix}.pt"
    metadata = json.loads(vector_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if any(metadata.get(k) != v for k, v in material_metadata(extract_data).items()):
        raise ValueError("向量与所选目录的提取材料不一致，请检查结果目录。")
    snapshot = json.loads((run_dir / "prompts_snapshot.json").read_text(encoding="utf-8"))
    if (metadata.get("scoring_config") != snapshot["judge"]["extract"]
            or snapshot["judge"]["extract"].get("rubric_template") != extract_data["eval_prompt"]):
        raise ValueError("向量筛选记录与本轮归档裁判提示不一致，请检查结果目录。")
    if metadata.get("model_path") != MODEL_PATH:
        raise ValueError("MODEL_PATH与提取向量时不同，请使用同一模型。")
    vectors = torch.load(vector_path, map_location="cpu", weights_only=True).float()
    layers = sorted(set(LAYERS))
    coefficients = sorted(set(float(value) for value in COEFFICIENTS))
    ids = list(range(len(eval_data["questions"]))) if QUESTION_IDS is None else list(QUESTION_IDS)
    if vectors.ndim != 2 or not layers or any(type(layer) is not int or not 1 <= layer <= vectors.shape[0] - 1 for layer in layers):
        raise ValueError("LAYERS必须是实际层号，从1到模型总层数；向量第0行是嵌入层。")
    if vectors.shape[0] - 1 in layers and metadata.get("last_layer_state") != "before_final_norm":
        raise ValueError("扫描包含最后一层，但旧向量的最后一行经过最终RMSNorm，与注入位置不一致。"
                         "请重新运行更新后的3_extract_vector.py，再运行第六步；不必重跑第一、第二步。")
    if 0 not in coefficients or any(not math.isfinite(c) for c in coefficients):
        raise ValueError("COEFFICIENTS需为有限数字并包含0。")
    if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < len(eval_data["questions"]) for i in ids):
        raise ValueError("QUESTION_IDS需为不重复的有效题号，从0开始。")
    if BATCH_SIZE < 1 or MAX_NEW_TOKENS < 1:
        raise ValueError("BATCH_SIZE和MAX_NEW_TOKENS必须大于0。")
    prompt_file = run_dir / "neutral_system_prompt.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"缺少本轮归档作答提示：{prompt_file}；不能用当前提示代替。")
    if (VECTOR_METHOD == "response_mean" and metadata.get("response_context") == "neutral"
            and metadata.get("neutral_system_prompt") != prompt_file.read_text(encoding="utf-8")):
        raise ValueError("中性重编码向量与归档评测提示不同，请使用同一提示重新提取。")
    settings = {**material_metadata(eval_data), "model_path": MODEL_PATH, "load_in_8bit": LOAD_IN_8BIT,
                "weight_read_backend": WEIGHT_READ_BACKEND,
                "intervention_scope": STEERING_POSITIONS,
                "vector_method": VECTOR_METHOD, "vector_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
                "layers": layers, "coefficients": coefficients, "question_ids": ids,
                "system_prompt": prompt_file.read_text(encoding="utf-8"),
                "max_new_tokens": MAX_NEW_TOKENS, "do_sample": DO_SAMPLE, "temperature": TEMPERATURE,
                "top_p": TOP_P, "random_seed": RANDOM_SEED, "batch_size": BATCH_SIZE}
    if metadata.get("response_context") == "neutral":
        settings["response_context"] = "neutral"
    if not OUTPUT_FOLDER or Path(OUTPUT_FOLDER).name != OUTPUT_FOLDER or OUTPUT_FOLDER in (".", ".."):
        raise ValueError("OUTPUT_FOLDER请填写一个子文件夹名称。")
    output = run_dir / OUTPUT_FOLDER
    output.mkdir(exist_ok=True)
    settings_file = output / "settings.json"
    if settings_file.exists() and json.loads(settings_file.read_text(encoding="utf-8")) != settings:
        raise ValueError("此文件夹已有不同参数的结果，请修改开头的OUTPUT_FOLDER，避免混用。")
    save_json(settings_file, settings)
    save_json(output / "trait_data_eval.json", eval_data)
    answers_path, scores_path = output / "answers.json", output / "scores.json"
    expected = len(layers) * len(coefficients) * len(ids)
    print(f"维度：{TRAIT} | 方法：{VECTOR_METHOD} | 层：{layers} | alpha：{coefficients}")
    print(f"向量提取上下文：{metadata.get('response_context', 'profile')}")
    print(f"题目、Rubric和作答提示来自：{run_dir}")
    print("缺少评分时调用persona_judge_utils.py；完整旧评分默认复用，实际裁判参数保存至judge_config.json。")
    print(f"每组{len(ids)}题，共{expected}条记录；alpha=0共用回答。结果目录：{output}")
    print(f"批量上限：{BATCH_SIZE}；每层每个alpha的首批实际{min(BATCH_SIZE, len(ids))}条。")
    rows = generate_answers(answers_path, settings, vectors, eval_data) if GENERATE_ANSWERS else json.loads(answers_path.read_text(encoding="utf-8"))
    # 旧扫描记录没有辅助选项字段，可从同版本题库补齐，不重生成回答。
    for row in rows:
        if row.get("positive_option") is None and eval_data.get("positive_options"):
            row["positive_option"] = eval_data["positive_options"][row["question_id"]]
    expected_keys = {(layer, coeff, qid) for layer in layers for coeff in coefficients for qid in ids}
    if len(rows) != expected or {(r["physical_layer"], r["coeff"], r["question_id"]) for r in rows} != expected_keys:
        raise ValueError("回答尚未完成，请将GENERATE_ANSWERS改为True继续运行。")
    saved = complete_saved_scores(scores_path, rows, eval_data["eval_prompt"]) if REUSE_COMPLETE_SCORES else None
    if saved is not None:
        scored = saved
        used_judge_config = scored[0]["scoring_config"]
        print("已有完整且对应相同回答的评分，直接复用，不请求API；保留原裁判配置。")
    else:
        scored = judge_answers(scores_path, rows, eval_data["eval_prompt"]) if SCORE_ANSWERS else json.loads(scores_path.read_text(encoding="utf-8"))
        used_judge_config = scoring_config(eval_data["eval_prompt"])
    if len(scored) != expected or {(r["physical_layer"], r["coeff"], r["question_id"]) for r in scored} != expected_keys:
        raise ValueError("评分尚未完成，请将SCORE_ANSWERS改为True继续运行。")
    if any(r.get("scoring_config") != used_judge_config for r in scored):
        raise ValueError("评分参数已改变，请将SCORE_ANSWERS改为True重新评分。")
    save_json(output / "judge_config.json", used_judge_config)
    for row in scored:
        option = eval_data.get("positive_options", [None]*len(eval_data["questions"]))[row["question_id"]]
        selected = extract_selected_option(row["response"], row["question"])
        row.update(positive_option=option, selected_option=selected,
            positive_choice=(selected == option if row["quality_ok"] and selected and option else None))
    summary = []
    for field in ("score", "quality_valid_score"):
        part = summarize(scored, layers, coefficients, ids, field)
        summary.extend(part)
        draw_figure(part, settings, output, field)
        if any(r["n"] == 0 for r in part):
            print(f"提示：{field}部分位置没有有效分数，图中留空，没有填成0。")
    save_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    choices = summarize_choices(scored, layers, coefficients, ids)
    save_json(output / "choices_summary.json", choices)
    with (output / "choices_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(choices[0]))
        writer.writeheader()
        writer.writerows(choices)
    draw_figure(choices, settings, output, "positive_choice")
    raw_choices = [dict(r, mean_score=r["raw_choice_rate"], n=r["raw_n"]) for r in choices]
    draw_figure(raw_choices, settings, output, "positive_choice_raw")
    print(f"完成。曲线图PNG/PDF、各组均值及有效题数CSV已保存至：{output}")


if __name__ == "__main__":
    main()
