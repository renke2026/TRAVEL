# -*- coding: utf-8 -*-
"""流水线产物路径管理（所有阶段共享）。

设计目标：
- 每跑一轮流水线，产物统一放进 runs/<时间戳>_<特质>/ 文件夹；
- 阶段①调用 new_run() 新建文件夹，并用 latest.txt 指针记录"当前 run"；
- 阶段②~⑤调用 current_run() 自动定位最新一次 run 文件夹；
- 如需指定已有结果，直接修改文件开头的 RUN_DIR。

用法（每个阶段文件顶部 import）：
    from run_paths import new_run, current_run, artifact

    run_dir = new_run()       # 仅阶段①（新建本次运行目录）
    # 或
    run_dir = current_run()   # 阶段②~⑤（定位当前最新目录）
"""

# Portable paths for the public research release.
import os as _TRAVEL_OS
import sys as _TRAVEL_SYS
from pathlib import Path as _TRAVEL_PATH
from travel_config import MODEL_PATH as _TRAVEL_MODEL_PATH, LLAMA_MODEL_PATH as _TRAVEL_LLAMA_PATH, QWEN3B_MODEL_PATH as _TRAVEL_QWEN3B_PATH

import json
from datetime import datetime
from pathlib import Path
from methology.persona.trait_instructions import FRAMEWORK_VERSION, TRAITS, TRAIT_DEFINITIONS, NEUTRAL_SYSTEM_PROMPT, get_response_rules, get_neutral_system_prompt
from methology.persona.build_trait_artifacts import fingerprint

# ==================== 在这里修改本轮实验 ====================
RUN_DIR = _TRAVEL_OS.environ.get("TRAVEL_RUN_DIR", "")           # 留空：使用最近一次结果；也可填写已有结果文件夹名称或完整路径
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = PROJECT_ROOT / "runs"
LATEST_POINTER = RUNS_ROOT / "latest.txt"

# 维度在第一步开头选择。后续脚本启动时，从所选结果目录读取，不需再改一遍。
TRAIT = "comfort"  # 尚无本轮结果时的占位；第一步会用自己的TRAIT覆盖它。
selected_run = Path(RUN_DIR) if RUN_DIR else None
if selected_run is None and LATEST_POINTER.exists():
    selected_run = Path(LATEST_POINTER.read_text(encoding="utf-8").strip())
if selected_run is not None:
    if not selected_run.is_absolute():
        selected_run = RUNS_ROOT / selected_run
    manifest_path = selected_run / "run_manifest.json"
    if manifest_path.exists():
        selected_trait = json.loads(manifest_path.read_text(encoding="utf-8")).get("trait")
        if selected_trait in TRAITS:
            TRAIT = selected_trait


def trait_data_path(split):
    if split not in {"extract", "eval"}:
        raise ValueError("split 必须为 extract 或 eval。")
    return PROJECT_ROOT / "methology/persona/data_generation" / f"trait_data_{split}" / f"{TRAIT}.json"


def load_trait_data(split):
    path = trait_data_path(split)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("framework_version") != FRAMEWORK_VERSION or data.get("trait") != TRAIT:
        raise ValueError(f"{path} 与当前构念框架不匹配，请完整重新生成数据。")
    if data.get("data_fingerprint") != fingerprint(data):
        raise ValueError(f"{path} 的内容与指纹不匹配，请通过数据入口重新生成。")
    return data


def record_metadata(split):
    data = load_trait_data(split)
    return {"framework_version": FRAMEWORK_VERSION, "trait": TRAIT,
            "data_fingerprint": data["data_fingerprint"]}


def assert_record_compatibility(records, split):
    expected = record_metadata(split)
    if any(any(item.get(key) != value for key, value in expected.items()) for item in records):
        raise ValueError("输入记录属于不同框架、维度或数据版本；请从第一阶段建立新实验。")


def validate_run(run_dir):
    path = run_dir / "run_manifest.json"
    if not path.exists():
        raise ValueError("历史实验缺少当前框架清单，不能重用；请重新运行第一阶段。")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("framework_version") != FRAMEWORK_VERSION or manifest.get("trait") != TRAIT:
        raise ValueError("当前结果不是本轮五维材料生成的实验，请先运行第一步建立新实验。旧十极点结果保留在原目录，不会覆盖。")
    for split in ("extract", "eval"):
        if manifest.get(f"{split}_data_fingerprint") != load_trait_data(split)["data_fingerprint"]:
            raise ValueError("题目或提示已变更，请建立新实验，不能与历史向量和回答混用。")
    return run_dir


def save_run_prompts(run_dir, extract_data, eval_data, backfilled=False):
    """保存本轮实际提示全文；JSON便于恢复，TXT便于直接阅读。"""
    from methology.persona.trait_instructions import COMMON_CONSTRAINTS
    from persona_judge_utils import scoring_config

    run_dir = Path(run_dir)
    response_rules = get_response_rules(extract_data["trait"])
    neutral_prompt = (run_dir / "neutral_system_prompt.txt").read_text(encoding="utf-8")
    snapshot = {
        "run": run_dir.name, "trait": extract_data["trait"],
        "framework_version": extract_data["framework_version"],
        "saved_when": "backfilled_after_run" if backfilled else "run_start",
        "extract_data_fingerprint": extract_data["data_fingerprint"],
        "eval_data_fingerprint": eval_data["data_fingerprint"],
        "trait_definition": extract_data["trait_definition"],
        "generation": {
            "common_constraints": COMMON_CONSTRAINTS,
            "response_rules": response_rules,
            "response_rules_location": "user_question_suffix" if extract_data["trait"] == "flexibility" else "system",
            "extraction_system_prompts": extract_data["instruction"],
            "extraction_questions": extract_data["questions"],
            "evaluation_system_prompt": neutral_prompt,
            "evaluation_questions": eval_data["questions"],
        },
        # scoring_config只导出评分/质检参数和提示，不包含API密钥。
        "judge": {"extract": scoring_config(extract_data["eval_prompt"]),
                  "eval": scoring_config(eval_data["eval_prompt"])},
    }
    snapshot_path = run_dir / "prompts_snapshot.json"
    if snapshot_path.exists() and json.loads(snapshot_path.read_text(encoding="utf-8")) != snapshot:
        raise ValueError("本轮已有不同版本的提示归档，不能用后来修改的提示覆盖。")

    note = ("运行结束后补存；已核对本轮材料与补存版本一致。" if backfilled
            else "创建本轮实验时保存；后续编辑项目提示不会修改此归档。")
    lines = [f"本次运行的完整Prompts：{run_dir.name}", note,
             "第一步生成对照回答时：对应正/负提示作为system，提取题作为user。",
             "第三步读取隐藏状态时：RESPONSE_CONTEXT=profile保留原角色前缀；neutral用下面同一中性system重编码原回答Token。prompt_last始终来自原角色提示。实际选择另存于向量metadata。",
             "评测时：共同作答系统提示作为system，评测题作为user。",
             "评分/质检时：将题目和回答填入下列模板，作为user发送给裁判。",
             "本文件只归档材料；复现时须恢复这些材料，并保持模型和运行参数一致。", ""]
    if extract_data["trait"] == "flexibility":
        lines += ["本轮灵活性：角色和共同事实约束在system；作答要求位于每道user题目末尾，不再重复放进system。Rubric只发送给裁判。", ""]

    def section(title, text):
        lines.extend(["=" * 60, title, "=" * 60, text, ""])

    section("一、维度定义", json.dumps(snapshot["trait_definition"], ensure_ascii=False, indent=2))
    section("二、共同事实约束", COMMON_CONSTRAINTS)
    section("三、提取回答的共同作答要求（评测完整提示另列）", response_rules)
    section("四、评测时实际使用的完整作答系统提示", neutral_prompt)
    for index, pair in enumerate(extract_data["instruction"], 1):
        section(f"五、提取用五对系统提示：第{index}对／正向（完整system）", pair["pos"])
        section(f"五、提取用五对系统提示：第{index}对／负向（完整system）", pair["neg"])
    for split, title in (("extraction_questions", "六、提取题"), ("evaluation_questions", "七、评测题")):
        for index, question in enumerate(snapshot["generation"][split]):
            section(f"{title}：题号{index}", question)
    for split, title in (("extract", "八、第二步筛选Rubric"), ("eval", "九、第五／第六步评分Rubric")):
        section(title, snapshot["judge"][split]["rubric_template"])
    judge = snapshot["judge"]["extract"]
    section("十、独立事实一致性质检提示", judge["quality_prompt"])
    section("十一、质检独立复核时追加在最前面的文字", judge["quality_review_prefix"])
    section("十二、特质评分格式失败重试时追加的文字", judge["retry_format_reminder"])
    if "quality_retry_reminder" in judge:
        section("十二附、质检JSON或证据引文失败重试时追加的文字", judge["quality_retry_reminder"])
    section("十三、裁判参数与完整模板记录", json.dumps(snapshot["judge"], ensure_ascii=False, indent=2))

    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "本次运行Prompts.txt").write_text("\n".join(lines), encoding="utf-8")


def new_run(label=None):
    """阶段①专用：新建以时间命名的产物文件夹，并登记为当前 run。"""
    extract_data, eval_data = load_trait_data("extract"), load_trait_data("eval")
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    label = label or TRAIT
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{stamp}_{label}"
    run_dir = RUNS_ROOT / base

    # 同一秒内重复运行时追加序号，避免目录冲突
    counter = 1
    while run_dir.exists():
        counter += 1
        run_dir = RUNS_ROOT / f"{base}_{counter}"

    run_dir.mkdir(parents=True, exist_ok=False)
    # 每轮保留实际使用的材料，方便查看和比较实验。
    for split, data in (("extract", extract_data), ("eval", eval_data)):
        (run_dir / f"trait_data_{split}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "neutral_system_prompt.txt").write_text(get_neutral_system_prompt(TRAIT), encoding="utf-8")
    save_run_prompts(run_dir, extract_data, eval_data)
    manifest = {"framework_version": FRAMEWORK_VERSION, "trait": TRAIT,
                "dimension": TRAIT_DEFINITIONS[TRAIT]["dimension"],
                "vector_direction": "high_attention_minus_low_attention",
                "extract_data_fingerprint": extract_data["data_fingerprint"],
                "eval_data_fingerprint": eval_data["data_fingerprint"],
                "source": extract_data["source"]}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    LATEST_POINTER.write_text(run_dir.name, encoding="utf-8")
    return run_dir


def current_run():
    """阶段②~⑤专用：定位当前（最新）产物文件夹。"""
    # 1) 优先使用显式指定的 run 目录（历史回放场景）
    if RUN_DIR:
        run_dir = Path(RUN_DIR)
        if not run_dir.is_absolute():
            run_dir = RUNS_ROOT / run_dir
        if not run_dir.is_dir():
            raise FileNotFoundError(
                f"run_paths.py 中 RUN_DIR 指定的文件夹不存在: {run_dir}"
            )
        return validate_run(run_dir)

    # 2) 否则读取 latest.txt 指针
    if not LATEST_POINTER.exists():
        raise FileNotFoundError(
            "找不到 runs/latest.txt，请先运行阶段① (1_pt_shengcheng.py) 创建产物文件夹。"
        )
    name = LATEST_POINTER.read_text(encoding="utf-8").strip()
    run_dir = RUNS_ROOT / name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"latest.txt 指向的文件夹不存在: {run_dir}")
    return validate_run(run_dir)


def artifact(run_dir, filename):
    """在 run 文件夹内拼接产物完整路径。"""
    return run_dir / filename
