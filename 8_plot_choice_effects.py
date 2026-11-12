"""论文出行选择图：只读取已有最优候选层的结果，不生成回答、不请求裁判。"""

# Portable paths for the public research release.
import os as _TRAVEL_OS
import sys as _TRAVEL_SYS
from pathlib import Path as _TRAVEL_PATH
from travel_config import MODEL_PATH as _TRAVEL_MODEL_PATH, LLAMA_MODEL_PATH as _TRAVEL_LLAMA_PATH, QWEN3B_MODEL_PATH as _TRAVEL_QWEN3B_PATH


# ==================== 本节绘图参数 ====================
COEFFICIENTS = [-1, 0, 1, 2, 3]
QUESTION_COUNT = 20
RUNS = [
    # 维度、选用层、原实验目录、扫描子目录。沿用第七步选择，不为曲线好看再换层。
    ("comfort", 18, "20261018_141422_comfort", "layer_scan_prompt_last_batch64"),
    ("environment", 16, "20261018_161705_environment", "layer_scan_prompt_last_prompt_batch32"),
    ("safety", 14, "20261018_183835_safety", "layer_scan_response_mean_all_batch32"),
    ("convenience", 14, "20261018_212758_convenience", "layer_scan_response_mean_system_batch32"),
    ("flexibility", 18, "20261018_235055_flexibility", "layer_scan_response_mean_all_batch32"),
]
BASELINE_ALPHA = 0
COMPARISON_ALPHA = 3
SHOW_RAW_WILSON_INTERVALS = True  # 描述跨题差异；不是重复运行误差或选层校正后的区间。
FIGURE_DPI = 300
OUTPUT_DIR = "runs/五维出行选择干预图_20261019"
# ====================================================

import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW_COLOR = "#0072B2"
PAIRED_COLOR = "#D55E00"
HIGH_COLOR = "#009E73"
LOW_COLOR = "#7BAFD4"
INVALID_COLOR = "#BCC2C9"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson(high, total):
    if not total:
        return None, None
    z = 1.959963984540054
    rate = high / total
    center = (rate + z*z/(2*total)) / (1 + z*z/total)
    half = z * np.sqrt(rate*(1-rate)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return float(max(0, center-half)*100), float(min(1, center+half)*100)


def style(ax, ylabel="High-preference choice rate"):
    ax.set(xlabel="Steering coefficient", ylabel=ylabel, xticks=COEFFICIENTS, ylim=(-4, 108))
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.axvline(0, color="#777777", ls=":", lw=1, zorder=0)
    ax.grid(axis="y", alpha=.16)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig, out, stem):
    for extension in ("png", "svg"):
        fig.savefig(out / f"{stem}.{extension}", dpi=FIGURE_DPI)
    plt.close(fig)


def main():
    out = ROOT / OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "svg.fonttype": "none"})
    cells, summaries, transitions, questions, sources = [], [], [], [], {}

    for trait, layer, folder, scan_folder in RUNS:
        scan = ROOT / "runs" / folder / scan_folder
        score_path = scan / "scores.json"
        sources[str(score_path)] = digest(score_path)
        records = [r for r in read(score_path) if r["physical_layer"] == layer]
        lookup = {(float(r["coeff"]), r["question_id"]): r for r in records}
        expected = {(float(a), q) for a in COEFFICIENTS for q in range(QUESTION_COUNT)}
        if len(records) != len(expected) or set(lookup) != expected:
            raise ValueError(f"{trait}第{layer}层缺项或存在重复。")
        if not all(r.get("positive_option") in ("A", "B") for r in records):
            raise ValueError("题目缺少高偏好选项标注，不能按固定A/B统计。")
        for q in range(QUESTION_COUNT):
            if len({lookup[float(a), q]["question"] for a in COEFFICIENTS}) != 1:
                raise ValueError("同一题在不同alpha下发生变化。")
            if len({lookup[float(a), q]["positive_option"] for a in COEFFICIENTS}) != 1:
                raise ValueError("高偏好端标注随alpha变化。")

        # raw也按同题比较：只要求所有系数均能识别明确选择，不以质检通过为前提。
        raw_ids = [q for q in range(QUESTION_COUNT) if all(
            lookup[float(a), q].get("selected_option") in ("A", "B") for a in COEFFICIENTS)]
        paired_ids = [q for q in raw_ids if all(lookup[float(a), q]["quality_ok"] for a in COEFFICIENTS)]
        for alpha in COEFFICIENTS:
            rows = [lookup[float(alpha), q] for q in range(QUESTION_COUNT)]
            explicit = [r for r in rows if r.get("selected_option") in ("A", "B")]
            high = lambda r: r["selected_option"] == r["positive_option"]
            raw_high = sum(high(r) for r in explicit)
            matched_high = sum(high(lookup[float(alpha), q]) for q in raw_ids)
            paired_high = sum(high(lookup[float(alpha), q]) for q in paired_ids)
            valid_high = sum(r["quality_ok"] and high(r) for r in explicit)
            valid_low = sum(r["quality_ok"] and not high(r) for r in explicit)
            invalid = QUESTION_COUNT - valid_high - valid_low
            low_ci, high_ci = wilson(matched_high, len(raw_ids))
            scores = [r["score"] for r in rows if r["score"] is not None]
            row = dict(trait=trait, layer=layer, alpha=alpha,
                       score_mean=float(np.mean(scores)), score_n=len(scores),
                       raw_high=raw_high, raw_n=len(explicit), raw_percent=100*raw_high/len(explicit) if explicit else None,
                       raw_matched_high=matched_high, raw_matched_n=len(raw_ids),
                       raw_matched_percent=100*matched_high/len(raw_ids) if raw_ids else None,
                       raw_wilson_low=low_ci, raw_wilson_high=high_ci,
                       paired_high=paired_high, paired_n=len(paired_ids),
                       paired_percent=100*paired_high/len(paired_ids) if paired_ids else None,
                       valid_high=valid_high, valid_low=valid_low, invalid=invalid,
                       total_n=QUESTION_COUNT, valid_high_all_percent=100*valid_high/QUESTION_COUNT,
                       valid_low_all_percent=100*valid_low/QUESTION_COUNT,
                       invalid_all_percent=100*invalid/QUESTION_COUNT,
                       quality_pass_n=sum(r["quality_ok"] for r in rows),
                       quality_pass_percent=100*sum(r["quality_ok"] for r in rows)/QUESTION_COUNT)
            cells.append(row)
            for q, r in enumerate(rows):
                questions.append(dict(trait=trait, layer=layer, alpha=alpha, question_id=q,
                                      positive_option=r["positive_option"], selected_option=r.get("selected_option"),
                                      raw_high=high(r) if r.get("selected_option") in ("A", "B") else None,
                                      quality_ok=bool(r["quality_ok"]),
                                      in_raw_matched_set=q in raw_ids, in_quality_paired_set=q in paired_ids))

        trait_cells = [r for r in cells if r["trait"] == trait]
        first, last = trait_cells[0], trait_cells[-1]
        summaries.append(dict(trait=trait, layer=layer, raw_n=len(raw_ids), paired_n=len(paired_ids),
                              raw_endpoint_gain_pp=last["raw_matched_percent"]-first["raw_matched_percent"],
                              paired_endpoint_gain_pp=(last["paired_percent"]-first["paired_percent"]) if paired_ids else None,
                              valid_high_all_endpoint_gain_pp=last["valid_high_all_percent"]-first["valid_high_all_percent"],
                              raw_curve_percent=[r["raw_matched_percent"] for r in trait_cells],
                              paired_curve_percent=[r["paired_percent"] for r in trait_cells],
                              valid_high_all_curve_percent=[r["valid_high_all_percent"] for r in trait_cells],
                              invalid_all_curve_percent=[r["invalid_all_percent"] for r in trait_cells]))
        for q in range(QUESTION_COUNT):
            b, e = lookup[float(BASELINE_ALPHA), q], lookup[float(COMPARISON_ALPHA), q]
            if b.get("selected_option") in ("A", "B") and e.get("selected_option") in ("A", "B"):
                transitions.append(dict(trait=trait, question_id=q, baseline_alpha=BASELINE_ALPHA,
                                        comparison_alpha=COMPARISON_ALPHA,
                                        baseline_high=bool(high(b)), comparison_high=bool(high(e)),
                                        both_quality_passed=bool(b["quality_ok"] and e["quality_ok"])))

    # 01：同一组回答的评分和明确选择，给这一节的主要结果总览。
    fig, axes = plt.subplots(2, 5, figsize=(16.5, 6.6), layout="constrained")
    for i, (trait, layer, _, _) in enumerate(RUNS):
        values = [r for r in cells if r["trait"] == trait]
        for row_index, key, label in ((0, "score_mean", "Trait expression score"),
                                       (1, "raw_matched_percent", "High-preference choices")):
            ax = axes[row_index, i]
            ys = [r[key] for r in values]
            ax.plot(COEFFICIENTS, ys, "o-", color=RAW_COLOR, lw=2, ms=5)
            style(ax, label if i == 0 else "")
            if row_index == 0:
                ax.yaxis.set_major_formatter(plt.ScalarFormatter())
            ax.set_title(f"{trait.capitalize()} | layer {layer}")
            if row_index == 1:
                for x, y in zip(COEFFICIENTS, ys):
                    ax.annotate(f"{y:g}%", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    fig.supxlabel("Same 20 scenarios per trait across coefficients. Explicit choices before quality filtering.\n"
                  "Quality outcomes are reported separately; layer selection is exploratory.", fontsize=10)
    save_figure(fig, out, "01_score_and_choice")

    # 02：独立选择率主图。比例由所有明确选择得到，不能用质检删除提高曲线。
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), layout="constrained")
    for ax, (trait, layer, _, _) in zip(axes.flat, RUNS):
        values = [r for r in cells if r["trait"] == trait]
        ys = [r["raw_matched_percent"] for r in values]
        if SHOW_RAW_WILSON_INTERVALS:
            ax.fill_between(COEFFICIENTS, [r["raw_wilson_low"] for r in values],
                            [r["raw_wilson_high"] for r in values], color=RAW_COLOR, alpha=.1, linewidth=0)
        ax.plot(COEFFICIENTS, ys, "o-", color=RAW_COLOR, lw=2, ms=6)
        for x, y in zip(COEFFICIENTS, ys):
            ax.annotate(f"{y:g}%", (x, y), xytext=(0, 8), textcoords="offset points", ha="center")
        style(ax)
        ax.set_title(f"{trait.capitalize()} | layer {layer} | N = {values[0]['raw_matched_n']}")
    ax = axes.flat[-1]
    ax.axis("off")
    ax.legend(handles=[Line2D([0], [0], color=RAW_COLOR, marker="o", label="Explicit high-preference choices"),
                       Patch(color=RAW_COLOR, alpha=.1, label="95% Wilson interval across scenarios"),
                       Line2D([0], [0], color="#777777", ls=":", label="No steering (alpha = 0)")],
              loc="upper left", frameon=False, fontsize=10)
    ax.text(.02, .52, "Same 20 scenarios at every coefficient.\nNo quality-based exclusion from this curve.\n\n"
            "An explicit choice may still have an\ninconsistent explanation or violate a constraint.\nSee the quality-outcome figure.\n\n"
            "Intervals do not account for layer selection\nor repeated generation/judging uncertainty.",
            transform=ax.transAxes, va="top", fontsize=10, linespacing=1.5)
    fig.suptitle("Steering changes the model's stated travel choices", fontsize=16)
    save_figure(fig, out, "02_choice_rates_raw")

    # 03：同一固定分母的三分类，显示提高选择率是否伴随无效回答增加。
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), layout="constrained")
    for ax, (trait, layer, _, _) in zip(axes.flat, RUNS):
        values = [r for r in cells if r["trait"] == trait]
        bottom = np.zeros(len(COEFFICIENTS))
        for key, color in (("valid_high_all_percent", HIGH_COLOR),
                           ("valid_low_all_percent", LOW_COLOR), ("invalid_all_percent", INVALID_COLOR)):
            heights = np.array([r[key] for r in values])
            ax.bar(COEFFICIENTS, heights, bottom=bottom, width=.65, color=color, edgecolor="white", lw=.6)
            for x, y, b in zip(COEFFICIENTS, heights, bottom):
                if y >= 15:
                    ax.text(x, b+y/2, f"{y:g}%", ha="center", va="center", fontsize=10,
                            color="white" if color == HIGH_COLOR else "#202020")
            bottom += heights
        style(ax, "Share of all 20 answers")
        ax.set_title(f"{trait.capitalize()} | layer {layer}")
    ax = axes.flat[-1]
    ax.axis("off")
    ax.legend(handles=[Patch(color=HIGH_COLOR, label="Quality-passed high-preference choice"),
                       Patch(color=LOW_COLOR, label="Quality-passed low-preference choice"),
                       Patch(color=INVALID_COLOR, label="Quality failure or unidentified choice")],
              loc="upper left", frameon=False, fontsize=10)
    ax.text(.02, .48, "Every bar contains all 20 answers.\nThe denominator never changes.\n\n"
            "Quality failures remain a separate outcome;\nthey are not silently discarded or\nreclassified as low-preference choices.\n\n"
            "The green segment is successful high-\npreference choice per attempted scenario,\nnot preference conditional on validity.",
            transform=ax.transAxes, va="top", fontsize=10, linespacing=1.5)
    fig.suptitle("Choice outcomes with answer validity retained", fontsize=16)
    save_figure(fig, out, "03_quality_outcomes")

    # 04：raw与原layer_choices_paired的敏感性对照，不选择较好看的口径。
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), layout="constrained")
    for ax, (trait, layer, _, _) in zip(axes.flat, RUNS):
        values = [r for r in cells if r["trait"] == trait]
        ax.plot(COEFFICIENTS, [r["raw_matched_percent"] for r in values], "o-", color=RAW_COLOR, lw=2)
        if values[0]["paired_n"]:
            ax.plot(COEFFICIENTS, [r["paired_percent"] for r in values], "s--", color=PAIRED_COLOR, lw=2)
        style(ax)
        ax.set_title(f"{trait.capitalize()} | layer {layer}\nRaw N = {values[0]['raw_matched_n']}; quality-paired N = {values[0]['paired_n']}")
    ax = axes.flat[-1]
    ax.axis("off")
    ax.legend(handles=[Line2D([0], [0], color=RAW_COLOR, marker="o", label="Raw: all explicitly identified choices"),
                       Line2D([0], [0], color=PAIRED_COLOR, marker="s", ls="--", label="Paired: passes quality at every alpha")],
              loc="upper left", frameon=False, fontsize=10)
    ax.text(.02, .56, "Quality-paired curves use one fixed subset\nwithin each trait, but that subset is selected\nafter observing every intervention.\n\n"
            "Raw and paired curves answer different\nquestions. A higher paired rate does not\nmake it a better estimate for all scenarios.\n\n"
            "Primary analysis: raw plus quality outcomes.\nSensitivity analysis: quality-paired subset.",
            transform=ax.transAxes, va="top", fontsize=10, linespacing=1.5)
    fig.suptitle("Sensitivity to quality-based sample selection", fontsize=16)
    save_figure(fig, out, "04_raw_vs_quality_paired")

    # 05：同题从无干预到正向干预，直接展示选择改变，而非仅有文字得分变化。
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), layout="constrained")
    transition_summary = []
    for ax, (trait, layer, _, _) in zip(axes.flat, RUNS):
        rows = [r for r in transitions if r["trait"] == trait]
        matrix = np.zeros((2, 2), dtype=int)
        quality_matrix = np.zeros((2, 2), dtype=int)
        for r in rows:
            matrix[int(r["baseline_high"]), int(r["comparison_high"])] += 1
            if r["both_quality_passed"]:
                quality_matrix[int(r["baseline_high"]), int(r["comparison_high"])] += 1
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=QUESTION_COUNT)
        for i in range(2):
            for j in range(2):
                ax.text(j, i-.08, str(matrix[i, j]), ha="center", va="center", fontsize=20,
                        color="white" if matrix[i, j] >= 12 else "#222222")
                ax.text(j, i+.17, f"QC: {quality_matrix[i, j]}", ha="center", va="center", fontsize=10,
                        color="white" if matrix[i, j] >= 12 else "#222222")
        ax.set_xticks([0, 1], ["Low\npreference", "High\npreference"])
        ax.set_yticks([0, 1], ["Low preference", "High preference"])
        ax.set(xlabel=f"Choice at alpha = {COMPARISON_ALPHA}", ylabel=f"Choice at alpha = {BASELINE_ALPHA}",
               title=f"{trait.capitalize()} | layer {layer} | N = {len(rows)}")
        ax.spines[:].set_visible(False)
        transition_summary.append(dict(trait=trait, layer=layer, n=len(rows), low_to_low=int(matrix[0, 0]),
                                       low_to_high=int(matrix[0, 1]), high_to_low=int(matrix[1, 0]),
                                       high_to_high=int(matrix[1, 1]),
                                       both_quality_passed_n=sum(r["both_quality_passed"] for r in rows),
                                       quality_passed_low_to_high=sum(not r["baseline_high"] and r["comparison_high"] and r["both_quality_passed"] for r in rows),
                                       quality_passed_high_to_low=sum(r["baseline_high"] and not r["comparison_high"] and r["both_quality_passed"] for r in rows)))
    ax = axes.flat[-1]
    ax.axis("off")
    ax.text(.02, .85, "Counts for the same scenarios.\nRows: no steering.\nColumns: positive steering.\n\n"
            "Upper-right: low-to-high switches.\nLower-left: high-to-low switches.\n\n"
            "Large number: all explicit choices.\nQC: both answers pass quality.\n\n"
            "All explicit choices are retained.\nQuality-passed counts are also\nrecorded separately in the CSV.",
            transform=ax.transAxes, va="top", fontsize=10, linespacing=1.6)
    fig.suptitle("Within-scenario choice transitions", fontsize=16)
    save_figure(fig, out, "05_choice_transitions")

    for name, rows in (("choice_curve_data", cells), ("choice_summary", summaries),
                       ("question_choices", questions), ("choice_transitions", transitions),
                       ("transition_summary", transition_summary)):
        save_json(out / (name + ".json"), rows)
        save_csv(out / (name + ".csv"), rows)
    human = [{"维度": r["trait"], "层数": r["layer"], "alpha": r["alpha"],
              "原始明确选择高端数": r["raw_high"], "原始明确选择分母": r["raw_n"],
              "原始高偏好选择率(%)": r["raw_percent"],
              "质检共同配对高端数": r["paired_high"], "质检共同配对题数": r["paired_n"],
              "质检共同配对高偏好选择率(%)": r["paired_percent"],
              "有效高偏好选择数量": r["valid_high"], "有效低偏好选择数量": r["valid_low"],
              "质检失败或未识别数量": r["invalid"], "全部题数": r["total_n"],
              "有效高偏好占全部题比例(%)": r["valid_high_all_percent"]} for r in cells]
    save_csv(out / "选择率与有效性统计.csv", human)
    if any(digest(Path(path)) != checksum for path, checksum in sources.items()):
        raise ValueError("分析期间原始评分文件发生变化。")
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    save_json(out / "figure_protocol.json", dict(coefficients=COEFFICIENTS, selected_layers=[(r[0], r[1]) for r in RUNS],
              baseline_alpha=BASELINE_ALPHA, comparison_alpha=COMPARISON_ALPHA,
              selection_source="runs/五维层数与余弦分析_20261019/best_layers.json",
              primary="Raw explicit choices on a fixed question set, accompanied by fixed-denominator validity outcomes.",
              sensitivity="Same questions passing quality checks at all five coefficients; post-intervention selected subset.",
              source_files_sha256=sources, source_files_unchanged=True,
              analysis_script_sha256=digest(Path(__file__)), wilson_intervals=SHOW_RAW_WILSON_INTERVALS,
              limitations=["Previously inspected scenarios and selected layers: exploratory results.",
                           "Explicit choice changes are not necessarily changes in valid decisions.",
                           "Simulated model decisions, not measured behavior of human travelers.",
                           "Wilson intervals describe across-scenario uncertainty under a binomial model, not selected-layer-confirmatory inference."]))
    report = ["出行偏好选择干预：图注与统计口径", "",
              "本节展示模型模拟旅客的出行选择，不是对真实旅客行为的实证检验。",
              "沿用第七步已选层：comfort18、environment16、safety14、convenience14、flexibility18。未按新的图形效果重新选层。",
              "主分析采用全部题目的明确选择raw，并并列报告固定总题数下的有效高偏好、有效低偏好和无效回答。",
              "layer_choices_paired不是一般意义上的同题配对；它还要求该题在所有alpha下均通过质检，是干预后筛选的子集。",
              "两者不能按哪条更好看选择。raw描述全部题目的明确选择，paired是始终合格子集上的敏感性分析。",
              "这五个选定层上所有题均识别到明确选择，所以raw每组20题，与本脚本不按质检筛选的同题raw严格一致。",
              "原始选择率仍可包含理由不一致或不符合约束的回答，不能单凭它宣称有效决策已获得同等改善。", "",
              "图01：同一批回答的评分与原始选择率。所有维度/alpha每格20题，不因质检失败删题。",
              "图02：原始高偏好选择率随alpha变化；浅色区为描述性Wilson95%区间，不校正选层、不代表生成或裁判重复误差。",
              "图03：全部20题固定分母三分类。绿色为有效高偏好，蓝色为有效低偏好，灰色为质量失败或未识别；三者之和100%。",
              "图04：raw与质检共同配对的对照。paired样本数在各维度内固定，但由干预后的质检结果决定。",
              f"图05：同一题从alpha={BASELINE_ALPHA}到alpha={COMPARISON_ALPHA}的选择转移计数；所有明确选择和双端均有效的转移另存CSV。", ""]
    for r in summaries:
        report.append(f"{r['trait']} L{r['layer']}：raw N={r['raw_n']}，高偏好率={np.round(r['raw_curve_percent'],2).tolist()}%；paired N={r['paired_n']}，比例={np.round(r['paired_curve_percent'],2).tolist()}%；有效高端/全部题={np.round(r['valid_high_all_curve_percent'],2).tolist()}%。")
    report += ["", "已有20题同时用于选层与作图，属于探索性证据；独立新题确认前不写成已经验证的普遍单调规律。",
               "特别是flexibility在较高alpha下质量失败增加。原始选择率上升不能掩盖有效高偏好选择占全部题比例的回落。",
               "评分与选择变化的幅度不直接跨维比较干预效率，因为各维提取方法、注入位置和Rubric不同。",
               "脚本参数均在开头；只使用已保存数据，不调用模型/API，不修改原始结果。"]
    (out / "图注与统计口径.txt").write_text("\n".join(report), encoding="utf-8")
    print("Saved figures and data: " + str(out))
    for r in summaries:
        print(f"{r['trait']}: layer={r['layer']}, raw N={r['raw_n']}, paired N={r['paired_n']}, raw={r['raw_curve_percent']}")


if __name__ == "__main__":
    main()
