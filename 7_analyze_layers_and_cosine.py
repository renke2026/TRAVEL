"""读取已有五维扫描：比较有效层，并画同层向量余弦图。无需加载大模型或调用API。"""

# Portable paths for the public research release.
import os as _TRAVEL_OS
import sys as _TRAVEL_SYS
from pathlib import Path as _TRAVEL_PATH
from travel_config import MODEL_PATH as _TRAVEL_MODEL_PATH, LLAMA_MODEL_PATH as _TRAVEL_LLAMA_PATH, QWEN3B_MODEL_PATH as _TRAVEL_QWEN3B_PATH


# ==================== 本次分析参数 ====================
LAYERS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
COEFFICIENTS = [-1, 0, 1, 2, 3]
QUESTION_COUNT = 20
RUNS = [
    # 维度、原结果目录、使用的向量、原扫描子目录。
    ("comfort", "20261018_141422_comfort", "prompt_last", "layer_scan_prompt_last_batch64"),
    ("environment", "20261018_161705_environment", "prompt_last", "layer_scan_prompt_last_prompt_batch32"),
    ("safety", "20261018_183835_safety", "response_mean", "layer_scan_response_mean_all_batch32"),
    ("convenience", "20261018_212758_convenience", "response_mean", "layer_scan_response_mean_system_batch32"),
    ("flexibility", "20261018_235055_flexibility", "response_mean", "layer_scan_response_mean_all_batch32"),
]
SCORE_WEIGHT = 0.5          # 两个目标等权；都先除以该维度最大的正向斜率。
CHOICE_WEIGHT = 0.5
FIGURE_LAYER = None        # None：选综合控制效果最好的共同层；填整数可另画指定层。
QUALITY_WARNING_RATE = 0.8 # 仅提示质量问题，不靠删除回答提高分数或排名。
BOOTSTRAP_REPEATS = 3000   # 按题配对重采样；不是重新生成回答或重新评分。
RANDOM_SEED = 20261019
FIGURE_DPI = 300
OUTPUT_DIR = "runs/五维层数与余弦分析_20261019"
# ====================================================

import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
TRAITS = [item[0] for item in RUNS]
ALPHAS = np.array(COEFFICIENTS, dtype=float)
X = ALPHAS - ALPHAS.mean()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slope(values):
    return float(X @ values / (X @ X))


def paired_interval(differences, rng):
    samples = rng.choice(differences, size=(BOOTSTRAP_REPEATS, len(differences)), replace=True).mean(axis=1)
    return [float(v) for v in np.quantile(samples, [.025, .975])]


def cosine(vectors, layer):
    # 只为余弦计算除以长度；不修改原.pt，不改变实际干预幅度。
    values = torch.stack([v[layer].double() for v in vectors])
    norms = values.norm(dim=1, keepdim=True)
    if not torch.isfinite(values).all() or (norms <= 0).any():
        raise ValueError(f"第{layer}层存在无效或零向量。")
    unit = values / norms
    return (unit @ unit.T).numpy()


def heatmap(ax, matrix, title, colorbar=True):
    # 与参考图类似，只画下三角；所有图统一-1到1的色标。
    mask = np.triu(np.ones_like(matrix, dtype=bool), 1)
    shown = np.ma.array(matrix, mask=mask)
    picture = ax.imshow(shown, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(5):
        for j in range(i + 1):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                    fontsize=10, color="white" if abs(value) > .62 else "#202020")
    ax.set_xticks(range(5), TRAITS, rotation=40, ha="right")
    ax.set_yticks(range(5), TRAITS)
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if colorbar:
        ax.figure.colorbar(picture, ax=ax, shrink=.75, label="Cosine similarity")
    return picture


def main():
    if 0 not in COEFFICIENTS or len(set(LAYERS)) != len(LAYERS):
        raise ValueError("系数应包含0；层数不可重复。")
    if not np.isclose(SCORE_WEIGHT + CHOICE_WEIGHT, 1):
        raise ValueError("两个目标的权重之和必须为1。")
    out = ROOT / OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    metrics, curves, provenance = [], [], []
    used_vectors, prompt_vectors, originals = [], [], {}

    for trait, folder, method, scan_folder in RUNS:
        run = ROOT / "runs" / folder
        scan = run / scan_folder
        score_path = scan / "scores.json"
        rows, settings = read(score_path), read(scan / "settings.json")
        lookup = {(r["physical_layer"], float(r["coeff"]), r["question_id"]): r for r in rows}
        expected = {(l, float(a), q) for l in LAYERS for a in COEFFICIENTS for q in range(QUESTION_COUNT)}
        if len(lookup) != len(rows) or set(lookup) != expected:
            raise ValueError(f"{trait}扫描有重复、缺项或额外记录，需先核对。")
        if settings["vector_method"] != method:
            raise ValueError(f"{trait}方法与扫描配置不符。")
        suffix = "prompt_last" if method == "prompt_last" else "all_layers"
        vector_path = run / f"{trait}_vectors_{suffix}.pt"
        prompt_path = run / f"{trait}_vectors_prompt_last.pt"
        if digest(vector_path) != settings["vector_sha256"]:
            raise ValueError(f"{trait}向量与扫描时的文件不同。")
        metadata = read(vector_path.with_suffix(".metadata.json"))
        prompt_metadata = read(prompt_path.with_suffix(".metadata.json"))
        used_vectors.append(torch.load(vector_path, map_location="cpu", weights_only=True))
        prompt_vectors.append(torch.load(prompt_path, map_location="cpu", weights_only=True))
        if used_vectors[-1].ndim != 2 or max(LAYERS) >= len(used_vectors[-1]):
            raise ValueError(f"{trait}逐层向量尺寸错误。")
        paths = [score_path, scan / "settings.json", scan / "trait_data_eval.json",
                 vector_path, vector_path.with_suffix(".metadata.json"),
                 prompt_path, prompt_path.with_suffix(".metadata.json")]
        for path in paths:
            originals[str(path)] = digest(path)
        provenance.append({"trait": trait, "run": str(run), "scan": str(scan),
                           "vector": str(vector_path), "vector_method": method,
                           "intervention_scope": settings["intervention_scope"],
                           "metadata": metadata, "prompt_last_metadata": prompt_metadata})

        for layer in LAYERS:
            cell = [[lookup[layer, float(a), q] for q in range(QUESTION_COUNT)] for a in COEFFICIENTS]
            # 评分固定同题配对；质检失败仍保留原始分，另算质检配对敏感性结果。
            score_ids = [q for q in range(QUESTION_COUNT)
                         if all(c[q]["score"] is not None and np.isfinite(c[q]["score"]) for c in cell)]
            choice_ids = [q for q in range(QUESTION_COUNT)
                          if all(c[q].get("selected_option") in ("A", "B") for c in cell)]
            quality_ids = [q for q in score_ids if all(c[q]["quality_ok"] for c in cell)]
            if not score_ids or not choice_ids:
                raise ValueError(f"{trait}第{layer}层无共同可用题，不能排名。")
            scores = np.array([[c[q]["score"] for q in score_ids] for c in cell])
            # 不使用positive_choice字段：该字段已将质检失败置空，会改变各组分母。
            choices = np.array([[c[q]["selected_option"] == c[q]["positive_option"]
                                 for q in choice_ids] for c in cell], dtype=float)
            means, rates = scores.mean(axis=1), choices.mean(axis=1) * 100
            quality_rates = np.array([np.mean([r["quality_ok"] for r in c]) * 100 for c in cell])
            quality_scores = np.array([[c[q]["score"] for q in quality_ids] for c in cell]) if quality_ids else None
            qmeans = quality_scores.mean(axis=1) if quality_ids else None
            score_ci = paired_interval(scores[-1] - scores[0], rng)
            choice_ci = paired_interval((choices[-1] - choices[0]) * 100, rng)
            baseline = COEFFICIENTS.index(0)
            row = dict(trait=trait, layer=layer, method=method, scope=settings["intervention_scope"],
                       score_n=len(score_ids), choice_n=len(choice_ids), score_slope=slope(means),
                       choice_slope_pp=slope(rates), score_delta=means[-1] - means[0],
                       score_delta_ci_low=score_ci[0], score_delta_ci_high=score_ci[1],
                       choice_delta_pp=rates[-1] - rates[0], choice_delta_ci_low=choice_ci[0],
                       choice_delta_ci_high=choice_ci[1], score_gain_from_zero=means[-1] - means[baseline],
                       choice_gain_from_zero_pp=rates[-1] - rates[baseline],
                       score_monotone=bool(np.all(np.diff(means) >= -1e-10)),
                       choice_monotone=bool(np.all(np.diff(rates) >= -1e-10)),
                       quality_min_percent=float(quality_rates.min()), quality_mean_percent=float(quality_rates.mean()),
                       quality_common_n=len(quality_ids), quality_score_slope=slope(qmeans) if quality_ids else None,
                       quality_score_delta=float(qmeans[-1] - qmeans[0]) if quality_ids else None,
                       quality_warning=bool(quality_rates.min() < 100 * QUALITY_WARNING_RATE))
            metrics.append(row)
            for i, alpha in enumerate(COEFFICIENTS):
                explicit = [r for r in cell[i] if r.get("selected_option") in ("A", "B")]
                curves.append(dict(trait=trait, layer=layer, alpha=alpha,
                                   score=float(means[i]), score_n=len(score_ids),
                                   choice_percent=float(rates[i]), choice_n=len(choice_ids),
                                   raw_available_choice_percent=float(np.mean([r["selected_option"] == r["positive_option"] for r in explicit]) * 100),
                                   raw_available_choice_n=len(explicit), quality_percent=float(quality_rates[i]),
                                   quality_paired_score=float(qmeans[i]) if quality_ids else None,
                                   quality_paired_n=len(quality_ids)))

        group = [m for m in metrics if m["trait"] == trait]
        score_max = max(m["score_slope"] for m in group)
        choice_max = max(m["choice_slope_pp"] for m in group)
        if score_max <= 0 or choice_max <= 0:
            raise ValueError(f"{trait}没有双目标正向响应，不能宣称有有效层。")
        for m in group:
            m["relative_control"] = (SCORE_WEIGHT * m["score_slope"] / score_max
                                     + CHOICE_WEIGHT * m["choice_slope_pp"] / choice_max)

    # 先按已有行为效果选共同层；余弦数值不参与选层。
    common = []
    for layer in LAYERS:
        group = [m for m in metrics if m["layer"] == layer]
        common.append(dict(layer=layer, relative_control=float(np.mean([m["relative_control"] for m in group])),
                           positive_both_traits=sum(m["score_slope"] > 0 and m["choice_slope_pp"] > 0 for m in group),
                           mean_quality_percent=float(np.mean([m["quality_mean_percent"] for m in group]))))
    common.sort(key=lambda m: (-m["relative_control"], m["layer"]))
    auto_layer = common[0]["layer"]
    figure_layer = auto_layer if FIGURE_LAYER is None else FIGURE_LAYER
    if figure_layer not in LAYERS:
        raise ValueError("FIGURE_LAYER应在LAYERS中。")
    winners = []
    for trait in TRAITS:
        group = sorted([m for m in metrics if m["trait"] == trait],
                       key=lambda m: (-m["relative_control"], m["layer"]))
        best = dict(group[0])
        score_best = max(group, key=lambda m: m["score_slope"])
        monotone = [m for m in group if m["score_monotone"] and m["choice_monotone"]
                    and m["score_slope"] > 0 and m["choice_slope_pp"] > 0]
        best.update(score_only_best_layer=score_best["layer"],
                    monotone_alternative_layer=monotone[0]["layer"] if monotone else None,
                    score_curve=[r["score"] for r in curves if r["trait"] == trait and r["layer"] == best["layer"]],
                    choice_curve_percent=[r["choice_percent"] for r in curves if r["trait"] == trait and r["layer"] == best["layer"]])
        winners.append(best)

    pair_rows, geometry = [], []
    matrices = {}
    for family, vectors in (("actual_selected_methods", used_vectors), ("all_prompt_last", prompt_vectors)):
        matrices[family] = {}
        for layer in LAYERS:
            matrix = cosine(vectors, layer)
            matrices[family][layer] = matrix
            pairs = matrix[np.triu_indices(5, 1)]
            geometry.append(dict(family=family, layer=layer, mean_abs_cosine=float(np.abs(pairs).mean()),
                                 max_abs_cosine=float(np.abs(pairs).max()), mean_cosine=float(pairs.mean())))
            for i in range(5):
                for j in range(i + 1, 5):
                    pair_rows.append(dict(family=family, layer=layer, trait_i=TRAITS[i], trait_j=TRAITS[j],
                                          cosine=float(matrix[i, j])))
            save_csv(out / f"cosine_{family}_layer_{layer}.csv",
                     [dict(trait=t, **{name: float(matrix[i, j]) for j, name in enumerate(TRAITS)})
                      for i, t in enumerate(TRAITS)])

    for name, rows in (("layer_metrics", metrics), ("curve_values", curves), ("best_layers", winners),
                       ("common_layer_ranking", common), ("cosine_pairs_all_layers", pair_rows),
                       ("cosine_layer_statistics", geometry)):
        save_json(out / (name + ".json"), rows)
        save_csv(out / (name + ".csv"), rows)
    human = []
    for best in winners:
        human.append({"维度": best["trait"], "综合响应最强层": best["layer"],
                      "评分增加(3减负1)": round(best["score_delta"], 4),
                      "选择率增加(百分点)": round(best["choice_delta_pp"], 4),
                      "评分每单位alpha斜率": round(best["score_slope"], 4),
                      "选择率每单位alpha斜率(百分点)": round(best["choice_slope_pp"], 4),
                      "评分共同题数": best["score_n"], "选择率共同题数": best["choice_n"],
                      "评分曲线单调": best["score_monotone"], "选择曲线单调": best["choice_monotone"],
                      "平均质检通过率(%)": best["quality_mean_percent"],
                      "最低质检通过率(%)": best["quality_min_percent"],
                      "五系数共同质检合格题数": best["quality_common_n"],
                      "双曲线单调的备选层": best["monotone_alternative_layer"]})
    save_csv(out / "最有效层汇总.csv", human)

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(8.4, 7), layout="constrained")
    heatmap(ax, matrices["actual_selected_methods"][figure_layer],
            f"Cosine similarity between travel preference vectors\nPhysical layer {figure_layer}")
    fig.supxlabel("Current selected extraction methods; same physical layer.\n"
                  "Layer selected by behavioral control, not by cosine values.", fontsize=9)
    for ext in ("png", "svg"):
        fig.savefig(out / f"Figure2_cosine_layer_{figure_layer}.{ext}", dpi=FIGURE_DPI)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.4, 7), layout="constrained")
    heatmap(ax, matrices["all_prompt_last"][figure_layer],
            f"Same extraction-position comparison: prompt_last\nPhysical layer {figure_layer}")
    fig.supxlabel("Extraction position matched across all five traits.\n"
                  "This is a geometric comparison, not a steering-performance result.", fontsize=9)
    for ext in ("png", "svg"):
        fig.savefig(out / f"Figure2_prompt_last_control_layer_{figure_layer}.{ext}", dpi=FIGURE_DPI)
    plt.close(fig)

    # 附图保留所有层，防止只看到某一层的热图。
    fig, axes = plt.subplots(4, 4, figsize=(20, 20), layout="constrained")
    for ax, layer in zip(axes.flat, LAYERS):
        heatmap(ax, matrices["actual_selected_methods"][layer], f"Layer {layer}", colorbar=False)
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(LAYERS):]:
        ax.axis("off")
    fig.suptitle("Same-layer cosine similarity | all 14 scanned layers", fontsize=20)
    fig.savefig(out / "cosine_all_14_layers.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(19, 8), layout="constrained")
    for i, best in enumerate(winners):
        lines = [r for r in curves if r["trait"] == best["trait"] and r["layer"] == best["layer"]]
        for row_index, field, label in ((0, "score", "Trait expression (0-100)"),
                                        (1, "choice_percent", "High-preference choice (%)")):
            ax = axes[row_index, i]
            ax.plot(COEFFICIENTS, [r[field] for r in lines], "o-", color="#277da1", linewidth=2)
            ax.set(title=f"{best['trait']} | layer {best['layer']}", xlabel="Steering coefficient",
                   ylabel=label if i == 0 else "", ylim=(-3, 103), xticks=COEFFICIENTS)
            ax.grid(axis="y", alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Layers selected from existing results | equal weight for score and choice trends", fontsize=16)
    fig.supxlabel("All numeric answers for scores; same explicitly identified questions across coefficients for choices.\n"
                  "Quality failures retained and reported separately. Exploratory selection on 20 questions per trait.", fontsize=10)
    fig.savefig(out / "selected_layer_score_and_choice_curves.png", dpi=FIGURE_DPI)
    plt.close(fig)

    protocol = dict(layers=LAYERS, coefficients=COEFFICIENTS, score_weight=SCORE_WEIGHT,
                    choice_weight=CHOICE_WEIGHT, automatic_common_layer=auto_layer, displayed_layer=figure_layer,
                    selection="For each trait, equal-weight OLS slopes normalized by that trait's positive maximum; shared layer maximizes the five-trait mean.",
                    figure_selection_uses_cosine=False, quality_warning_rate=QUALITY_WARNING_RATE,
                    bootstrap_repeats=BOOTSTRAP_REPEATS, random_seed=RANDOM_SEED,
                    source_files_sha256=originals, sources=provenance,
                    limitations=["Exploratory selection using already inspected questions, not independent confirmation.",
                                 "Current selected vectors use different extraction methods and injection positions.",
                                 "Quality failures remain in primary scores; complete quality-passed pair analyses may have small N.",
                                 "Question bootstrap intervals are descriptive, do not correct layer selection, and omit generation/judge replication uncertainty.",
                                 "prompt_last comparison matches extraction position, not prompt wording, sample selection, or steering performance.",
                                 "Near-zero cosine does not prove behavioral or psychological independence."])
    if any(digest(Path(path)) != checksum for path, checksum in originals.items()):
        raise ValueError("分析期间来源文件发生变化，请核查。")
    protocol["source_files_unchanged"] = True
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    protocol["analysis_script_sha256"] = digest(Path(__file__))
    save_json(out / "analysis_protocol_and_sources.json", protocol)

    report = ["五维最有效层与同层余弦分析", "", f"系数：{COEFFICIENTS}；候选层：{LAYERS}",
              "只读取已有回答、评分和向量；不调用模型/API，不改动原实验文件。", "",
              "选层依据：每维分别计算五个alpha均值对alpha的OLS斜率，评分和选择率斜率各除以该维所有层的最大正向斜率，等权平均。",
              "共同展示层选择该指标在五维上的平均值最大者；不按余弦接近零程度挑层。",
              "这是围绕两个研究目标的透明操作性定义，不是唯一学术标准；所有层、原始指标和质检结果一并保留。",
              "评分在同题同系数配对数据上计算。选择率按各层所有系数均识别明确选择的共同题计算，不先按质检筛选。",
              "不得直接把scores.json中的positive_choice当作原始选择：该字段已把质检失败置空。", ""]
    for best in winners:
        report.extend([f"{best['trait']}：双目标候选层={best['layer']}；只按评分斜率选={best['score_only_best_layer']}；双曲线单调候选={best['monotone_alternative_layer']}",
                       f"  分数曲线={np.round(best['score_curve'], 3).tolist()}",
                       f"  高关注选择率(%)={np.round(best['choice_curve_percent'], 2).tolist()}；共同题数={best['choice_n']}",
                       f"  端点分差={best['score_delta']:.3f}；端点选择差={best['choice_delta_pp']:.2f}个百分点；分差按题bootstrap 95%区间=[{best['score_delta_ci_low']:.2f}, {best['score_delta_ci_high']:.2f}]",
                       f"  每格最低质检通过率={best['quality_min_percent']:.0f}%；平均={best['quality_mean_percent']:.0f}%；五系数均通过质检的题数={best['quality_common_n']}。", ""])
    report.extend([f"共同展示层={figure_layer}；按行为指标自动选层={auto_layer}。",
                   "主图使用当前实际选用的五维向量；提取方法混合，因此不把差异全部归因于维度本身。",
                   "另附所有维度prompt_last的同位置对照；不把该对照说成所有维度的最佳干预方法。",
                   "余弦热图统一色标[-1,1]，仅画下三角，10个不重复的跨维度关系；没有强制正交化。",
                   "共同层第16层不一定最接近正交。几何结果如有明显重叠，应保留，不把重叠解释为已经证明行为混淆。",
                   "当前最优属于已有20题的探索性候选；独立新题验证后才能主张稳定优于其他层。",
                   "质检失败可能影响原始评分和选择解释。质量配对结果单独报告，不能只展示筛选后更高的曲线。",
                   "按题bootstrap不构成对14层挑选后的显著性证明。",
                   "以后若研究多向量组合，不同层联合干预应另做交互验证。"])
    (out / "分析结论与图注.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print("\n输出：" + str(out))


if __name__ == "__main__":
    main()
