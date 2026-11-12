"""交叉干预结果：同题差分、评分/选择/质量矩阵和描述性不确定性。"""

# Portable paths for the public research release.
import os as _TRAVEL_OS
import sys as _TRAVEL_SYS
from pathlib import Path as _TRAVEL_PATH
from travel_config import MODEL_PATH as _TRAVEL_MODEL_PATH, LLAMA_MODEL_PATH as _TRAVEL_LLAMA_PATH, QWEN3B_MODEL_PATH as _TRAVEL_QWEN3B_PATH


# ==================== 本章分析参数 ====================
RUN_DIR = "runs/cross_trait_original_20261019"
COEFFICIENTS = [-1, 0, 1, 2, 3]
DISPLAY_COEFFICIENTS = [-1, 1, 2, 3]
BOOTSTRAP_REPEATS = 3000
RANDOM_SEED = 20261019
SMALL_EFFECT_MARGIN = 5.0     # 仅描述性标记±5分，不把“不显著”称为独立。
FIGURE_DPI = 300
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
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
TRAITS = ["comfort", "environment", "safety", "convenience", "flexibility"]
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def valid_quality(row):
    return bool(row["quality_ok"] and not row.get("whole_answer_truncated",False))


def quality_status(row):
    if row.get("whole_answer_truncated",False):return "truncated"
    if row["quality_ok"]:return "pass"
    summaries=[d for d in row["quality_diagnostics"] if d.get("reason")=="quality_summary"]
    if summaries:return "fail" if summaries[-1].get("status")=="fail" else "uncertain"
    if any(d.get("reason")=="missing_explicit_choice" for d in row["quality_diagnostics"]):return "fail"
    return "uncertain"


def estimate(values, rng):
    if not len(values):
        return None,None,None
    values=np.asarray(values,dtype=float)
    samples=values[rng.integers(0,len(values),size=(BOOTSTRAP_REPEATS,len(values)))].mean(axis=1)
    return float(values.mean()),float(np.quantile(samples,.025)),float(np.quantile(samples,.975))


def save_figure(fig,out,name):
    for ext in ("png","svg"):
        fig.savefig(out/f"{name}.{ext}",dpi=FIGURE_DPI)
    plt.close(fig)


def main():
    run=ROOT/RUN_DIR
    out=run/"analysis"
    out.mkdir(exist_ok=True)
    protocol=read(run/"protocol.json")
    scores=read(run/"module_scores.json")
    answer_rows=read(run/"answers.json") if (run/"answers.json").exists() else []
    answer_lookup={(r["condition"],r["case_id"],r["probe_trait"]):r for r in answer_rows}
    if any(r["prompt_control"] for r in scores):
        scores=[r for r in scores if not r["prompt_control"]]
    lookup={(r["condition"],r["case_id"],r["scored_trait"]):r for r in scores}
    if len(lookup)!=len(scores):
        raise ValueError("同一条件、情境和被测维度存在重复。")
    case_ids=sorted({q["case_id"] for q in protocol["questions"]})
    expected_conditions=["baseline"]+[f"{t}_alpha_{a}" for t in TRAITS for a in COEFFICIENTS if a!=0]
    expected={(condition,q,t) for condition in expected_conditions for q in case_ids for t in TRAITS}
    if set(lookup)!=expected:
        raise ValueError(f"评分记录未完整，缺{len(expected-set(lookup))}项；不能把缺项填0。")
    rng=np.random.default_rng(RANDOM_SEED)
    effects=[]
    for injected in TRAITS:
        for scored in TRAITS:
            conditions=["baseline" if a==0 else f"{injected}_alpha_{a}" for a in COEFFICIENTS]
            numeric_cases=[q for q in case_ids if all(lookup[c,q,scored]["score"] is not None for c in conditions)]
            choice_cases=[q for q in case_ids if all(lookup[c,q,scored]["high_choice"] is not None for c in conditions)]
            quality_cases=[q for q in numeric_cases if all(valid_quality(lookup[c,q,scored]) for c in conditions)]
            for alpha,condition in zip(COEFFICIENTS,conditions):
                differences=[lookup[condition,q,scored]["score"]-lookup["baseline",q,scored]["score"] for q in numeric_cases]
                delta,lo,hi=estimate(differences,rng)
                choice_delta,clo,chi=estimate([100*(int(lookup[condition,q,scored]["high_choice"])-int(lookup["baseline",q,scored]["high_choice"])) for q in choice_cases],rng)
                quality_delta,qlo,qhi=estimate([100*(int(valid_quality(lookup[condition,q,scored]))-int(valid_quality(lookup["baseline",q,scored]))) for q in case_ids],rng)
                quality_score,qslo,qshi=estimate([lookup[condition,q,scored]["score"]-lookup["baseline",q,scored]["score"] for q in quality_cases],rng)
                comparison_quality_cases=[q for q in case_ids if all(lookup[c,q,scored]["score"] is not None and valid_quality(lookup[c,q,scored]) for c in ("baseline",condition))]
                pair_quality,pqlo,pqhi=estimate([lookup[condition,q,scored]["score"]-lookup["baseline",q,scored]["score"] for q in comparison_quality_cases],rng)
                active=[lookup[condition,q,scored] for q in case_ids]
                baselines=[lookup["baseline",q,scored] for q in case_ids]
                active_tokens=float(np.mean([answer_lookup[condition,q,scored]["response_tokens"] for q in case_ids])) if answer_lookup else None
                base_tokens=float(np.mean([answer_lookup["baseline",q,scored]["response_tokens"] for q in case_ids])) if answer_lookup else None
                effects.append(dict(injected_trait=injected,scored_trait=scored,alpha=alpha,
                    score_delta=delta,score_ci_low=lo,score_ci_high=hi,numeric_paired_n=len(numeric_cases),
                    active_paired_score_mean=float(np.mean([lookup[condition,q,scored]["score"] for q in numeric_cases])) if numeric_cases else None,
                    baseline_paired_score_mean=float(np.mean([lookup["baseline",q,scored]["score"] for q in numeric_cases])) if numeric_cases else None,
                    choice_delta_pp=choice_delta,choice_ci_low=clo,choice_ci_high=chi,choice_paired_n=len(choice_cases),
                    quality_delta_pp=quality_delta,quality_ci_low=qlo,quality_ci_high=qhi,total_n=len(case_ids),
                    active_quality_percent=100*sum(valid_quality(r) for r in active)/len(active),
                    baseline_quality_percent=100*sum(valid_quality(r) for r in baselines)/len(baselines),
                    active_quality_fail_percent=100*sum(quality_status(r)=="fail" for r in active)/len(active),
                    active_quality_uncertain_percent=100*sum(quality_status(r)=="uncertain" for r in active)/len(active),
                    active_truncated_percent=100*sum(quality_status(r)=="truncated" for r in active)/len(active),
                    active_response_tokens_mean=active_tokens,baseline_response_tokens_mean=base_tokens,
                    response_length_delta_tokens=active_tokens-base_tokens if active_tokens is not None else None,
                    active_choice_percent=100*sum(bool(r["high_choice"]) for r in active)/sum(r["high_choice"] is not None for r in active) if any(r["high_choice"] is not None for r in active) else None,
                    baseline_choice_percent=100*sum(bool(r["high_choice"]) for r in baselines)/sum(r["high_choice"] is not None for r in baselines) if any(r["high_choice"] is not None for r in baselines) else None,
                    active_explicit_choice_n=sum(r["high_choice"] is not None for r in active),baseline_explicit_choice_n=sum(r["high_choice"] is not None for r in baselines),
                    active_valid_high_all_percent=100*sum(valid_quality(r) and bool(r["high_choice"]) for r in active)/len(active),
                    quality_paired_score_delta=quality_score,quality_paired_ci_low=qslo,quality_paired_ci_high=qshi,quality_paired_n=len(quality_cases),
                    pairwise_quality_score_delta=pair_quality,pairwise_quality_ci_low=pqlo,pairwise_quality_ci_high=pqhi,pairwise_quality_n=len(comparison_quality_cases),
                    ci_inside_small_effect_margin=(lo>=-SMALL_EFFECT_MARGIN and hi<=SMALL_EFFECT_MARGIN) if lo is not None else None))
    save_json(out/"cross_trait_effects.json",effects)
    save_csv(out/"cross_trait_effects.csv",effects)
    table={(r["injected_trait"],r["scored_trait"],r["alpha"]):r for r in effects}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"svg.fonttype":"none"})
    for field,label,stem in (("score_delta","Trait expression change (points)","01_score_effect_matrices"),
                             ("choice_delta_pp","High-preference choice change (pp)","02_choice_effect_matrices"),
                             ("quality_delta_pp","Quality-pass rate change (pp)","03_quality_effect_matrices")):
        fig,axes=plt.subplots(2,2,figsize=(13,11),layout="constrained")
        for ax,alpha in zip(axes.flat,DISPLAY_COEFFICIENTS):
            values=np.array([[table[i,j,alpha][field] if table[i,j,alpha][field] is not None else np.nan for j in TRAITS] for i in TRAITS])
            picture=ax.imshow(values,cmap="RdBu_r",vmin=-100,vmax=100)
            for i in range(5):
                for j in range(5):
                    v=values[i,j]
                    label="NA" if np.isnan(v) else f"{v:+.1f}"
                    if field=="choice_delta_pp":label+=f"\n(n={table[TRAITS[i],TRAITS[j],alpha]['choice_paired_n']})"
                    ax.text(j,i,label,ha="center",va="center",
                            fontsize=10 if field=="choice_delta_pp" else 12,color="white" if abs(v)>62 else "#222222")
                ax.add_patch(Rectangle((i-.5,i-.5),1,1,fill=False,edgecolor="#202020",lw=1.6))
            ax.set_xticks(range(5),TRAITS,rotation=35,ha="right")
            ax.set_yticks(range(5),TRAITS)
            ax.set(xlabel="Evaluated dimension",ylabel="Injected dimension",title=f"alpha = {alpha:g} vs alpha = 0")
            fig.colorbar(picture,ax=ax,shrink=.8,label=label)
        caption=("Complete explicit-choice pairs across all strengths; n is shown per cell. No quality-based exclusions."
                 if field=="choice_delta_pp" else "Missing scores remain NA; quality failures are retained in primary trait-expression estimates.")
        fig.supxlabel("Same questions paired within each evaluated dimension. Diagonal boxes mark the intended target.\n"+caption,fontsize=10)
        save_figure(fig,out,stem)

    fig,axes=plt.subplots(2,3,figsize=(13,8),layout="constrained")
    for ax,injected in zip(axes.flat,TRAITS):
        for scored,color in zip(TRAITS,COLORS):
            rows=[table[injected,scored,a] for a in COEFFICIENTS]
            ys=[r["score_delta"] for r in rows]
            ax.plot(COEFFICIENTS,ys,"o-" if injected==scored else "o--",color=color,
                    lw=2.5 if injected==scored else 1.3,ms=5 if injected==scored else 3)
            if injected==scored:
                ax.fill_between(COEFFICIENTS,[r["score_ci_low"] for r in rows],
                                [r["score_ci_high"] for r in rows],color=color,alpha=.1)
        ax.axhspan(-SMALL_EFFECT_MARGIN,SMALL_EFFECT_MARGIN,color="#888888",alpha=.08,zorder=0)
        ax.axhline(0,color="#777777",lw=.8)
        ax.set(title=f"Inject {injected}",xlabel="Steering coefficient",ylabel="Score change from baseline",xticks=COEFFICIENTS,ylim=(-100,100))
        ax.grid(axis="y",alpha=.15);ax.spines[["top","right"]].set_visible(False)
    ax=axes.flat[-1];ax.axis("off")
    ax.legend(handles=[Line2D([0],[0],color=c,marker="o",label=t) for t,c in zip(TRAITS,COLORS)],
              title="Evaluated dimension",loc="upper left",frameon=False)
    ax.text(.02,.35,"Solid: intended dimension.\nDashed: other dimensions.\n\n"
            "Band around target: descriptive paired\nbootstrap 95% interval across scenarios.\n\n"
            "Gray region: +/-5 score points;\nnot a proof of independence.",transform=ax.transAxes,va="top",fontsize=10,linespacing=1.5)
    save_figure(fig,out,"04_cross_trait_dose_trajectories")

    # 论文主图：alpha=3的25格均值变化及其配对区间，不只展示对角线。
    alpha=3
    fig,ax=plt.subplots(figsize=(11,9),layout="constrained")
    values=np.array([[table[i,j,alpha]["score_delta"] if table[i,j,alpha]["score_delta"] is not None else np.nan for j in TRAITS] for i in TRAITS])
    picture=ax.imshow(values,cmap="RdBu_r",vmin=-100,vmax=100)
    for i,injected in enumerate(TRAITS):
        for j,scored in enumerate(TRAITS):
            r=table[injected,scored,alpha]
            if r["score_delta"] is None:label="NA"
            else:label=f"{r['score_delta']:+.1f}\n[{r['score_ci_low']:+.1f}, {r['score_ci_high']:+.1f}]\nn = {r['numeric_paired_n']}"
            ax.text(j,i,label,ha="center",va="center",fontsize=10,color="white" if abs(values[i,j])>62 else "#222222",linespacing=1.6)
        ax.add_patch(Rectangle((i-.5,i-.5),1,1,fill=False,edgecolor="#222222",lw=2))
    layers={r["trait"]:r["layer"] for r in protocol["vectors"]}
    ax.set_xticks(range(5),TRAITS,rotation=25,ha="right")
    ax.set_yticks(range(5),[f"{t} (L{layers[t]})" for t in TRAITS])
    ax.set(xlabel="Evaluated dimension (its original 20-question bank)",ylabel="Injected dimension",
           title="Cross-dimension effects: alpha = 3 versus no intervention")
    fig.colorbar(picture,ax=ax,shrink=.85,label="Trait expression change (points)")
    fig.supxlabel("Mean paired score changes; brackets: descriptive bootstrap 95% intervals.\n"
                  "Quality failures are retained here and reported separately. Selected layers and original banks: exploratory analysis.",fontsize=10)
    save_figure(fig,out,"05_main_matrix_alpha3_with_intervals")

    fig,axes=plt.subplots(2,3,figsize=(14,8),layout="constrained")
    statuses=[("pass","Passed","#009E73"),("fail","Failed","#D55E00"),
              ("uncertain","Uncertain","#E69F00"),("truncated","Truncated","#777777")]
    for ax,condition in zip(axes.flat,["baseline"]+[f"{t}_alpha_3" for t in TRAITS]):
        left=np.zeros(5)
        for status,label,color in statuses:
            counts=np.array([sum(quality_status(lookup[condition,q,t])==status for q in case_ids)/len(case_ids)*100 for t in TRAITS])
            ax.barh(range(5),counts,left=left,color=color,label=label,height=.7)
            for j,value in enumerate(counts):
                if value>=10:ax.text(left[j]+value/2,j,f"{value:.0f}",ha="center",va="center",fontsize=9,color="white" if status in ("pass","fail","truncated") else "#222222")
            left+=counts
        ax.set_yticks(range(5),TRAITS);ax.invert_yaxis()
        ax.set(xlim=(0,100),xlabel="All original questions (%)",title="No intervention" if condition=="baseline" else condition.replace("_alpha_3"," / alpha = 3"))
        ax.spines[["top","right"]].set_visible(False)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=4,frameon=False)
    save_figure(fig,out,"06_quality_statuses_alpha3")

    tradeoffs=[]
    fig,axes=plt.subplots(2,5,figsize=(17,7),layout="constrained")
    for k,injected in enumerate(TRAITS):
        series=[]
        for alpha in COEFFICIENTS:
            rows=[table[injected,j,alpha] for j in TRAITS]
            own=table[injected,injected,alpha]
            off=[abs(r["score_delta"]) for r in rows if r["scored_trait"]!=injected and r["score_delta"] is not None]
            row=dict(injected_trait=injected,alpha=alpha,target_score_delta=own["score_delta"],
                mean_abs_off_target_delta=float(np.mean(off)),mean_quality_percent=float(np.mean([r["active_quality_percent"] for r in rows])),
                minimum_bank_quality_percent=min(r["active_quality_percent"] for r in rows))
            series.append(row);tradeoffs.append(row)
        axes[0,k].plot(COEFFICIENTS,[r["target_score_delta"] for r in series],"o-",color="#009E73",label="Target score change")
        axes[0,k].plot(COEFFICIENTS,[r["mean_abs_off_target_delta"] for r in series],"s--",color="#D55E00",label="Mean absolute off-target change")
        axes[0,k].axhline(0,color="#777777",lw=.7)
        axes[0,k].set(title=injected,ylim=(-40,80),ylabel="Score points" if k==0 else "")
        axes[1,k].plot(COEFFICIENTS,[r["mean_quality_percent"] for r in series],"o-",color="#0072B2",label="Overall quality-pass rate")
        axes[1,k].plot(COEFFICIENTS,[r["minimum_bank_quality_percent"] for r in series],"s--",color="#CC79A7",label="Lowest bank pass rate")
        axes[1,k].set(ylim=(0,100),xlabel="Steering coefficient",ylabel="Quality-pass rate (%)" if k==0 else "")
        for ax in axes[:,k]:
            ax.set_xticks(COEFFICIENTS);ax.grid(axis="y",alpha=.15);ax.spines[["top","right"]].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels();h2,l2=axes[1,0].get_legend_handles_labels()
    fig.legend(handles+h2,labels+l2,loc="outside upper center",ncol=4,frameon=False,fontsize=9)
    fig.supxlabel("All steering strengths retained. Off-target averages are descriptive summaries across different rubrics.\nOverall quality equally weights the five original banks; a high target score alone does not establish selective, valid control.",fontsize=9)
    save_figure(fig,out,"09_intensity_selectivity_quality_tradeoff")
    save_csv(out/"intensity_tradeoffs.csv",tradeoffs)

    selectivity=[]
    for injected in TRAITS:
        for alpha in DISPLAY_COEFFICIENTS:
            own=table[injected,injected,alpha]
            off=[table[injected,j,alpha] for j in TRAITS if j!=injected]
            nonmissing=[r for r in off if r["score_delta"] is not None]
            largest=max(nonmissing,key=lambda r:abs(r["score_delta"])) if nonmissing else None
            selectivity.append(dict(injected_trait=injected,alpha=alpha,target_delta=own["score_delta"],
                target_ci_low=own["score_ci_low"],target_ci_high=own["score_ci_high"],
                target_quality_percent=own["active_quality_percent"],
                mean_abs_off_target_delta=float(np.mean([abs(r["score_delta"]) for r in nonmissing])) if nonmissing else None,
                largest_off_target_dimension=largest["scored_trait"] if largest else None,
                largest_off_target_delta=largest["score_delta"] if largest else None,
                off_target_intervals_inside_5_points=sum(bool(r["ci_inside_small_effect_margin"]) for r in off),
                off_target_dimensions=4))
    save_json(out/"target_and_spillover_summary.json",selectivity)
    save_csv(out/"target_and_spillover_summary.csv",selectivity)
    examples=[]
    for injected in TRAITS:
        largest=max((table[injected,j,3] for j in TRAITS if j!=injected and table[injected,j,3]["score_delta"] is not None),key=lambda r:abs(r["score_delta"]))
        scored=largest["scored_trait"]
        candidates=[q for q in case_ids if lookup[f"{injected}_alpha_3",q,scored]["score"] is not None and lookup["baseline",q,scored]["score"] is not None]
        candidates.sort(key=lambda q:abs(lookup[f"{injected}_alpha_3",q,scored]["score"]-lookup["baseline",q,scored]["score"]),reverse=True)
        for q in candidates[:2]:
            baseline=lookup["baseline",q,scored];active=lookup[f"{injected}_alpha_3",q,scored]
            examples.append(dict(injected_trait=injected,scored_trait=scored,case_id=q,question=active["question"],
                baseline_answer=baseline["answer_segment"],active_answer=active["answer_segment"],
                baseline_score=baseline["score"],active_score=active["score"],
                baseline_quality=quality_status(baseline),active_quality=quality_status(active),
                selection="Two largest absolute per-question changes in each row's largest absolute mean off-target effect at alpha=3; illustrative extremes, not representative or independent evidence."))
    save_json(out/"largest_spillover_examples.json",examples)
    summary=["跨维度干预分析", "",
             "沿用原来的五套评测题，每列20题；每个注入条件完整运行全部五套题库。不同列的题目不是同一个情境。",
             "矩阵每列使用该列属性有明确表达机会的探针。同一题不涉及的其他维度不是0分；其他列使用各自对应探针回答。",
             "这不是把一段单属性回答交给五套Rubric评分；没有用目标属性缺席制造非目标不变。",
             "每个注入方向沿用自己的已选层、向量方法和注入范围。因此结论是当前整套干预方法的功能性外溢，而非纯粹几何正交的证明。",
             "同一被测维度内，所有注入方向和强度使用完全相同的原题、原作答要求、固定Rubric和模型参数。不同列保留各自原有作答要求。基线每套题只有20条，不把复用基线算成额外独立样本。",
             "原最优层与原题库已用于前期选择，本章属于现有题库上的探索性跨维度检验，不称为独立确认。复用500条原回答，新增1600条回答，全部统一重新评分及质检。",
             "每个评分矩阵格子在该注入方向的所有alpha上固定有数字分数的相同情境，质量失败单独报告，不按结果筛选题目。",
             "质检通过、证据明确的错误、待核对以及生成截断分别统计；待核对不是已经证实的事实错误。严格通过率把待核对留在分母。",
             "区间按同一情境的干预减基线分差配对bootstrap，未包含生成和裁判重复的不确定性，未校正多格查看。",
             "±5分只是事先给出的描述性小效应范围；不显著或点估计小都不能直接证明独立。",
             "若区间很宽，正确结论是证据不足；若非目标显著变化，应如实记录外溢，不通过改题或改Rubric把它压到零。", ""]
    for alpha in DISPLAY_COEFFICIENTS:
        summary.append(f"alpha={alpha}：")
        for r in selectivity:
            if r["alpha"]==alpha:
                summary.append(f"  {r['injected_trait']}：目标变化={r['target_delta']}，非目标绝对变化均值={r['mean_abs_off_target_delta']}；最大非目标={r['largest_off_target_dimension']}({r['largest_off_target_delta']})；目标回答质检通过={r['target_quality_percent']:.1f}%。")
    (out/"实验结论与图注.txt").write_text("\n".join(summary),encoding="utf-8")
    # 便于论文写作：同时输出设计、结果和证据边界，数值直接取完整统计表。
    text=["单维偏好干预的跨维度效应与选择性", "",
          "实验目的", "检验在增强某一出行偏好的同时，其他偏好的表达及选择是否变化，并检查事实一致性。向量几何上的相似或正交不能替代这种行为检验。", "",
          "实验设计", "沿用五套原评测题，每套20道。分别注入五条已选向量，在alpha=-1、0、1、2、3下作答。每列使用该维度原题、原作答要求与固定Rubric；列内各注入条件完全一致。每个回答只按其题库对应维度评分，其他维度不记零分。",
          "共享100条无干预回答；五个注入方向各有4个非零强度，每个条件回答100题。总计2100条独立保存的回答。500条来自原扫描的基线/同维度回答，1600条为本次新增；所有回答统一重新评分、质检。",
          "每次特质评分使用首Token的Top-20候选数字概率计算0—100的条件期望，固定重复3次取平均；请求温度1。独立事实质检温度0。裁判只见题目、原回答和被测维度Rubric，不知道注入方向、层号和强度。",
          "每个矩阵格以同题干预分减去同题基线分，再对题目取平均。在该注入方向全部强度上固定数字完整题集；报告按题配对bootstrap的描述性95%区间。原始明确选择比例、质检通过率、待核对率、截断率及全强度质检通过的配对敏感性结果另存统计表。", "",
          "注入设置（实际层号；原始向量直接乘alpha，无额外缩放）"]
    for r in protocol["vectors"]:
        text.append(f"{r['trait']}：第{r['layer']}层；{r['method']}；注入范围={r['scope']}。")
    text.extend(["", "主要结果：alpha=3相对无干预", "维度 | 目标分差[95%区间] | 高偏好选择率变化(百分点) | 目标回答通过率 | 非目标绝对分差均值 | 最大外溢"])
    for trait in TRAITS:
        own=table[trait,trait,3]
        r=next(r for r in selectivity if r["injected_trait"]==trait and r["alpha"]==3)
        text.append(f"{trait} | {own['score_delta']:+.2f}[{own['score_ci_low']:+.2f},{own['score_ci_high']:+.2f}] | {own['choice_delta_pp']:+.1f} | {own['active_quality_percent']:.1f}% | {r['mean_abs_off_target_delta']:.2f} | {r['largest_off_target_dimension']} {r['largest_off_target_delta']:+.2f}")
    text.extend(["", "剂量趋势（均为相对本维度无干预的评分变化）"])
    for trait in TRAITS:
        text.append(trait+"："+"；".join(f"alpha={a}：{table[trait,trait,a]['score_delta']:+.2f}" for a in COEFFICIENTS))
    text.extend(["", "质检配对敏感性：alpha=3与基线双方均通过的题目（与全强度固定通过题集分开）"])
    for trait in TRAITS:
        own=table[trait,trait,3]
        text.append(f"{trait}：目标配对分差={own['pairwise_quality_score_delta']}，N={own['pairwise_quality_n']}；全强度均通过题集N={own['quality_paired_n']}。")
    text.extend(["", "解释时应区分的结论", "1. 对角线反映目标控制，非对角线反映跨维度外溢。目标增强而非目标变化较小，只支持相对的控制选择性；非目标变化不必严格为零。",
        "2. 不同列采用不同题目与Rubric，评分虽同为0—100，不能假定五个量表具有完全相同的心理测量单位。非目标绝对分差均值只是描述性摘要，不是已校准的独立性指标。",
        "3. 各向量使用各自已选层、提取方法和注入范围。本实验检验的是现有完整干预方法的功能性影响，不能单独归因于向量方向的语义纯度。",
        "4. 原层数和原题库参与过前期选择，本章为探索性检验；目标效果较强不是独立数据上的重复验证。新情境迁移应另设补充实验，不能以效果较好的题库替换较差结果。",
        "5. 置信区间只反映本批题目分差的不确定性，未包含生成重复和裁判重复的全部不确定性，也未作多重比较校正。20题上非目标不显著或区间覆盖0，不足以证明独立。±5分仅为预设描述范围。",
        "6. 主分析保留质检失败回答以避免按干预结果筛选；因此不能把所有评分提升都当作事实正确的偏好提升。应联合查看质检通过率、待核对比例和质检配对敏感性分析，后者也存在选择偏差。",
        "7. 这些是模拟旅客的模型回答与选择，不是现实人群的实际行为。", "", "文件说明",
        "05_main_matrix_alpha3_with_intervals.png/svg：建议正文主图；其余强度见01，完整剂量轨迹见04。",
        "02_choice_effect_matrices.png/svg：五个注入方向对五类高偏好端选择率的影响。",
        "03_quality_effect_matrices.png/svg与06_quality_statuses_alpha3.png/svg：通过率变化及明确错误/待核对/截断构成。",
        "cross_trait_effects.csv：125行完整数据，含均值变化、区间、选择比例、质量和质检配对敏感性分析。"])
    (out/"本章实验分析.txt").write_text("\n".join(text),encoding="utf-8")
    for name in ("10_analyze_cross_trait.py",):shutil.copy2(ROOT/name,out/name)
    save_json(out/"analysis_settings.json",dict(run_dir=str(run),coefficients=COEFFICIENTS,
        bootstrap_repeats=BOOTSTRAP_REPEATS,random_seed=RANDOM_SEED,small_effect_margin=SMALL_EFFECT_MARGIN,
        source_score_sha256=hashlib.sha256((run/"module_scores.json").read_bytes()).hexdigest(),
        complete_score_count=len(scores),scenarios_per_evaluated_dimension=len(case_ids)))
    print("\n".join(summary),flush=True)


if __name__=="__main__":main()
