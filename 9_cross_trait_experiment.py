"""五套原评测题的交叉干预：每个向量分别回答五套题，按题库对应Rubric评分。"""

# Portable paths for the public research release.
import os as _TRAVEL_OS
import sys as _TRAVEL_SYS
from pathlib import Path as _TRAVEL_PATH
from travel_config import MODEL_PATH as _TRAVEL_MODEL_PATH, LLAMA_MODEL_PATH as _TRAVEL_LLAMA_PATH, QWEN3B_MODEL_PATH as _TRAVEL_QWEN3B_PATH


# ==================== 本章实验参数 ====================
STEP = "all"                 # all：补回答并评分；generate：只补回答；score：只补评分。
OUTPUT_DIR = "runs/cross_trait_original_20261019"
MODEL_PATH = _TRAVEL_MODEL_PATH
LOAD_IN_8BIT = True
WEIGHT_READ_BACKEND = "pread"
BATCH_SIZE = 32               # 每套20题一起回答，实际批量20，与原扫描一致。
MAX_NEW_TOKENS = 160
DO_SAMPLE = False
RANDOM_SEED = 20261017
COEFFICIENTS = [-1, 0, 1, 2, 3]
QUESTION_IDS = list(range(20))
API_WORKERS = 12
MAX_SCORING_PASSES = 2        # 仅补失败或缺失项，不按分数高低重试。
REUSE_ORIGINAL_ANSWERS = True # 只复用原回答；本章统一重新评分、质检。
RUNS = [
    # 注入维度、实际层号、方法、注入范围、原运行目录、原扫描目录。
    ("comfort", 18, "prompt_last", "response", "20261018_141422_comfort", "layer_scan_prompt_last_batch64"),
    ("environment", 16, "prompt_last", "prompt", "20261018_161705_environment", "layer_scan_prompt_last_prompt_batch32"),
    ("safety", 14, "response_mean", "all", "20261018_183835_safety", "layer_scan_response_mean_all_batch32"),
    ("convenience", 14, "response_mean", "system", "20261018_212758_convenience", "layer_scan_response_mean_system_batch32"),
    ("flexibility", 18, "response_mean", "all", "20261018_235055_flexibility", "layer_scan_response_mean_all_batch32"),
]
# ====================================================

from concurrent.futures import ThreadPoolExecutor, as_completed
import ast
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import torch
from transformers import AutoTokenizer, set_seed
from model_loading import load_local_model
from steering_utils import SteeringHook, encode_steering_batch
import persona_judge_utils as judge

ROOT = Path(__file__).resolve().parent
TRAITS = [r[0] for r in RUNS]
sys.stdout.reconfigure(encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conditions():
    return [dict(name="baseline", injected_trait=None, alpha=0)] + [
        dict(name=f"{trait}_alpha_{alpha}", injected_trait=trait, alpha=alpha)
        for trait in TRAITS for alpha in COEFFICIENTS if alpha != 0]


def prepare():
    out = ROOT / OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    judge.CACHE_DIR = out / "judge_cache"  # 本章缓存独立，不复用旧评分或旧质检。
    questions, rubrics, vectors, sources = [], {}, {}, {}
    for index, (trait, layer, method, scope, folder, scan) in enumerate(RUNS):
        run = ROOT / "runs" / folder
        data = read(run / "trait_data_eval.json")
        settings = read(run / scan / "settings.json")
        vector_file = run / f"{trait}_vectors_{'prompt_last' if method == 'prompt_last' else 'all_layers'}.pt"
        if digest(vector_file) != settings["vector_sha256"]:
            raise ValueError(f"{trait}的向量与原扫描不一致。")
        # A local checkpoint directory can change when the public release is moved.
        assert settings["load_in_8bit"] == LOAD_IN_8BIT
        assert settings["vector_method"] == method and settings["intervention_scope"] == scope
        assert settings["max_new_tokens"] == MAX_NEW_TOKENS and settings["do_sample"] == DO_SAMPLE
        vectors[trait] = torch.load(vector_file, map_location="cpu", weights_only=True)[layer]
        rubrics[trait] = data["eval_prompt"]
        for qid in QUESTION_IDS:
            question = data["questions"][qid]
            module = dict(number=1, trait=trait, judge_question=question, positive_option=data["positive_options"][qid])
            questions.append(dict(question_id=qid*10+index, case_id=qid, scored_trait=trait,
                applicable_traits=[trait], question=question, system_prompt=settings["system_prompt"], modules=[module]))
        for source in (vector_file, run/"trait_data_eval.json", run/scan/"settings.json", run/scan/"scores.json"):
            sources[str(source)] = digest(source)
    protocol = dict(question_bank="original_five_banks", questions=questions, coefficients=COEFFICIENTS,
        vectors=[dict(trait=t, layer=l, method=m, scope=s, run=r, scan=c) for t,l,m,s,r,c in RUNS],
        model_path=MODEL_PATH, load_in_8bit=LOAD_IN_8BIT, weight_read_backend=WEIGHT_READ_BACKEND,
        batch_size=BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE, random_seed=RANDOM_SEED,
        judge_configs={t:judge.scoring_config(r) for t,r in rubrics.items()}, source_files_sha256=sources,
        vector_normalization="none; alpha directly multiplies the saved vector",
        judge_inputs="Only original question, verbatim response and evaluated dimension's fixed rubric; injection, alpha and layer are hidden.",
        design_note="Each column uses its own original bank and archived response instructions, held identical across all injected directions and alpha. Baselines are shared. Exploratory cross-effects of selected intervention procedures; not independent confirmation or proof of orthogonality.",
        analysis_note="Within-question differences; missing scores are not zero. Primary expression analysis retains quality failures; quality and quality-paired sensitivity are reported separately.")
    path = out / "protocol.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("本目录已存参数与当前不同，请用新的OUTPUT_DIR保存不同实验。")
    save(path, protocol)
    save(out/"questions.json",questions)
    save(out/"rubrics.json",rubrics)
    torch.save(vectors,out/"intervention_vectors.pt")  # 保存本章实际使用的五条向量。
    # 保存原五对提示、提取题、评测题、作答提示及Rubric，原目录保持原样。
    for trait, _, _, _, folder, scan in RUNS:
        target=out/"source_prompts"/trait
        target.mkdir(parents=True,exist_ok=True)
        for name in ("trait_data_eval.json","trait_data_extract.json","prompts_snapshot.json","neutral_system_prompt.txt"):
            shutil.copy2(ROOT/"runs"/folder/name,target/name)
        shutil.copy2(ROOT/"runs"/folder/scan/"settings.json",target/"original_scan_settings.json")
    for name in ("9_cross_trait_experiment.py","steering_utils.py","model_loading.py"):
        shutil.copy2(ROOT/name,out/name)
    # 裁判实现也保存，但先移除密钥。
    source=(ROOT/"persona_judge_utils.py").read_text(encoding="utf-8")
    lines=source.splitlines(keepends=True)
    for node in reversed(ast.parse(source).body):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="API_KEY" for t in node.targets):
            lines[node.lineno-1:node.end_lineno]=['API_KEY = ""  # 复现时填写自己的密钥。\n']
    (out/"persona_judge_utils_without_key.py").write_text("".join(lines),encoding="utf-8")
    return out,questions,rubrics,vectors


def answer_key(row):
    return row["condition"],row["question_id"]


def reuse_answers(out,questions):
    path=out/"answers.json"
    answers=read(path) if path.exists() else []
    existing={answer_key(r) for r in answers}
    by_trait={t:{q["case_id"]:q for q in questions if q["scored_trait"]==t} for t in TRAITS}
    if REUSE_ORIGINAL_ANSWERS:
        for trait,layer,method,scope,folder,scan in RUNS:
            source=ROOT/"runs"/folder/scan/"scores.json"
            for row in read(source):
                if row["physical_layer"]!=layer or row["question_id"] not in QUESTION_IDS or row["coeff"] not in COEFFICIENTS:
                    continue
                q=by_trait[trait][row["question_id"]]
                alpha=int(row["coeff"])
                condition="baseline" if alpha==0 else f"{trait}_alpha_{alpha}"
                if (condition,q["question_id"]) in existing:continue
                assert row["question"]==q["question"] and row["positive_option"]==q["modules"][0]["positive_option"]
                assert row["vector_method"]==method
                assert alpha==0 or row["intervention_scope"]==scope
                assert row["generation_actual_batch_size"]==len(QUESTION_IDS)==20
                answers.append(dict(condition=condition,injected_trait=trait if alpha else None,alpha=alpha,prompt_control=None,
                    question_id=q["question_id"],case_id=q["case_id"],probe_trait=trait,
                    layer=layer if alpha else None,vector_method=method if alpha else None,intervention_scope=scope if alpha else None,
                    response=row["response"],response_tokens=row["response_new_tokens"],
                    response_hit_token_limit=row["response_hit_token_limit"],actual_batch_size=row["generation_actual_batch_size"],
                    system_prompt=q["system_prompt"],reused_from=str(source),
                    reused_source_key=dict(physical_layer=layer,coeff=row["coeff"],question_id=row["question_id"])))
                existing.add((condition,q["question_id"]))
    save(path,answers)
    print(f"已有回答{len(answers)}条；其中复用原回答{sum('reused_from' in r for r in answers)}条。旧分数和质检不复用。",flush=True)
    return answers


def generate(out,questions,vectors,answers):
    existing={answer_key(r) for r in answers}
    todo=[]
    for condition in conditions():
        for bank in TRAITS:
            batch=[q for q in questions if q["scored_trait"]==bank and (condition["name"],q["question_id"]) not in existing]
            if batch:todo.append((condition,bank,batch))
    print(f"需要新增{sum(len(b) for _,_,b in todo)}条回答；共{len(todo)}个条件×题库批次。",flush=True)
    if not todo:return answers
    model=load_local_model(MODEL_PATH,LOAD_IN_8BIT,WEIGHT_READ_BACKEND)
    model.eval()
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side="left"
    if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.eos_token_id
    device=model.get_input_embeddings().weight.device
    for group,(condition,bank,missing) in enumerate(todo,1):
        trait=condition["injected_trait"]
        _,layer,method,scope,_,_=next((r for r in RUNS if r[0]==trait),RUNS[0])
        vector=vectors[trait] if trait else torch.zeros_like(vectors[TRAITS[0]])
        system=missing[0]["system_prompt"]
        started=time.monotonic()
        with SteeringHook(model,layer,vector,condition["alpha"],scope) as hook:
            for start in range(0,len(missing),BATCH_SIZE):
                batch=missing[start:start+BATCH_SIZE]
                assert all(q["system_prompt"]==system for q in batch)
                set_seed(RANDOM_SEED)
                texts=[tokenizer.apply_chat_template([dict(role="system",content=system),
                    dict(role="user",content=q["question"])],tokenize=False,add_generation_prompt=True) for q in batch]
                inputs,mask=encode_steering_batch(tokenizer,texts,system,scope)
                inputs=inputs.to(device);hook.system_mask=mask
                length=inputs["input_ids"].shape[1]
                with torch.inference_mode():
                    outputs=model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS,do_sample=DO_SAMPLE,use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
                for q,output in zip(batch,outputs):
                    tokens=output[length:].tolist()
                    ended=tokenizer.eos_token_id in tokens
                    if ended:tokens=tokens[:tokens.index(tokenizer.eos_token_id)]
                    answers.append(dict(condition=condition["name"],injected_trait=trait,alpha=condition["alpha"],prompt_control=None,
                        question_id=q["question_id"],case_id=q["case_id"],probe_trait=bank,layer=layer if trait else None,
                        vector_method=method if trait else None,intervention_scope=scope if trait else None,
                        response=tokenizer.decode(tokens,skip_special_tokens=True).strip(),response_tokens=len(tokens),
                        response_hit_token_limit=not ended,actual_batch_size=len(batch),system_prompt=system))
                save(out/"answers.json",answers)
        print(f"生成{group}/{len(todo)}：{condition['name']} → {bank}；累计{len(answers)}条；本批{time.monotonic()-started:.1f}秒。",flush=True)
    del model,tokenizer
    gc.collect();torch.cuda.empty_cache()
    return answers


def score_modules(out,questions,rubrics,answers):
    question_map={q["question_id"]:q for q in questions}
    path=out/"module_scores.jsonl"
    previous=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    latest={(r["condition"],r["question_id"],r["scored_trait"]):r for r in previous}
    tasks=[]
    for answer in answers:
        module=question_map[answer["question_id"]]["modules"][0]
        old=latest.get((answer["condition"],answer["question_id"],module["trait"]))
        if not old or (old["score"] is None and old["scoring_pass"]<MAX_SCORING_PASSES):tasks.append((answer,module,old))
    if not tasks:return list(latest.values())
    client=judge.create_deepseek_client().with_options(timeout=90,max_retries=0)

    def work(task):
        answer,module,old=task
        question,segment=module["judge_question"],answer["response"]
        # 裁判只见原题、原回答和该列固定Rubric，不见注入方向、alpha或层号。
        score,diagnostics=judge.score_answer(client,rubrics[module["trait"]],question,segment)
        quality,quality_diagnostics=judge.assess_answer_quality(client,question,segment)
        option=judge.extract_selected_option(segment,question)
        return dict(condition=answer["condition"],question_id=answer["question_id"],case_id=answer["case_id"],
            injected_trait=answer["injected_trait"],alpha=answer["alpha"],prompt_control=None,scored_trait=module["trait"],
            applicable=True,question=question,answer_segment=segment,positive_option=module["positive_option"],
            scoring_pass=old["scoring_pass"]+1 if old else 1,whole_answer_truncated=answer["response_hit_token_limit"],
            score=score,score_diagnostics=diagnostics,selected_option=option,
            high_choice=(option==module["positive_option"]) if option else None,
            quality_ok=bool(quality and not answer["response_hit_token_limit"]),quality_diagnostics=quality_diagnostics)

    print(f"待评{len(tasks)}条；特质温度{judge.JUDGE_TEMPERATURE}；固定重复{judge.SCORE_REPEATS}；独立质检温度{judge.QUALITY_TEMPERATURE}。",flush=True)
    started=time.monotonic()
    with path.open("a",encoding="utf-8") as log,ThreadPoolExecutor(max_workers=API_WORKERS) as pool:
        futures=[pool.submit(work,task) for task in tasks]
        for i,future in enumerate(as_completed(futures),1):
            row=future.result()
            log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n");log.flush()
            latest[row["condition"],row["question_id"],row["scored_trait"]]=row
            if i%50==0 or i==len(tasks):print(f"评分{i}/{len(tasks)}，耗时{time.monotonic()-started:.0f}秒。",flush=True)
    result=list(latest.values());save(out/"module_scores.json",result)
    return result


def main():
    if STEP not in ("all","generate","score"):raise ValueError("STEP应为all、generate或score。")
    out,questions,rubrics,vectors=prepare()
    answers=reuse_answers(out,questions)
    if STEP!="score":answers=generate(out,questions,vectors,answers)
    if STEP!="generate":
        for _ in range(MAX_SCORING_PASSES):scores=score_modules(out,questions,rubrics,answers)
        print(f"评分已保存{len(scores)}条，数字分有效{sum(r['score'] is not None for r in scores)}条。",flush=True)
    print("本章数据目录："+str(out),flush=True)


if __name__=="__main__":main()
