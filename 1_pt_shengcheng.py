
# Portable paths for the public research release.
import os as _TRAVEL_OS
import sys as _TRAVEL_SYS
from pathlib import Path as _TRAVEL_PATH
from travel_config import MODEL_PATH as _TRAVEL_MODEL_PATH, LLAMA_MODEL_PATH as _TRAVEL_LLAMA_PATH, QWEN3B_MODEL_PATH as _TRAVEL_QWEN3B_PATH

# ① 生成对比回答：5 对提示 × 20 道题 × 正反两侧 = 200 条回答。
# 本步保存文字和 Token IDs；第三步再读取隐藏状态，生成 .pt 向量。

# ==================== 本步骤实验参数 ====================
TRAIT = "flexibility"  # environment、safety、comfort、convenience、flexibility
MODEL_PATH = _TRAVEL_MODEL_PATH
LOAD_IN_8BIT = True
WEIGHT_READ_BACKEND = "pread"  # Windows稳定读取，模型数值不变
BATCH_SIZE = 32         # 本机统一批量上限；第四、第六步使用同一上限
MAX_NEW_TOKENS = 192
DO_SAMPLE = False      # False 表示贪心生成
RESUME_RUN_DIR = ""     # 留空：新建实验；续跑时填写上次未完成的结果目录
# ======================================================

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model():
    """第一步只生成回答，不保存隐藏状态。"""
    from model_loading import load_local_model
    model = load_local_model(MODEL_PATH, LOAD_IN_8BIT, WEIGHT_READ_BACKEND)
    model.config.output_hidden_states = False
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "left"  # 不同长度的输入，左侧补齐后一起生成
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def generate_batch(model, tokenizer, tasks):
    """每条任务仍有自己的系统提示和题目，互不共享对话。"""
    texts = [tokenizer.apply_chat_template([
        {"role": "system", "content": task["system_prompt"]},
        {"role": "user", "content": task["question"]},
    ], tokenize=False, add_generation_prompt=True) for task in tasks]
    inputs = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
    inputs = inputs.to(model.get_input_embeddings().weight.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    results = []
    input_width = inputs.input_ids.shape[1]
    for mask, output in zip(inputs.attention_mask, outputs):
        # 补齐的 token 不能写入第三步的输入；只保留本条真正的提示。
        prompt = output[:input_width][mask.bool()].tolist()
        generated = output[input_width:].tolist()
        ended = tokenizer.eos_token_id in generated
        if ended:
            # 保留本条第一个 EOS，去掉为等待其他回答而补齐的后续 token。
            generated = generated[:generated.index(tokenizer.eos_token_id) + 1]
        results.append({
            "response": tokenizer.decode(generated, skip_special_tokens=True),
            "outputs_tokens": prompt + generated,
            "prompt_len": len(prompt),
            "new_tokens": len(generated),
            "hit_token_limit": len(generated) >= MAX_NEW_TOKENS and not ended,
        })
    return results


def main():
    import run_paths
    from methology.persona.trait_instructions import TRAITS

    if TRAIT not in TRAITS:
        raise ValueError(f"TRAIT请选择五个维度之一：{', '.join(TRAITS)}")
    if not isinstance(BATCH_SIZE, int) or BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE必须是大于等于1的整数。")
    run_paths.TRAIT = TRAIT
    trait_data = run_paths.load_trait_data("extract")
    metadata = {key: trait_data[key] for key in ("framework_version", "trait", "data_fingerprint")}
    instructions, questions = trait_data["instruction"], trait_data["questions"]
    total_pairs = len(instructions) * len(questions)

    if RESUME_RUN_DIR:
        run_dir = Path(RESUME_RUN_DIR)
        if not run_dir.is_absolute():
            run_dir = run_paths.RUNS_ROOT / run_dir
        run_paths.validate_run(run_dir)
    else:
        run_dir = run_paths.new_run()
    output_file = run_dir / f"{TRAIT}_raw_pairs_100.json"
    progress_file = run_dir / "generation_progress.json"
    config = {"model_path": MODEL_PATH, "load_in_8bit": LOAD_IN_8BIT, "weight_read_backend": WEIGHT_READ_BACKEND,
              "batch_size": BATCH_SIZE, "max_new_tokens": MAX_NEW_TOKENS, "do_sample": DO_SAMPLE}
    config_path = run_dir / "generation_config.json"
    if RESUME_RUN_DIR:
        previous = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        previous.setdefault("weight_read_backend", "mmap")  # 旧记录默认采用内存映射
        if previous != config:
            raise ValueError("续跑须使用相同模型、批量和生成参数；请恢复原参数或留空RESUME_RUN_DIR新建实验。")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.LATEST_POINTER.write_text(run_dir.name, encoding="utf-8")

    saved_file = output_file if output_file.exists() else progress_file
    saved = json.loads(saved_file.read_text(encoding="utf-8")) if saved_file.exists() else []
    run_paths.assert_record_compatibility(saved, "extract")
    pairs = {item["pair_id"]: item for item in saved}
    tasks = []
    for inst_idx, instruction in enumerate(instructions):
        for q_idx, question in enumerate(questions):
            pair_id = inst_idx * len(questions) + q_idx
            if pair_id in pairs:
                continue
            pairs[pair_id] = {
                **metadata, "pair_id": pair_id, "instruction_idx": inst_idx,
                "question_idx": q_idx, "question_set_version": trait_data.get("question_set_version"),
                "question": question, "sys_pos": instruction["pos"], "sys_neg": instruction["neg"],
                "generation_batch_size": BATCH_SIZE,
            }
            for side in ("pos", "neg"):
                tasks.append({"pair_id": pair_id, "side": side,
                              "system_prompt": instruction[side], "question": question})

    print(f"本次生成维度: {TRAIT} | 每批 {BATCH_SIZE} 条 | 回答上限 {MAX_NEW_TOKENS} token")
    print(f"本次运行产物目录: {run_dir}")
    print(f"共 {total_pairs} 对、{total_pairs * 2} 条回答；已有 {len(saved)} 对，待生成 {len(tasks)} 条。", flush=True)
    started = time.perf_counter()
    if tasks:
        model, tokenizer = load_model()
        generation_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS, "do_sample": DO_SAMPLE,
            "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        for start in range(0, len(tasks), BATCH_SIZE):
            batch = tasks[start:start + BATCH_SIZE]
            try:
                results = generate_batch(model, tokenizer, batch)
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError("显存不足。请减小开头的BATCH_SIZE，并留空RESUME_RUN_DIR重跑；已完成的数据仍在上面的目录。") from None
            for task, result in zip(batch, results):
                pair, side = pairs[task["pair_id"]], task["side"]
                pair[f"response_{side}"] = result["response"]
                pair[f"outputs_{side}_tokens"] = result["outputs_tokens"]
                pair[f"prompt_{side}_len"] = result["prompt_len"]
                pair[f"{side}_new_tokens"] = result["new_tokens"]
                pair[f"{side}_hit_token_limit"] = result["hit_token_limit"]
                pair["generation_kwargs"] = generation_kwargs
            # 每批保存完整的正负对；中断后可从这些已完成的样本继续。
            saved = [pairs[key] for key in sorted(pairs)
                     if "response_pos" in pairs[key] and "response_neg" in pairs[key]]
            temporary = progress_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(progress_file)
            if len(saved) == total_pairs:
                progress_file.replace(output_file)
            print(f"已生成 {len(saved)}/{total_pairs} 对 | 本次用时 {(time.perf_counter() - started) / 60:.1f} 分钟", flush=True)

    truncated = sum(item[f"{side}_hit_token_limit"] for item in saved for side in ("pos", "neg"))
    print("\n=============== 第一阶段完成 ===============")
    print(f"成功生成 {len(saved)} 对回答；达到生成上限：{truncated}/{len(saved) * 2} 条。")
    print(f"文字、Token IDs 和提示边界：{output_file}")


if __name__ == "__main__":
    main()
