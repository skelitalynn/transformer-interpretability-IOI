import json
import re
import time
from typing import List, Dict, Tuple
from tqdm import tqdm
import random
import argparse


def generate_data() -> Tuple[List[Dict], float]:
    t0 = time.time()
    random.seed(42)
    with open('names.json', 'r', encoding='utf-8') as f:
        names = json.load(f)
    with open('sentences.json', 'r', encoding='utf-8') as f:
        sentences = json.load(f)

    results = []
    for _ in range(50):
        name_pair = random.choice(names)
        sp = random.choice(sentences)
        clean = sp["clean"].replace("A ", name_pair["A"] + " ").replace("B ", name_pair["B"] + " ")
        corrupted = sp["corrupted"].replace("A ", name_pair["A"] + " ").replace("B ", name_pair["B"] + " ")
        clean_answer = " " + name_pair["B"]
        corrupted_answer = " " + name_pair["A"]
        results.append({
            "clean": clean,
            "corrupted": corrupted,
            "clean_answer": clean_answer,
            "corrupted_answer": corrupted_answer
        })
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    duration = time.time() - t0
    print(f"[OK] 生成数据 data.json 共 {len(results)} 条，用时 {duration:.3f}s")
    return results, duration


def check_sentence_structure(data: List[Dict]) -> Tuple[List[Dict], float]:
    t0 = time.time()
    checked = []
    clean_pattern = r"After (.+?) and (.+?) (.+?), \1 (.+?) to"
    corrupted_pattern = r"After (.+?) and (.+?) (.+?), \2 (.+?) to"
    for item in data:
        cm = re.match(clean_pattern, item["clean"])
        if not cm:
            continue
        xm = re.match(corrupted_pattern, item["corrupted"])
        if not xm:
            continue
        a1, b1, u1, v1 = cm.groups()
        a2, b2, u2, v2 = xm.groups()
        if a1 != a2 or b1 != b2:
            continue
        if u1 != u2 or v1 != v2:
            continue
        if " " + a1 != item["corrupted_answer"] or " " + b1 != item["clean_answer"]:
            continue
        checked.append(item)
    with open('data_check1.json', 'w', encoding='utf-8') as f:
        json.dump(checked, f, indent=2)
    duration = time.time() - t0
    print(f"[OK] 结构校验 data_check1.json 保留 {len(checked)}/{len(data)} 条，用时 {duration:.3f}s")
    return checked, duration


def filter_with_gpt2(data: List[Dict]) -> Tuple[List[Dict], float]:
    t0 = time.time()
    import torch
    from transformer_lens import HookedTransformer

    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )
    filtered = []
    for item in tqdm(data, desc="Filter with GPT-2"):
        with torch.no_grad():
            cg = model.generate(item["clean"], max_new_tokens=1, temperature=0, do_sample=False, return_type="tokens")
            kg = model.generate(item["corrupted"], max_new_tokens=1, temperature=0, do_sample=False, return_type="tokens")
            ct = model.to_string(cg[0, -1])
            kt = model.to_string(kg[0, -1])
        if (ct in item["clean_answer"] or item["clean_answer"] in ct) and (kt in item["corrupted_answer"] or item["corrupted_answer"] in kt):
            item["clean_generated"] = ct
            item["corrupted_generated"] = kt
            filtered.append(item)
    with open('data_check2.json', 'w', encoding='utf-8') as f:
        json.dump(filtered, f, indent=2)
    duration = time.time() - t0
    print(f"[OK] 筛选 data_check2.json 保留 {len(filtered)}/{len(data)} 条，用时 {duration:.3f}s")
    return filtered, duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["generate", "check", "filter", "all"], default="all")
    args = parser.parse_args()

    timings = {}

    if args.step == "generate":
        _, t_gen = generate_data()
        timings["local_generate_s"] = t_gen
    elif args.step == "check":
        with open('data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        _, t_check = check_sentence_structure(data)
        timings["local_check_s"] = t_check
    elif args.step == "filter":
        with open('data_check1.json', 'r', encoding='utf-8') as f:
            data_check1 = json.load(f)
        filtered, t_filter = filter_with_gpt2(data_check1)
        timings["local_filter_s"] = t_filter
        if len(filtered) == 0:
            print("[WARN] data_check2.json 为空，后续步骤可能没有可用样本。")
    else:  # all
        data, t_gen = generate_data()
        timings["local_generate_s"] = t_gen
        checked, t_check = check_sentence_structure(data)
        timings["local_check_s"] = t_check
        filtered, t_filter = filter_with_gpt2(checked)
        timings["local_filter_s"] = t_filter
        if len(filtered) == 0:
            print("[WARN] data_check2.json 为空，后续步骤可能没有可用样本。")

    with open('local_pre_timing.json', 'w', encoding='utf-8') as f:
        json.dump(timings, f, indent=2)


if __name__ == "__main__":
    main()


