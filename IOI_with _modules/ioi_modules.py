"""
IOI 项目的独立模块：每个GPU密集环节可单独调用并返回计时
与 IOI.ipynb 逻辑完全一致
"""
import os
# 强制使用本地缓存，不联网下载
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import time
import torch
import gc
import random
from functools import partial
from tqdm import tqdm
from transformer_lens import HookedTransformer, utils
from jaxtyping import Float
from torch import Tensor
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def load_model_safely(device="cpu"):
    """
    安全加载模型，优先使用本地缓存
    """
    print(f"[日志] 加载 GPT-2 模型（离线模式）...")
    torch.set_grad_enabled(False)
    
    # 设置离线模式后，会自动从 ~/.cache/huggingface 查找
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )
    model.to(device)
    model.eval()
    print(f"[日志] 模型加载成功，设备: {device}")
    return model


def filter_with_gpt2(input_file: str, output_file: str) -> dict:
    """
    使用GPT-2筛选样本
    """
    t0 = time.time()
    torch.set_grad_enabled(False)
    
    model = load_model_safely(device="cpu")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filtered = []
    for item in tqdm(data, desc="Filter with GPT-2"):
        with torch.no_grad():
            cg = model.generate(item["clean"], max_new_tokens=1, temperature=0, do_sample=False, return_type="tokens")
            kg = model.generate(item["corrupted"], max_new_tokens=1, temperature=0, do_sample=False, return_type="tokens")
            ct = model.to_string(cg[0, -1])
            kt = model.to_string(kg[0, -1])
        
        if (ct in item["clean_answer"] or item["clean_answer"] in ct) and \
           (kt in item["corrupted_answer"] or item["corrupted_answer"] in kt):
            item["clean_generated"] = ct
            item["corrupted_generated"] = kt
            filtered.append(item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, indent=2)
    
    elapsed = time.time() - t0
    print(f"[OK] GPT-2样本筛选完成，保留 {len(filtered)}/{len(data)} 条，用时 {elapsed:.3f}s -> {output_file}")
    
    return { "time": elapsed, "filtered_count": len(filtered), "total_count": len(data) }


def get_clean_activations(input_file: str, output_file: str) -> dict:
    # ... (函数内容不变) ...
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    model = load_model_safely(device=device)
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    def get_logits_diff(logits, token1, token2): return logits[token1] - logits[token2]
    clean_z, clean_logits_diffs, corrupted_logits_diffs = [], [], []
    clean_sentences, corrupted_sentences, clean_answers, corrupted_answers = [], [], [], []
    for item in tqdm(data, desc="Collect activations"):
        clean_tokens = model.to_tokens(item["clean"]).to(device)
        corrupted_tokens = model.to_tokens(item["corrupted"]).to(device)
        if clean_tokens.shape != corrupted_tokens.shape: continue
        clean_ans = model.to_tokens(item["clean_generated"])[0][1].to(device)
        corrupt_ans = model.to_tokens(item["corrupted_generated"])[0][1].to(device)
        with torch.no_grad():
            corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens, names_filter=lambda n: n.endswith("hook_z"), remove_batch_dim=True)
            clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda n: n.endswith("hook_z"), remove_batch_dim=True)
        cld = get_logits_diff(clean_logits[0][-1], clean_ans, corrupt_ans)
        cod = get_logits_diff(corrupted_logits[0][-1], clean_ans, corrupt_ans)
        clean_logits_diffs.append(cld); corrupted_logits_diffs.append(cod)
        clean_z.append(clean_cache.stack_activation("z").cpu())
        clean_sentences.append(clean_tokens.cpu()); corrupted_sentences.append(corrupted_tokens.cpu())
        clean_answers.append(clean_ans.cpu()); corrupted_answers.append(corrupt_ans.cpu())
        del clean_cache, corrupted_cache, clean_logits, corrupted_logits
        torch.cuda.empty_cache()
    save_data = {
        "clean_z": clean_z, "clean_sentences": clean_sentences, "corrupted_sentences": corrupted_sentences,
        "clean_answers": clean_answers, "corrupted_answers": corrupted_answers,
        "clean_logits_diff": clean_logits_diffs, "corrupted_logits_diff": corrupted_logits_diffs
    }
    torch.save(save_data, output_file)
    elapsed = time.time() - t0
    print(f"[OK] 缓存激活值完成，保留 {len(clean_answers)} 个有效样本，用时 {elapsed:.3f}s -> {output_file}")
    torch.cuda.empty_cache(); gc.collect()
    return { "time": elapsed, "valid_samples": len(clean_answers) }


def activation_patching(input_file: str, output_file: str) -> dict:
    # ... (函数内容不变) ...
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    model = load_model_safely(device=device)
    save_data = torch.load(input_file, map_location="cpu")
    clean_z = save_data["clean_z"]
    clean_logits_diffs = save_data["clean_logits_diff"]
    corrupted_logits_diffs = save_data["corrupted_logits_diff"]
    clean_sentences = save_data["clean_sentences"]
    corrupted_sentences = save_data["corrupted_sentences"]
    clean_answers = save_data["clean_answers"]
    corrupted_answers = save_data["corrupted_answers"]
    def get_logits_diff(logits, token1, token2): return logits[token1] - logits[token2]
    def patch_head_vector(c_vec, hook, head_index, cl_vec):
        c_vec[:, :, head_index] = cl_vec[:, head_index]
        return c_vec
    def ioi_metric(clean, corrupted, patched): return (patched - corrupted) / (clean - corrupted)
    case_n = len(clean_answers)
    rdm = list(range(case_n)) if case_n < 10 else random.sample(range(case_n), 10)
    results = torch.zeros(len(rdm), model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)
    total_patches = len(rdm) * model.cfg.n_layers * model.cfg.n_heads
    with tqdm(total=total_patches, desc="Activation patching") as pbar:
        for idx, i in enumerate(rdm):
            model.reset_hooks()
            corrupt_sent_i = corrupted_sentences[i].to(device)
            clean_ans_i = clean_answers[i].to(device)
            corrupt_ans_i = corrupted_answers[i].to(device)
            for layer in range(model.cfg.n_layers):
                for head in range(model.cfg.n_heads):
                    with torch.no_grad():
                        hook_fn = partial(patch_head_vector, head_index=head, cl_vec=clean_z[i][layer].to(device))
                        pl = model.run_with_hooks(corrupt_sent_i, fwd_hooks=[(utils.get_act_name("z", layer), hook_fn)], return_type="logits")
                        pld = get_logits_diff(pl[0][-1], clean_ans_i, corrupt_ans_i)
                        results[idx, layer, head] = ioi_metric(clean_logits_diffs[i], corrupted_logits_diffs[i], pld)
                        model.reset_hooks()
                        del pl
                    pbar.update(1)
            torch.cuda.empty_cache(); gc.collect()
    result_mean = results.mean(dim=0).cpu()
    torch.save(result_mean, output_file)
    elapsed = time.time() - t0
    print(f"[OK] 修补激活值完成，已聚合 {total_patches} 次patch为平均矩阵，用时 {elapsed:.3f}s -> {output_file}")
    torch.cuda.empty_cache(); gc.collect()
    return { "time": elapsed, "total_patches": total_patches }


def plot_heatmap(input_file: str, output_file: str) -> dict:
    # ... (函数内容不变) ...
    t0 = time.time()
    data = torch.load(input_file, map_location="cpu")
    if data.is_cuda: data = data.cpu()
    arr = data.numpy()
    plt.figure(figsize=(12, 8))
    sns.heatmap(arr, cmap=plt.cm.RdBu_r, center=0, annot=True, fmt=".2f", cbar_kws={'label': 'Attention Value'})
    plt.title("Patching Attention Heads", fontsize=16, fontweight='bold')
    plt.xlabel("Head", fontsize=12); plt.ylabel("Layer", fontsize=12)
    plt.tight_layout(); plt.savefig(output_file, dpi=300, bbox_inches='tight'); plt.close()
    elapsed = time.time() - t0
    print(f"[OK] 绘制热力图完成，用时 {elapsed:.3f}s -> {output_file}")
    return { "time": elapsed }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["filter", "collect", "patch", "plot"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timing-output", default="timing.json")
    args = parser.parse_args()
    if args.task == "filter": result = filter_with_gpt2(args.input, args.output)
    elif args.task == "collect": result = get_clean_activations(args.input, args.output)
    elif args.task == "patch": result = activation_patching(args.input, args.output)
    elif args.task == "plot": result = plot_heatmap(args.input, args.output)
    with open(args.timing_output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()