"""
对比三种模式的计时报告生成汇总表格
"""
import json
import sys
from typing import Dict, List


def load_timing(path: str) -> Dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"警告: 未找到 {path}")
        return {}


def format_time(t: float) -> str:
    if t < 1:
        return f"{t*1000:.1f}ms"
    elif t < 60:
        return f"{t:.2f}s"
    else:
        m = int(t // 60)
        s = t % 60
        return f"{m}m{s:.1f}s"


def compare_reports(local_all: str = "timing_local_all.json",
                   remote_all: str = "timing_remote_all.json",
                   hybrid: str = "timing_hybrid.json"):
    
    local_data = load_timing(local_all)
    remote_data = load_timing(remote_all)
    hybrid_data = load_timing(hybrid)
    
    if not any([local_data, remote_data, hybrid_data]):
        print("错误: 未找到任何计时报告文件")
        print("请先运行三种模式生成报告：")
        print("  python ioi_orchestrator.py --config configs/local_all.json")
        print("  python ioi_orchestrator.py --config configs/remote_all.json")
        print("  python ioi_orchestrator.py --config configs/hybrid.json")
        return
    
    # 定义要对比的关键指标（与IOI.ipynb一致）
    metrics = [
        ("filter_gpt2_time", "GPT-2样本筛选"),
        ("collect_activations_time", "缓存激活值"),
        ("patch_activations_time", "修补激活值"),
        ("plot_heatmap_time", "绘制热力图"),
    ]
    
    comm_metrics = [
        ("filter_gpt2_upload_time", "筛选-上传"),
        ("filter_gpt2_download_time", "筛选-下载"),
        ("collect_activations_upload_time", "缓存-上传"),
        ("collect_activations_download_time", "缓存-下载"),
        ("patch_activations_upload_time", "修补-上传"),
        ("patch_activations_download_time", "修补-下载"),
        ("plot_heatmap_upload_time", "绘图-上传"),
        ("plot_heatmap_download_time", "绘图-下载"),
    ]
    
    print("=" * 100)
    print("IOI 项目本地/云端/混合模式计时对比")
    print("=" * 100)
    print()
    
    # 1. 主要环节对比
    print("【计算环节耗时对比】")
    print("-" * 100)
    print(f"{'环节':<20} {'全本地':>15} {'全云端':>15} {'混合模式':>15} {'最优':>10}")
    print("-" * 100)
    
    for key, name in metrics:
        local_val = local_data.get(key, None)
        remote_val = remote_data.get(key, None)
        hybrid_val = hybrid_data.get(key, None)
        
        vals = {}
        if local_val is not None:
            vals['全本地'] = local_val
        if remote_val is not None:
            vals['全云端'] = remote_val
        if hybrid_val is not None:
            vals['混合'] = hybrid_val
        
        best = min(vals.items(), key=lambda x: x[1])[0] if vals else "-"
        
        local_str = format_time(local_val) if local_val else "-"
        remote_str = format_time(remote_val) if remote_val else "-"
        hybrid_str = format_time(hybrid_val) if hybrid_val else "-"
        
        print(f"{name:<20} {local_str:>15} {remote_str:>15} {hybrid_str:>15} {best:>10}")
    
    print("-" * 100)
    print()
    
    # 2. 通信开销对比（仅远端和混合模式有）
    print("【通信开销对比】(仅远端和混合模式)")
    print("-" * 100)
    print(f"{'通信环节':<30} {'全云端':>15} {'混合模式':>15}")
    print("-" * 100)
    
    for key, name in comm_metrics:
        remote_val = remote_data.get(key, None)
        hybrid_val = hybrid_data.get(key, None)
        
        remote_str = format_time(remote_val) if remote_val else "-"
        hybrid_str = format_time(hybrid_val) if hybrid_val else "-"
        
        if remote_val or hybrid_val:
            print(f"{name:<30} {remote_str:>15} {hybrid_str:>15}")
    
    print("-" * 100)
    print()
    
    # 3. 总耗时对比
    print("【总耗时对比】")
    print("-" * 100)
    
    def calc_total(data: Dict) -> float:
        total = data.get("local_prepare_time", 0)
        for key, _ in metrics:
            total_key = key.replace("_time", "_total_time")
            if total_key in data:
                total += data[total_key]
            elif key in data:
                total += data[key]
        return total
    
    local_total = calc_total(local_data) if local_data else None
    remote_total = calc_total(remote_data) if remote_data else None
    hybrid_total = calc_total(hybrid_data) if hybrid_data else None
    
    print(f"{'全本地总耗时:':<20} {format_time(local_total) if local_total else '-'}")
    print(f"{'全云端总耗时:':<20} {format_time(remote_total) if remote_total else '-'}")
    print(f"{'混合模式总耗时:':<20} {format_time(hybrid_total) if hybrid_total else '-'}")
    
    totals = {}
    if local_total: totals['全本地'] = local_total
    if remote_total: totals['全云端'] = remote_total
    if hybrid_total: totals['混合'] = hybrid_total
    
    if totals:
        best_mode = min(totals.items(), key=lambda x: x[1])
        print(f"\n最优模式: {best_mode[0]} ({format_time(best_mode[1])})")
        
        if '全本地' in totals and '混合' in totals:
            speedup = (totals['全本地'] - totals['混合']) / totals['全本地'] * 100
            print(f"混合模式相比全本地加速: {speedup:+.1f}%")
    
    print("-" * 100)
    print()
    
    # 4. 执行位置汇总
    print("【各环节执行位置】")
    print("-" * 100)
    for data, mode_name in [(local_data, "全本地"), (remote_data, "全云端"), (hybrid_data, "混合模式")]:
        if not data:
            continue
        print(f"\n{mode_name}:")
        for key, name in metrics:
            loc_key = key.replace("_time", "_location")
            loc = data.get(loc_key, "local" if mode_name == "全本地" else "remote" if mode_name == "全云端" else "-")
            loc_cn = "本地" if loc == "local" else "云端" if loc == "remote" else "-"
            print(f"  {name:<20} -> {loc_cn}")
    
    print("=" * 100)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        local_path = sys.argv[1] if len(sys.argv) > 1 else "timing_local_all.json"
        remote_path = sys.argv[2] if len(sys.argv) > 2 else "timing_remote_all.json"
        hybrid_path = sys.argv[3] if len(sys.argv) > 3 else "timing_hybrid.json"
        compare_reports(local_path, remote_path, hybrid_path)
    else:
        compare_reports()

