import json
import os

BASELINE_FILE = "eval_results_baseline.json"
SFT_FILE = "eval_results_sft.json"

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # 转成以 ID 为 key 的字典，方便对比
        return {item["id"]: item for item in data}

def analyze():
    print("🚀 正在加载评测数据...")
    base_data = load_data(BASELINE_FILE)
    sft_data = load_data(SFT_FILE)
    
    if not base_data or not sft_data:
        return

    common_ids = set(base_data.keys()).intersection(set(sft_data.keys()))
    total = len(common_ids)
    
    base_hits = sum(1 for qid in common_ids if base_data[qid]["is_hit"])
    sft_hits = sum(1 for qid in common_ids if sft_data[qid]["is_hit"])
    
    base_latency = sum(base_data[qid]["latency_seconds"] for qid in common_ids) / total
    sft_latency = sum(sft_data[qid]["latency_seconds"] for qid in common_ids) / total

    # 统计修复和退化的 Case
    fixed_cases = []      # 原来错，现在对
    regressed_cases = []  # 原来对，现在错
    
    for qid in common_ids:
        base_hit = base_data[qid]["is_hit"]
        sft_hit = sft_data[qid]["is_hit"]
        
        if not base_hit and sft_hit:
            fixed_cases.append(qid)
        elif base_hit and not sft_hit:
            regressed_cases.append(qid)

    # 打印酷炫的报告
    print("\n" + "="*50)
    print("📊 xv6 Agent 评测对比报告 (Baseline vs SFT)")
    print("="*50)
    print(f"题目总数: {total}")
    
    print("\n🎯 检索命中率 (Hit Rate):")
    print(f"  - Baseline:  {base_hits}/{total} ({(base_hits/total)*100:.1f}%)")
    print(f"  - SFT 模型:  {sft_hits}/{total} ({(sft_hits/total)*100:.1f}%)")
    
    hit_diff = sft_hits - base_hits
    if hit_diff > 0:
        print(f"  📈 结论: SFT 后检索命中率提升了 {hit_diff} 题！")
    elif hit_diff < 0:
        print(f"  📉 结论: SFT 后检索命中率下降了 {abs(hit_diff)} 题 (发生退化)。")
    else:
        print("  ➖ 结论: 检索命中率持平。")

    print("\n⏱️ 平均响应延迟 (Average Latency):")
    print(f"  - Baseline:  {base_latency:.2f} 秒")
    print(f"  - SFT 模型:  {sft_latency:.2f} 秒")

    print("\n🛠️ Bad Case 深度追踪:")
    if fixed_cases:
        print("  ✅ [成功修复] 的题目 ID:")
        for qid in fixed_cases:
            print(f"    -> ID {qid}: {sft_data[qid]['question']}")
            print(f"       (SFT 检索结果: {sft_data[qid]['retrieved_files']})")
    else:
        print("  ✅ [成功修复]: 无。")
        
    if regressed_cases:
        print("\n  ❌ [严重退化] 的题目 ID (原来对，现在错):")
        for qid in regressed_cases:
            print(f"    -> ID {qid}: {sft_data[qid]['question']}")
            print(f"       (SFT 检索结果: {sft_data[qid]['retrieved_files']})")
    else:
        print("\n  ❌ [严重退化]: 零退化！完美兼容原有能力！")
        
    print("="*50 + "\n")

if __name__ == "__main__":
    analyze()