import json
import time
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 1. 配置区域
# ==========================================
BASE_MODEL_ID = "models/qwen-7b"
DPO_LORA_PATH = "./out/dpo_512.pth"  
TEST_FILE = "eval_dataset.json" # 你的测试集
OUTPUT_FILE = "dpo_compare_results.json"

# ==========================================
# 2. 简易评测算法 (类似 ROUGE-L / 词汇召回率)
# ==========================================
def is_hit(generated_text, expected_text, threshold=0.5):
    """
    判断模型回答是否算作正确 (Hit)。
    工业界标准打法是调用 GPT-4 当裁判，这里用轻量级字符重合度代替。
    如果生成的回答覆盖了标准答案中 50% 以上的核心字符，记为 1 (Hit)。
    """
    # 过滤掉常见标点符号
    ignore_chars = set(" ，。！？；：“”\n\r\t()（）、")
    expected_chars = set(expected_text) - ignore_chars
    
    if not expected_chars:
        return False
        
    hit_count = sum(1 for c in expected_chars if c in generated_text)
    recall_rate = hit_count / len(expected_chars)
    
    # 你可以根据实际情况调整阈值，0.5 是个比较严格的标准
    return recall_rate >= threshold

# ==========================================
# 3. 核心推理逻辑
# ==========================================
def generate_response(model, tokenizer, prompt_text):
    messages = [
        {"role": "system", "content": "你是一个严格的 xv6 操作系统内核专家，回答要求极其精简、准确。"},
        {"role": "user", "content": prompt_text}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=150,
            temperature=0.1,  # 降低随机性以保证评测稳定
            repetition_penalty=1.1
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

# ==========================================
# 4. 自动化跑分流水线
# ==========================================
def run_evaluation():
    if not os.path.exists(DPO_LORA_PATH):
        print(f"❌ 找不到 LoRA 权重文件夹: {DPO_LORA_PATH}")
        return

    print("⏳ 正在加载分词器和 Base 模型...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
    )
    base_model.eval()

    print("🔥 正在挂载 DPO LoRA 补丁...")
    dpo_model = PeftModel.from_pretrained(base_model, DPO_LORA_PATH)
    dpo_model.eval()

    print(f"🚀 开始加载评测集: {TEST_FILE}")
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_questions = len(dataset)
    base_hits = 0
    dpo_hits = 0
    results = []

    print(f"共加载 {total_questions} 道测试题。开始自动化跑分...\n" + "="*50)

    for i, item in enumerate(dataset):
        question = item["question"]
        expected_answer = item["expected_answer"]
        
        print(f"[{i+1}/{total_questions}] Q: {question}")
        
        # 1. 测 Base 模型 (禁用 LoRA)
        start_time = time.time()
        with dpo_model.disable_adapter():
            base_ans = generate_response(base_model, tokenizer, question)
        base_latency = time.time() - start_time
        base_is_hit = is_hit(base_ans, expected_answer)
        if base_is_hit: base_hits += 1

        # 2. 测 DPO 模型 (启用 LoRA)
        start_time = time.time()
        dpo_ans = generate_response(dpo_model, tokenizer, question)
        dpo_latency = time.time() - start_time
        dpo_is_hit = is_hit(dpo_ans, expected_answer)
        if dpo_is_hit: dpo_hits += 1

        # 终端实时状态打印
        print(f"  🔴 Base: {'✅ 命中' if base_is_hit else '❌ 错答'} ({base_latency:.2f}s) | 字数: {len(base_ans)}")
        print(f"  🟢 DPO : {'✅ 命中' if dpo_is_hit else '❌ 错答'} ({dpo_latency:.2f}s) | 字数: {len(dpo_ans)}")
        print("-" * 30)

        # 记录每道题的详细对比
        results.append({
            "id": i + 1,
            "question": question,
            "expected_answer": expected_answer,
            "base_model": {
                "answer": base_ans,
                "is_hit": base_is_hit,
                "latency": round(base_latency, 2),
                "length": len(base_ans)
            },
            "dpo_model": {
                "answer": dpo_ans,
                "is_hit": dpo_is_hit,
                "latency": round(dpo_latency, 2),
                "length": len(dpo_ans)
            }
        })

    # ==========================================
    # 5. 生成最终的酷炫报告
    # ==========================================
    base_rate = (base_hits / total_questions) * 100
    dpo_rate = (dpo_hits / total_questions) * 100
    diff_rate = dpo_rate - base_rate
    diff_count = dpo_hits - base_hits

    # 计算平均字数 (DPO 的字数通常会显著减少，因为戒掉了废话)
    avg_base_len = sum(r["base_model"]["length"] for r in results) / total_questions
    avg_dpo_len = sum(r["dpo_model"]["length"] for r in results) / total_questions

    print("\n" + "="*50)
    print("🎉 DPO 偏好对齐自动化评测完成！")
    print("="*50)
    print(f"🎯 语义命中率 (Semantic Hit Rate):")
    print(f"  - Baseline 模型:  {base_rate:.2f}% ({base_hits}/{total_questions})")
    print(f"  - DPO 微调模型:   {dpo_rate:.2f}% ({dpo_hits}/{total_questions})")
    
    if diff_rate > 0:
        print(f"  📈 结论: DPO 后回答准确率 绝对提升了 {diff_rate:.2f}% (多对 {diff_count} 题)！")
    elif diff_rate < 0:
        print(f"  📉 结论: DPO 后发生退化，下降了 {abs(diff_rate):.2f}%。")
    else:
        print("  ➖ 结论: 准确率持平。")

    print(f"\n📏 平均输出长度 (越短说明废话越少):")
    print(f"  - Baseline 模型:  {avg_base_len:.1f} 字符")
    print(f"  - DPO 微调模型:   {avg_dpo_len:.1f} 字符")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 详细对比结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_evaluation()