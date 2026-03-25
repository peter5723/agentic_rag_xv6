import json
import random
import os
from openai import OpenAI
from tqdm import tqdm

# --- 配置区域 ---
LLM_API_BASE = "http://localhost:8000/v1"
LLM_API_KEY = "EMPTY"
SEED_FILE = "train_seed.json"
OUTPUT_FILE = "xv6_sft_train_data.jsonl"
KB_FILE = "xv6_kb.jsonl" # 用于抽取负样本

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)

# 1. 加载种子数据
with open(SEED_FILE, "r", encoding="utf-8") as f:
    seeds = json.load(f)

# 2. 加载知识库以便抽取“干扰项” (RAFT 策略)
all_code_snippets = []
with open(KB_FILE, "r", encoding="utf-8") as f:
    for line in f:
        all_code_snippets.append(json.loads(line)['text'])

def get_llm_response(prompt):
    """简单封装调用 vLLM"""
    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

print(f"🚀 开始数据增强，目标：将 {len(seeds)} 条种子扩展至约 2000 条...")

augmented_data = []

for item in tqdm(seeds):
    question = item['question']
    golden_answer = item['expected_answer']
    correct_file = item['source_file']
    
    # --- 策略 A: 指令多样化 (Rewriting) ---
    # 1. 修改 Prompt：明确禁止模型输出序号
    rewrite_prompt = f"请将以下关于 xv6 内核的问题改写成 5 种不同的询问方式，保持原意不变，直接输出改写后的问题，每行一个。注意：必须纯文本输出，绝对不要带任何序号（如 1. 2. 3.）、前缀或破折号：\n{question}"
    variants = get_llm_response(rewrite_prompt).split('\n')
    
    # 2. 增加代码兜底：使用 lstrip 强制剔除可能残留的数字、点和空格
    all_questions = [question] + [v.lstrip('0123456789.、- ').strip() for v in variants if v.strip()]
    for q in all_questions:
        # --- 策略 B: 构造 RAFT 负样本 (Distractors) ---
        # 随机抽取 2 个不相关的代码片段作为干扰
        distractors = random.sample(all_code_snippets, 2)
        
        # --- 策略 C: 合成思维链 (Chain of Thought) ---
        # 这里模拟一个高质量的回答过程
        cot_prompt = f"""你是一个内核专家。请根据以下代码片段，写出回答“{q}”的推理过程。
要求：先分析代码逻辑，最后给出结论。
代码片段：
{distractors[0]} (干扰项)
{distractors[1]} (干扰项)
(核心参考): {golden_answer}
"""
        thought_process = get_llm_response(cot_prompt)

        # 组装成 SFT 格式 (Alpaca 或 ShareGPT 格式)
        augmented_data.append({
            "instruction": q,
            "input": f"参考代码文件: {correct_file}",
            "output": f"### 思考过程\n{thought_process}\n\n### 最终结论\n{golden_answer}"
        })

# 保存结果
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in augmented_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ 增强完成！共生成 {len(augmented_data)} 条训练数据，保存至 {OUTPUT_FILE}")