import json
import random
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
DEEPSEEK_API_KEY = ""
BASE_URL = "https://api.deepseek.com" # DeepSeek 官方接口
MODEL_NAME = "deepseek-reasoner"          # 或者用 deepseek-reasoner 跑深度思考

KB_FILE = "xv6_kb.jsonl"
OUTPUT_FILE = "dpo_dataset_deepseek.jsonl"
TARGET_COUNT = 250
# ============================================

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

print("📚 正在加载 xv6 (x86) 知识库...")
kb_snippets = []
with open(KB_FILE, "r", encoding="utf-8") as f:
    for line in f:
        kb_snippets.append(json.loads(line)['text'])

def generate_dpo_pair(snippet):
    """利用 DeepSeek 针对 x86 源码片段自动生成 DPO 三元组"""
    
    # 这里的 System Prompt 是灵魂，重兵防守架构幻觉
    system_prompt = """你是一个世界顶级的操作系统内核安全专家，专门研究 MIT xv6 (经典 x86 架构版本)。
你需要阅读提供的【x86 架构 xv6 源码片段】，构造一个用于大模型偏好对齐（DPO）训练的 JSON 数据。

【极其严厉的架构警告】：
1. 绝对禁止使用任何 RISC-V 概念（如 satp, sstatus, ecall）！
2. 绝对禁止混入现代 Linux 概念（如 CFS, RCU, Swap 分区, cgroups）！
3. 必须严格围绕 x86 硬件机制（如 CR3, TSS, IDT, GDT, int 指令）。

【输出格式要求】：
请按以下 JSON 格式输出，并且必须将尖括号 <...> 中的说明文字替换为你真正生成的、基于代码的具体内容！绝对不能直接输出尖括号里的原话！

{
  "prompt": "<请在这里填写你根据代码生成的硬核问题>",
  "chosen": "<请在这里填写基于 x86 机制的正确原理回答>",
  "rejected": "<请在这里填写包含 Linux 或 RISC-V 幻觉的错误回答>"
}"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请针对以下代码片段构造 DPO 数据：\n{snippet[:1000]}"}
            ],
            temperature=0.7, # 稍微给点温度，让它能发散出不同的 Rejected 幻觉
            response_format={"type": "json_object"} # 强制 JSON 输出
        )
        
        result_text = response.choices[0].message.content.strip()
        result = json.loads(result_text)
        
        if all(k in result for k in ["prompt", "chosen", "rejected"]):
            if "基于这段代码" in result["prompt"] or "<请在这里填写" in result["prompt"]:
                print("⚠️ 抓到一个偷懒的复读机，数据已丢弃。")
                return None
            return result
        return None
    except Exception as e:
        print(f"⚠️ API 调用或解析失败: {e}")
        return None

print(f"🚀 开始调用 DeepSeek 自动生成 {TARGET_COUNT} 条 DPO 数据...")
dpo_dataset = []

with tqdm(total=TARGET_COUNT) as pbar:
    while len(dpo_dataset) < TARGET_COUNT:
        snippet = random.choice(kb_snippets)
        dpo_pair = generate_dpo_pair(snippet)
        
        if dpo_pair:
            dpo_dataset.append(dpo_pair)
            pbar.update(1)
            
            # 实时追加保存，防止网络中断导致心血白费
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(dpo_pair, ensure_ascii=False) + "\n")

print(f"🎉 DeepSeek 生成完毕！数据已保存至 {OUTPUT_FILE}")