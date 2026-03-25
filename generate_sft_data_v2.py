import json
import random
from collections import defaultdict
from openai import OpenAI
from tqdm import tqdm

# === 配置外部大模型 ===
LLM_API_BASE = "https://api.deepseek.com/v1" 
LLM_API_KEY = "" 
TEACHER_MODEL = "deepseek-chat" 

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)

def paraphrase_question(question):
    """让大模型把一个问题换 4 种不同的问法"""
    prompt = f"""请将以下关于 xv6 内核的问题，改写成 4 种不同的询问方式（保持原意，语气可以多样化：比如新手提问、直接命令、详细追问等）。
请直接输出纯文本，每行一个改写后的问题，不要带任何序号(1. 2.)或前缀！

【原问题】: {question}"""
    
    response = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7 # 温度稍微调高一点，增加问题的多样性
    )
    # 按行切分，并清理可能的空白符
    variants = [line.strip("- *").strip() for line in response.choices[0].message.content.split('\n') if line.strip()]
    return variants[:4] # 确保最多返回4个

def generate_teacher_cot(question, code_snippet, golden_answer, is_negative):
    """生成真实的思考过程 (与之前逻辑一致)"""
    if is_negative:
        prompt = f"你是一个严谨的C语言专家。请写出分析以下代码为何【无法解答】问题的思考过程。字数控制在100字左右。\n\n【问题】: {question}\n【无关代码】:\n{code_snippet}"
    else:
        prompt = f"你是一个严谨的C语言专家。请写出根据以下代码推导出【最终结论】的思考过程。字数控制在100字左右。\n\n【问题】: {question}\n【相关代码】:\n{code_snippet}\n【最终结论】: {golden_answer}"
    
    response = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# ================= 数据构建逻辑 =================
SEED_FILE = "train_seed.json"
OUTPUT_FILE = "xv6_sft_train_data_v4_massive.jsonl"
KB_FILE = "xv6_kb.jsonl"

kb_dict = defaultdict(list)
all_snippets = []
with open(KB_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        all_snippets.append(data['text'])
        kb_dict[data['file']].append(data['text'])

with open(SEED_FILE, "r", encoding="utf-8") as f:
    seeds = json.load(f)

augmented_data = []

print("🚀 启动数据大爆炸引擎！(这可能需要10-20分钟)...")
for item in tqdm(seeds):
    original_q = item['question']
    golden_answer = item['expected_answer']
    target_file = item['source_file']
    
    if target_file not in kb_dict or len(kb_dict[target_file]) == 0:
        continue
        
    correct_code_snippet = kb_dict[target_file][0]
    
    # 🌟 核心提效：将1个问题变成5个问题（原问题 + 4个变体）
    try:
        paraphrased_qs = paraphrase_question(original_q)
        all_questions = [original_q] + paraphrased_qs
    except Exception as e:
        print(f"API请求出错，退回原问题: {e}")
        all_questions = [original_q]

    # 针对每一个问题变体，生成 4 种任务数据！
    for q in all_questions:
        distractors = random.sample(all_snippets, 2)

        # 1. 裁判节点 (正)
        augmented_data.append({
            "instruction": "你是一个极其严格的 C 语言内核代码评审专家。请判断下面的【代码片段】是否足以完美解答【用户问题】。请严格只输出 yes 或 no。",
            "input": f"【用户问题】: {q}\n【代码片段】:\n{correct_code_snippet}",
            "output": "yes"
        })
        
        # 2. 裁判节点 (负)
        augmented_data.append({
            "instruction": "你是一个极其严格的 C 语言内核代码评审专家。请判断下面的【代码片段】是否足以完美解答【用户问题】。请严格只输出 yes 或 no。",
            "input": f"【用户问题】: {q}\n【代码片段】:\n{distractors[0]}",
            "output": "no"
        })

        # 3. 生成节点
        if random.random() < 0.2:
            # RAFT 兜底负样本
            dynamic_cot = generate_teacher_cot(q, distractors[1], golden_answer, is_negative=True)
            augmented_data.append({
                "instruction": "你是一个 xv6 内核专家。请基于提供的代码片段回答问题，必须引用代码中的变量或函数名。",
                "input": f"代码片段:\n{distractors[1]}\n\n问题: {q}",
                "output": f"### 思考过程\n{dynamic_cot}\n### 最终结论\n抱歉，当前检索到的代码不足以回答该问题。"
            })
        else:
            # 真实逻辑推理正样本
            dynamic_cot = generate_teacher_cot(q, correct_code_snippet, golden_answer, is_negative=False)
            augmented_data.append({
                "instruction": "你是一个 xv6 内核专家。请基于提供的代码片段回答问题，必须引用代码中的变量或函数名。",
                "input": f"代码片段:\n{correct_code_snippet}\n\n问题: {q}",
                "output": f"### 思考过程\n{dynamic_cot}\n### 最终结论\n{golden_answer}"
            })

random.shuffle(augmented_data)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in augmented_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ 数据大爆炸完成！共生成 {len(augmented_data)} 条多维泛化数据！")