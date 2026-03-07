import faiss
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --- 配置 ---
KB_FILE = "cpp_knowledge_base.jsonl"
INDEX_FILE = "cpp_primer.index"
DEVICE_EMBED = 'cpu' # Embedding 模型在卡 1
LLM_API_URL = "http://localhost:8000/v1" # vLLM 默认地址
LLM_MODEL_NAME = "./models/qwen-7b" # 这里填 vLLM 启动时的模型名（或路径）

# --- 1. 初始化资源 ---
print("正在初始化 RAG 系统...")

# A. 加载 Embedding 模型
print("加载 BGE-M3...")
embed_model = SentenceTransformer("BAAI/bge-m3", device=DEVICE_EMBED)

# B. 加载 FAISS 索引
print("加载索引...")
index = faiss.read_index(INDEX_FILE)

# C. 加载文本数据 (做成 id -> text 的映射，方便检索)
# 内存优化提示：如果文件几个G，这里可以用 mmap 或数据库，但一本书直接读内存没事
print("加载文本数据库...")
db = {}
with open(KB_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        db[data['id']] = data # 存下整条数据

# D. 初始化 LLM 客户端
client = OpenAI(
    api_key="EMPTY", # vLLM 本地不需要 key
    base_url=LLM_API_URL,
)

# --- 2. 核心函数 ---

def retrieve(query, top_k=3):
    """检索最相关的文档块"""
    # 1. Query 向量化
    query_vec = embed_model.encode([query], normalize_embeddings=True)
    
    # 2. FAISS 搜索
    # D 是距离(相似度)，I 是索引 ID
    D, I = index.search(query_vec, top_k)
    
    results = []
    for idx, score in zip(I[0], D[0]):
        if idx in db:
            doc = db[idx]
            results.append({
                "text": doc['text'],
                "page": doc.get('page', 'Unknown'),
                "score": float(score)
            })
    return results

def generate_answer(query, context_chunks):
    """组装 Prompt 并调用 LLM"""
    
    # 拼接上下文
    context_str = "\n\n---\n\n".join(
        [f"[Page {c['page']}] {c['text']}" for c in context_chunks]
    )
    
    # 编写 Prompt (这是 RAG 的灵魂)
    prompt = f"""你是一个精通 C++ 的智能助手。请基于以下参考文档回答用户的问题。
如果参考文档中没有答案，请根据你的知识回答，并说明文档中未提及。

参考文档：
{context_str}

用户问题：{query}
"""

    # 调用 vLLM
    response = client.chat.completions.create(
        model=LLM_MODEL_NAME, # 注意这里要填 vLLM 实际加载的模型名
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1024
    )
    
    return response.choices[0].message.content

# --- 3. 交互循环 ---
def main():
    print("\n✅ RAG 系统已就绪！(输入 'exit' 退出)")
    
    while True:
        query = input("\n请输入关于 C++ 的问题: ")
        if query.strip().lower() == 'exit':
            break
            
        if not query.strip():
            continue
            
        print("🔍 正在检索知识库...")
        results = retrieve(query, top_k=3)
        
        # 打印检索到的片段 (Debug 模式，面试演示时很有用)
        print(f"--- 检索到 {len(results)} 条相关内容 ---")
        for i, res in enumerate(results):
            print(f"[{i+1}] Page {res['page']} (相似度: {res['score']:.4f}): {res['text'][:50]}...")
            
        print("🧠 正在思考...")
        answer = generate_answer(query, results)
        
        print("\n" + "="*50)
        print(f"🤖 回答:\n{answer}")
        print("="*50)

if __name__ == "__main__":
    main()