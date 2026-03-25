import os
import sys
import json
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"

# ================= 配置区域 =================
# 1. 知识库文件路径
KB_FILE = "xv6_kb.jsonl"      # 你的 xv6 知识库
INDEX_FILE = "xv6.index"      # 你的 FAISS 索引

# 2. Embedding 模型配置
# 如果你已经下载到了本地文件夹，改成文件夹路径，比如 "./bge-m3-local"
# 如果没下载，保持 "BAAI/bge-m3"，它会尝试联网
EMBED_MODEL_PATH = "BAAI/bge-m3" 
# 【关键】强制使用 CPU 避免与 vLLM 抢显存导致死锁
DEVICE_EMBED = 'cpu' 

# 3. vLLM 配置
LLM_API_BASE = "http://localhost:8000/v1"
LLM_API_KEY = "EMPTY"

# ===========================================

# 终端颜色代码，让界面更好看
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class RAGSystem:
    def __init__(self):
        print(f"{Colors.HEADER}正在初始化 RAG 系统...{Colors.ENDC}")
        
        # 1. 初始化 vLLM 客户端并自动获取模型名
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
        self.llm_model_name = self._get_vllm_model_name()
        print(f"{Colors.GREEN}✔ 成功连接 vLLM，模型名称: {self.llm_model_name}{Colors.ENDC}")

        # 2. 加载 Embedding 模型
        print(f"正在加载 Embedding 模型 ({DEVICE_EMBED})...")
        if os.path.exists("./bge-m3-local"):
            # 优先检查是否有本地离线模型
            print(f"发现本地模型，从 ./bge-m3-local 加载...")
            self.embed_model = SentenceTransformer("./bge-m3-local", device=DEVICE_EMBED)
        else:
            self.embed_model = SentenceTransformer(EMBED_MODEL_PATH, device=DEVICE_EMBED)
        print(f"{Colors.GREEN}✔ Embedding 模型加载完毕{Colors.ENDC}")

        # 3. 加载 FAISS 索引
        if not os.path.exists(INDEX_FILE) or not os.path.exists(KB_FILE):
            print(f"{Colors.RED}❌ 错误：找不到 {INDEX_FILE} 或 {KB_FILE}。请先运行 build_xv6_kb.py 和 build_index.py{Colors.ENDC}")
            sys.exit(1)
            
        self.index = faiss.read_index(INDEX_FILE)
        print(f"{Colors.GREEN}✔ FAISS 索引加载完毕 (包含 {self.index.ntotal} 条向量){Colors.ENDC}")

        # 4. 加载文本数据到内存 (ID -> Text 映射)
        self.db = {}
        with open(KB_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.db[data['id']] = data
        print(f"{Colors.GREEN}✔ 知识库文本加载完毕{Colors.ENDC}")

    def _get_vllm_model_name(self):
        """自动查询 vLLM 当前加载的模型名称，防止配置错误"""
        try:
            models = self.client.models.list()
            return models.data[0].id
        except Exception as e:
            print(f"{Colors.RED}❌ 无法连接 vLLM 服务: {e}{Colors.ENDC}")
            print(f"{Colors.YELLOW}提示: 请检查 'nohup python -m vllm...' 是否正在后台运行。{Colors.ENDC}")
            sys.exit(1)

    def retrieve(self, query, top_k=5):
        """检索相关代码片段"""
        # 编码查询
        query_vec = self.embed_model.encode([query], normalize_embeddings=True)
        # 搜索
        D, I = self.index.search(query_vec, top_k)
        
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx in self.db:
                item = self.db[idx]
                results.append({
                    "file": item.get('file', 'Unknown'),
                    "text": item['text'],
                    "score": float(score)
                })
        return results

    def chat(self):
        print(f"\n{Colors.BOLD}🚀 系统就绪！你可以问关于 xv6 源码的问题了 (输入 'exit' 退出){Colors.ENDC}")
        
        while True:
            try:
                query = input(f"\n{Colors.BLUE}User > {Colors.ENDC}")
                if query.strip().lower() in ['exit', 'quit']:
                    print("Bye!")
                    break
                if not query.strip():
                    continue

                # --- 1. 检索阶段 ---
                print(f"{Colors.YELLOW}🔍 正在检索代码...{Colors.ENDC}")
                context_results = self.retrieve(query, top_k=4) # 代码通常比较长，Top 4 够了

                # 打印引用源 (Debug info)
                print(f"--- 参考了以下文件 ---")
                for res in context_results:
                    print(f"📄 {res['file']} (相似度: {res['score']:.3f})")
                print(f"----------------------")

                # --- 2. 生成阶段 ---
                # 拼接 Prompt
                context_str = "\n\n".join([f"// Source File: {r['file']}\n{r['text']}" for r in context_results])
                
                system_prompt = """你是一个操作系统内核专家，精通 xv6 源码 (x86版)。
请根据提供的【代码片段】回答用户问题。
要求：
1. 回答必须基于提供的代码逻辑。
2. 引用具体的函数名、变量名。
3. 如果代码中没体现，请直说。
"""
                user_prompt = f"【参考代码片段】:\n{context_str}\n\n【用户问题】: {query}"

                print(f"{Colors.YELLOW}🤖 正在生成回答...{Colors.ENDC}")
                print(f"{Colors.GREEN}", end="") # 绿色输出回答

                # 流式调用
                stream = self.client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=2048,
                    temperature=0.7,
                    stream=True
                )

                full_response = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        print(text, end="", flush=True)
                        full_response += text
                
                print(f"{Colors.ENDC}") # 恢复颜色

            except KeyboardInterrupt:
                print("\n操作取消。")
                break
            except Exception as e:
                print(f"\n{Colors.RED}❌ 发生错误: {e}{Colors.ENDC}")

if __name__ == "__main__":
    # 简单的入口保护
    try:
        rag = RAGSystem()
        rag.chat()
    except Exception as e:
        print(f"系统启动失败: {e}")