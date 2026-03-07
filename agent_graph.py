import os
import json
import torch
from typing import List, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import faiss
from openai import OpenAI

# ================= 1. 配置区域 =================
LLM_API_BASE = "http://localhost:8000/v1"
LLM_API_KEY = "EMPTY"
KB_FILE = "xv6_kb.jsonl"
INDEX_FILE = "xv6.index"

# 屏蔽代理
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"

# ================= 2. 初始化组件 =================
print("🚀 正在初始化 Agent 组件...")

# --- 2.1 自动获取 vLLM 模型名称 ---
temp_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
try:
    models_list = temp_client.models.list()
    valid_model_name = models_list.data[0].id
    print(f"✅ 连接 vLLM 成功: {valid_model_name}")
except Exception as e:
    print(f"❌ 无法连接 vLLM，请检查服务。错误: {e}")
    exit(1)

llm = ChatOpenAI(model="xv6-expert", openai_api_key=LLM_API_KEY, openai_api_base=LLM_API_BASE, temperature=0)

# --- 2.2 加载 FAISS & BGE-M3 (粗排, 强制 CPU) ---
print("正在加载 BGE-M3 粗排模型...")
script_dir = os.path.dirname(os.path.abspath(__file__))
bge_path = os.path.join(script_dir, "bge-m3-local")
embed_model = SentenceTransformer(bge_path, device='cpu')

index = faiss.read_index(INDEX_FILE)
db = {}
with open(KB_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        db[data['id']] = data

# --- 2.3 加载 Reranker (精排, 放到 cuda:1) ---
print("正在加载 Reranker 精排模型 (至 cuda:1)...")
rerank_path = os.path.join(script_dir, "bge-reranker-local")
rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_path)
rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_path).to('cuda:1')
rerank_model.eval()
print("✅ 所有模型加载完毕！")

# ================= 3. 核心检索函数 (双路检索) =================
def retrieve_docs(query, top_k=3, fetch_k=20):
    """粗排召回 Top-20，精排提取 Top-3"""
    query_vec = embed_model.encode([query], normalize_embeddings=True)
    D, I = index.search(query_vec, fetch_k)
    
    rough_results = [db[idx] for idx in I[0] if idx in db]
    if not rough_results: return []

    pairs = [[query, doc['text']] for doc in rough_results]
    with torch.no_grad():
        inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to('cuda:1')
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1,).float()
        
    scored_results = sorted(zip(rough_results, scores.cpu().numpy()), key=lambda x: x[1], reverse=True)
    return [item[0] for item in scored_results[:top_k]]

# ================= 4. LangGraph 状态机定义 =================

class AgentState(TypedDict):
    question: str                # 用户原始问题
    current_query: str           # 当前用于搜索的关键词 (可能会被改写)
    context: List[str]           # 检索到的代码片段
    iteration: int               # 循环次数 (防死循环)
    is_relevant: str             # 裁判打分: "yes" 或 "no"
    answer: str                  # 最终回答

# --- 节点 1: 检索节点 ---
def retrieve_node(state: AgentState):
    query = state.get("current_query", state["question"])
    print(f"\n[🔍 Retrieve] 正在检索关键词: {query}")
    docs = retrieve_docs(query, top_k=3)
    context_text = [f"// File: {d['file']}\n{d['text']}" for d in docs]
    return {"context": context_text, "current_query": query, "iteration": state.get("iteration", 0)}

# --- 节点 2: 裁判节点 (核心激活！) ---
# --- 节点 2: 裁判节点 ---
def grade_documents_node(state: AgentState):
    print("[⚖️ Grade] 裁判 LLM 正在评估代码是否包含答案...")
    question = state["question"]
    context_str = "\n\n".join(state["context"])
    
    # 严苛的否定条件 Prompt
    prompt = f"""你是一个极其严格的 C 语言内核代码评审专家。
请判断下面的【代码片段】是否足以完美解答【用户问题】。
要求：
1. 如果代码只包含表面调用，没有底层实现逻辑，请输出 "no"。
2. 如果代码完全无关，请输出 "no"。
3. 如果你不确定，请输出 "no"。
只有当代码真正包含了回答问题所需的核心逻辑时，才输出 "yes"。
请严格只输出 "yes" 或 "no"。

【用户问题】: {question}
【代码片段】:
{context_str}
"""
    response = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    grade = "yes" if "yes" in response else "no"
    
    print(f"  -> 评估结果: {grade.upper()}")
    
    # 【就是这里！】之前漏掉了这行返回，导致状态变成了 None
    return {"is_relevant": grade}

# --- 节点 3: 问题重写节点 ---
def rewrite_query_node(state: AgentState):
    print("[🔄 Rewrite] 信息不足，LLM 正在重新思考搜索关键词...")
    question = state["question"]
    current_query = state["current_query"]
    iteration = state["iteration"] + 1
    
    prompt = f"""你是一个操作系统专家。为了回答 "{question}"，我们需要在 xv6 C语言源码中检索。
请根据你的操作系统知识，提取出 1 到 2 个最关键的 C语言底层实体名词（例如具体的宏定义、寄存器名、或是通用底层机制，如 'LRU', 'page table', 'swtch'）。
请千万不要编造不存在的函数名！直接输出这几个关键词，用空格隔开。"""
    
    new_keywords = llm.invoke([HumanMessage(content=prompt)]).content.strip().replace('"', '').replace("'", "")
    
    # 【关键修复】将大模型提取的关键词，拼接到原问题后面！
    # 这样就算大模型猜错了，原问题的语义向量依然能在 FAISS 中兜底！
    combined_query = f"{question} {new_keywords}"
    print(f"  -> 决定尝试混合关键词: {combined_query}")
    
    return {"current_query": combined_query, "iteration": iteration}

# --- 节点 4: 最终生成节点 ---
def generate_node(state: AgentState):
    print("[🧠 Generate] 正在基于最终确定的代码生成回答...")
    context_str = "\n\n".join(state["context"])
    messages = [
        SystemMessage(content="你是一个 xv6 内核专家。请基于提供的代码片段回答问题，必须引用代码中的变量或函数名。"),
        HumanMessage(content=f"代码片段:\n{context_str}\n\n问题: {state['question']}")
    ]
    response = llm.invoke(messages)
    return {"answer": response.content}

# --- 条件路由判断 ---
# --- 条件路由判断 ---
def decide_to_generate(state: AgentState) -> Literal["generate", "rewrite"]:
    # 【关键修复】：使用 .get() 安全获取，即使 key 不存在也不会报错
    is_relevant = state.get("is_relevant", "no")
    iteration = state.get("iteration", 0)
    
    if is_relevant == "yes":
        return "generate"  # 找对了，直接回答
    elif iteration >= 2:
        print("  -> 已达到最大重试次数 (2次)，强制停止搜索并生成回答。")
        return "generate"  # 找不到算了，尽力答
    else:
        return "rewrite"   # 没找对，且还有机会，去重写关键词

# ================= 5. 构建与编译图 =================
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("rewrite", rewrite_query_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
# 关键：根据裁判的打分决定下一步去哪里
workflow.add_conditional_edges("grade", decide_to_generate, {"generate": "generate", "rewrite": "rewrite"})
workflow.add_edge("rewrite", "retrieve") # 重写后回到检索
workflow.add_edge("generate", END)

app = workflow.compile()

# ================= 测试执行 =================
if __name__ == "__main__":
    # 我们用那道“跨文件”错题来测试它的多跳能力
    test_q = "fork() 系统调用是如何复制父进程的内存空间的？调用的底层函数是什么？"
    
    print(f"\n=============================================")
    print(f"用户问题: {test_q}")
    print(f"=============================================")
    
    inputs = {"question": test_q, "current_query": test_q, "iteration": 0}
    
    for output in app.stream(inputs):
        pass # 所有的 print 都在节点函数里完成了
        
    print(f"\n最终回答:\n{output['generate']['answer']}")