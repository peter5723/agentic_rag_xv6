import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# 屏蔽代理，防止 vLLM 本地调用失败
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"

# 导入你已经写好的 LangGraph Agent
from agent_graph import app as agent_app

# --- 1. 定义数据结构 ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    thought_process: List[str]  # 用于记录 Agent 的思考和流转节点

# --- 2. 初始化 FastAPI 应用 ---
app = FastAPI(
    title="xv6 OS Agent API",
    description="基于 Qwen-7B 与 LangGraph 的内核源码问答接口",
    version="1.0.0"
)

# --- 3. 核心 API 路由 ---
@app.post("/chat", response_model=QueryResponse)
async def chat_with_agent(request: QueryRequest):
    inputs = {
        "question": request.question, 
        "current_query": request.question, 
        "iteration": 0,
        "is_relevant": "unknown",
        "context": [],
        "answer": ""
    }
    
    thought_process = []
    final_answer = ""
    
    try:
        # 遍历 LangGraph 的执行步骤，收集思考路径
        for output in agent_app.stream(inputs):
            for node_name, state_value in output.items():
                
                # 记录 Agent 走过了哪些节点
                if node_name == "retrieve":
                    query = state_value.get('current_query', '')
                    thought_process.append(f"[检索] 搜索关键词: {query}")
                elif node_name == "grade":
                    grade = state_value.get('is_relevant', 'no')
                    thought_process.append(f"[裁判] 评估结果: {grade.upper()}")
                elif node_name == "rewrite":
                    new_query = state_value.get('current_query', '')
                    thought_process.append(f"[重写] 生成新关键词: {new_query}")
                elif node_name == "generate":
                    thought_process.append("[生成] 开始生成最终回答")
                    final_answer = state_value.get("answer", "")
                    
        # 返回标准的 JSON 格式
        return QueryResponse(
            question=request.question,
            answer=final_answer,
            thought_process=thought_process
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 运行出错: {str(e)}")

# 健康检查接口
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "xv6 Agent is ready."}