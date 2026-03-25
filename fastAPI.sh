# 1. 注入 LangSmith 的监控环境变量
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY=""
export LANGCHAIN_PROJECT="xv6-agent-prod"

# 2. 启动 FastAPI 后端服务
uvicorn api_server:app --host 0.0.0.0 --port 8081 --reload
#api_server.py