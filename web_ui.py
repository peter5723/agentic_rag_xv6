import streamlit as st
import requests
import time

# --- 1. 配置参数 ---
# 这是你刚才启动的 FastAPI 后端地址
API_URL = "http://localhost:8080/chat"

st.set_page_config(page_title="xv6 OS Agent", page_icon="🐧", layout="centered")
st.title("🐧 xv6 内核源码 Agent")
st.caption("🚀 [前后端分离架构] 前端: Streamlit | 后端: FastAPI + LangGraph | 模型: vLLM-LoRA")

# --- 2. 初始化历史对话 ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是 xv6 代码助手。现在我已经全面升级为**微服务架构**！我的大脑运行在独立的 FastAPI 后端上。请问有什么内核源码问题我可以帮到你？"}
    ]

# 渲染历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 3. 处理用户提问 ---
if prompt := st.chat_input("向 xv6 Agent 提问..."):
    # 立即在界面上显示用户问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 准备显示 Agent 的回答
    with st.chat_message("assistant"):
        final_answer = ""
        
        # 使用 st.status 制作漂亮的折叠等待框
        with st.status("🤖 发送请求至 FastAPI 后端...", expanded=True) as status:
            try:
                # 【核心逻辑】：通过 HTTP POST 发送请求给后端
                response = requests.post(API_URL, json={"question": prompt}, timeout=120)
                response.raise_for_status()  # 检查是否报 404/500 等错误
                
                # 解析后端返回的 JSON 数据
                data = response.json()
                thought_process = data.get("thought_process", [])
                final_answer = data.get("answer", "后端没有返回答案。")
                
                # 在进度框里展示后端传回来的“思考路径”
                for step in thought_process:
                    st.write(step)
                    time.sleep(0.3)  # 加上微小的延迟，营造“实时流转”的高级视觉感
                    
                status.update(label="Agent 思考完毕！", state="complete", expanded=False)
                
            except requests.exceptions.ConnectionError:
                status.update(label="服务连接失败", state="error", expanded=False)
                st.error("❌ 无法连接到 FastAPI 后端！请确认 `python api_server.py` (端口 8080) 是否已启动。")
                final_answer = "抱歉，由于后端微服务未启动，无法处理请求。"
            except Exception as e:
                status.update(label="服务器内部错误", state="error", expanded=False)
                st.error(f"❌ 请求遇到问题: {str(e)}")
                final_answer = "抱歉，推理过程中遇到了异常。"

        # 将最终答案渲染到网页上
        st.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})