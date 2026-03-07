from vllm import LLM


# # 仅适用于生成式模型（task=generate）
# llm = LLM(model="Qwen/Qwen2.5-VL-7B-Instruct", task="generate")  # Name or path of your model
# output = llm.generate("Hello, my name is")
# print(output)


llm = LLM(
    model="./llm",  # 直接指向本地模型目录
    trust_remote_code=True,  # Qwen 需要此参数
    dtype="bfloat16",  # 节省显存（A100/H100可用float16）
    gpu_memory_utilization=0.96,  # 控制显存占用
    max_model_len=8192,
)
output = llm.generate("我爱你。")
print(output)
