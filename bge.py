import torch
from sentence_transformers import SentenceTransformer

# 1. 检查 CUDA 环境
# 确保 PyTorch 可以看到两张显卡
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    print("错误：未检测到足够的 GPU 设备（需要至少 2 张）。")
    # 如果只有一张卡，则使用 cuda:0
    DEVICE = 'cuda:0'
elif torch.cuda.device_count() > 1:
    # 目标：将 Embedding Model 部署到第二张卡 (cuda:1)
    DEVICE = 'cuda:1'
    print(f"成功检测到 {torch.cuda.device_count()} 张 GPU。BGE-M3 将部署到 {DEVICE}。")
else:
    DEVICE = 'cpu'
    print("GPU 不可用，模型将运行在 CPU 上。")
    
# 2. 加载 BGE-M3 模型
# 使用 sentence-transformers 库加载
try:
    print(f"正在加载 BGE-M3 模型到 {DEVICE}...")
    model_name = "BAAI/bge-m3"
    # SentenceTransformer 会自动处理模型到指定设备上的加载
    bge_model = SentenceTransformer(model_name, device=DEVICE)
    print("BGE-M3 模型加载成功！")

except Exception as e:
    print(f"加载模型时发生错误: {e}")
    
# 3. 测试编码功能
sentences = [
    "本项目旨在建立一个高性能的 RAG 服务。",
    "这是第二条测试语句，内容与上一句无关。",
    "RAG 服务需要高效的向量检索能力。"
]

print("\n开始测试编码...")
# 调用 encode 方法生成向量
embeddings = bge_model.encode(sentences, 
                              batch_size=32, 
                              normalize_embeddings=True, # 推荐：归一化向量
                              show_progress_bar=True)

print(f"输入句子数: {len(sentences)}")
print(f"输出向量维度: {embeddings.shape[1]}") # BGE-M3 的维度是 1024
print("编码测试完成。")

# 4. 验证模型是否在正确的设备上
# 检查模型参数是否位于 cuda:1
try:
    # 提取模型的主体（通常是 Transformer 模型）并检查其设备
    model_device = next(bge_model.parameters()).device
    print(f"模型实际运行设备: {model_device}")
    if str(model_device) == DEVICE.replace(':', ''):
         print("✅ 部署设备检查通过。")
    else:
         print(f"⚠️ 警告：模型未按预期部署到 {DEVICE}，而是在 {model_device}。")
except:
     # 如果模型参数不可见（如某些封装），则跳过此检查
     pass