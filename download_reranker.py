import os
from huggingface_hub import snapshot_download

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("正在下载 bge-reranker-v2-m3 模型...")
snapshot_download(
    repo_id="BAAI/bge-reranker-v2-m3",
    local_dir="./bge-reranker-local",
    local_dir_use_symlinks=False,
    ignore_patterns=["*.DS_Store", "imgs/*", "*.md"]
)
print("✅ 重排模型下载完成！")