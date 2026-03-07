from huggingface_hub import snapshot_download

print("正在导出模型真实文件...")
snapshot_download(
    repo_id="BAAI/bge-m3",
    local_dir="./bge-m3-local",  # 导出到当前目录
    local_dir_use_symlinks=False, # 关键：强制下载真实文件，不要软链接
    resume_download=True,
    ignore_patterns=["*.DS_Store", "imgs/*"]
)
print("完成！")