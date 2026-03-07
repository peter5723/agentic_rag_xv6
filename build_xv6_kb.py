import os
import json
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# --- 配置 ---
SOURCE_DIR = "./xv6-source"       # xv6 源码目录
OUTPUT_FILE = "xv6_kb.jsonl"      # 输出文件
DEVICE_EMBED = 'cpu'           # 你的 BGE 模型在辅卡 (或 cpu)

# --- 1. 加载 Embedding 模型 ---
# 如果你之前下载到了本地，这里换成本地路径
print(f"正在加载 BGE-M3 到 {DEVICE_EMBED}...")
embed_model = SentenceTransformer("BAAI/bge-m3", device=DEVICE_EMBED)

# --- 2. 初始化代码切分器 ---
# 针对 C 语言优化，它会优先在函数定义、大括号 } 处切分
code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.C,
    chunk_size=512,        # 代码块不要太大，保持函数完整性
    chunk_overlap=100      # 重叠部分要够，防止上下文丢失
)

# 针对汇编 (.S) 的简单切分器
asm_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)

def process_xv6_source():
    documents = []
    
    print(f"开始扫描目录: {SOURCE_DIR}")
    
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            # 只处理相关代码文件
            if not file.endswith(('.c', '.h', '.S')):
                continue
                
            file_path = os.path.join(root, file)
            # 获取相对路径，如 "kernel/proc.c"
            relative_path = os.path.relpath(file_path, start=SOURCE_DIR)
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                print(f"跳过文件 {relative_path}: {e}")
                continue
            
            # 选择切分器
            splitter = asm_splitter if file.endswith('.S') else code_splitter
            
            # 切分代码
            chunks = splitter.split_text(content)
            
            for chunk in chunks:
                # 【关键】把文件名加到内容里，让 LLM 知道这是哪个文件
                # 格式：[File: kernel/proc.c] ...code...
                augmented_text = f"// File: {relative_path}\n{chunk}"
                
                documents.append({
                    "text": augmented_text,
                    "file": relative_path,
                    "raw_code": chunk
                })

    print(f"共处理了 {len(documents)} 个代码片段。")
    return documents

def main():
    # 1. 读取并切分
    docs = process_xv6_source()
    
    # 2. 向量化
    texts = [d["text"] for d in docs]
    print("正在生成向量 (Batch Size = 32)...")
    embeddings = embed_model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    
    # 3. 保存
    print(f"正在写入 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for idx, (doc, vec) in enumerate(zip(docs, embeddings)):
            record = {
                "id": idx,
                "text": doc["text"],     # 带文件名的文本
                "file": doc["file"],     # 元数据：文件名
                "embedding": vec.tolist()
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print("✅ xv6 知识库构建完成！")

if __name__ == "__main__":
    main()