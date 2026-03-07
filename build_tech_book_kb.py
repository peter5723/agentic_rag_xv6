import pymupdf4llm
import json
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
import os

# --- 配置 ---
PDF_PATH = "Cpp_Primer.pdf" # 你的电子书路径
OUTPUT_FILE = "cpp_knowledge_base.jsonl"
DEVICE = 'cuda:1' 

# --- 1. 加载模型 ---
print(f"正在加载 BGE-M3 到 {DEVICE}...")
bge_model = SentenceTransformer("BAAI/bge-m3", device=DEVICE)

# --- 2. 核心：针对代码书籍的高级切分器 ---
# 面试亮点：你使用了“针对编程语言优化的递归切分策略”
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.CPP,  # 告诉它是 C++，它会优先按 }; } 等符号切
    chunk_size=512,         # 每个切片约 512 字符
    chunk_overlap=50        # 重叠 50 字符，防止上下文丢失
)

def process_book(file_path):
    if not os.path.exists(file_path):
        print("文件不存在")
        return

    print("正在解析 PDF (这可能需要几分钟)...")
    
    # pymupdf4llm 支持 page_chunks=True，返回每一页的 markdown
    # 这样我们可以逐页处理，清理页眉页脚
    pages = pymupdf4llm.to_markdown(file_path, page_chunks=True)
    
    all_chunks = []
    
    print(f"PDF 解析完成，共 {len(pages)} 页。开始切分...")

    for i, page in enumerate(pages):
        text = page["text"]
        
        # --- 数据清洗 (Data Cleaning) ---
        # 简单规则：如果这一页字数太少（可能是目录或空白页），跳过
        if len(text) < 50:
            continue
            
        # --- 智能切分 ---
        # LangChain 的 split_text 输入是字符串，输出是列表
        chunks = text_splitter.split_text(text)
        
        for chunk in chunks:
            # 再次清洗：去掉可能是页码的纯数字切片
            if chunk.strip().isdigit(): 
                continue
                
            all_chunks.append({
                "page_num": i + 1,
                "text": chunk
            })
            
    return all_chunks

def main():
    # 1. 处理文本
    chunks_data = process_book(PDF_PATH)
    print(f"有效切片总数: {len(chunks_data)}")
    
    # 2. 向量化 (Batch Processing)
    # 只有文本列表才能送入 encode
    texts = [c["text"] for c in chunks_data]
    
    print("正在生成向量 (Batch Size = 32)...")
    # 1000页的书切片可能很多，这里会有进度条
    embeddings = bge_model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    
    # 3. 保存
    print(f"正在写入 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i, (data, vec) in enumerate(zip(chunks_data, embeddings)):
            record = {
                "id": i,
                "text": data["text"],
                "page": data["page_num"],
                "source": PDF_PATH,
                "embedding": vec.tolist()
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print("✅ C++ Primer 知识库构建完成！")

if __name__ == "__main__":
    main()