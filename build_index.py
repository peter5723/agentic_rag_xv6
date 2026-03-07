import faiss
import json
import numpy as np
import os

# --- 配置 ---
# KB_FILE = "cpp_knowledge_base.jsonl"
# INDEX_FILE = "cpp_primer.index"
KB_FILE = "xv6_kb.jsonl"      # 指向刚才生成的文件
INDEX_FILE = "xv6.index"      # 新索引名

def build_faiss_index():
    print("正在读取知识库数据...")
    vectors = []
    
    # 1. 读取所有向量
    # 注意：我们不需要把所有文本读入内存，只需要向量构建索引
    # 文本可以通过 id 在检索后去 jsonl 里查（类似于数据库的“回表”查询）
    with open(KB_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            vectors.append(data["embedding"])
    
    # 转为 numpy 数组 (float32)
    vectors = np.array(vectors).astype('float32')
    print(f"加载了 {vectors.shape[0]} 条向量，维度: {vectors.shape[1]}")

    # 2. 创建索引
    # BGE-M3 的向量已经归一化了，所以 Inner Product (IP) 等价于 余弦相似度
    # IndexFlatIP 是精确检索，对于几十万条数据来说速度非常快
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # 3. 添加数据
    print("正在构建 FAISS 索引...")
    index.add(vectors)
    
    # 4. 保存索引
    faiss.write_index(index, INDEX_FILE)
    print(f"索引已保存至 {INDEX_FILE}")

if __name__ == "__main__":
    build_faiss_index()