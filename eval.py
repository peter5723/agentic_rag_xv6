import json
import time
import os

# 屏蔽代理，防止连接 vLLM 失败
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"

# 导入你构建好的 LangGraph Agent
from agent_graph import app

def run_evaluation(dataset_path="eval_dataset.json", output_path="eval_results_baseline.json"):
    print(f"🚀 开始加载评测集: {dataset_path}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    results = []
    total_questions = len(dataset)
    hit_count = 0  # 记录成功检索到目标文件的次数
    
    print(f"共加载 {total_questions} 道测试题。开始自动化跑分...\n" + "="*50)
    
    for i, item in enumerate(dataset):
        question = item["question"]
        expected_source = item["source_file"]
        
        print(f"[{i+1}/{total_questions}] Q: {question}")
        
        inputs = {"question": question, "iteration": 0}
        
        start_time = time.time()
        final_answer = ""
        retrieved_contexts = []
        
        try:
            # 运行 LangGraph
            for output in app.stream(inputs):
                for node_name, state_value in output.items():
                    # 捕获检索到的上下文
                    if "context" in state_value:
                        retrieved_contexts = state_value["context"]
                    # 捕获最终回答
                    if "answer" in state_value:
                        final_answer = state_value["answer"]
                        
        except Exception as e:
            final_answer = f"Error: {str(e)}"
            
        latency = time.time() - start_time
        
        # --- 计算检索命中 (Context Recall 简易版) ---
        # 我们的 context 长这样: "// File: kernel/proc.c\n代码内容..."
        retrieved_files = []
        for ctx in retrieved_contexts:
            first_line = ctx.split("\n")[0]
            if "// File:" in first_line:
                file_name = first_line.replace("// File:", "").strip()
                retrieved_files.append(file_name)
        
        # 判断期望的文件是否在检索到的文件列表中
        is_hit = False
        for f in retrieved_files:
            # 比如: "kernel/proc.c".endswith("proc.c") 为 True
            if expected_source.endswith(f) or f.endswith(expected_source):
                is_hit = True
                break
        if is_hit:
            hit_count += 1
            print(f"  ✅ 命中文件: {expected_source} (耗时: {latency:.2f}s)")
        else:
            print(f"  ❌ 未命中文件。期望: {expected_source}, 实际找到: {list(set(retrieved_files))}")
            
        # 记录结果
        result_record = {
            "id": i + 1,
            "question": question,
            "expected_source": expected_source,
            "retrieved_files": list(set(retrieved_files)),
            "is_hit": is_hit,
            "expected_answer": item["expected_answer"],
            "generated_answer": final_answer,
            "latency_seconds": round(latency, 2)
        }
        results.append(result_record)
        print("-" * 30)
        
    # --- 保存结果并输出统计 ---
    hit_rate = (hit_count / total_questions) * 100
    print("\n" + "="*50)
    print("🎉 自动化评测完成！")
    print(f"📊 检索命中率 (Hit Rate): {hit_rate:.2f}% ({hit_count}/{total_questions})")
    print(f"💾 详细结果已保存至: {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run_evaluation()