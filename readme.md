1. 运行 sh run.sh 来启动 vllm，加载大模型
2. 运行后端服务 FASTAPI：sh fastAPI.sh
3. 启动前端网页: streamlit run web_ui.py

LORA：人工标注数据：[eval_dataset.json](eval_dataset.json) [train_seed.json](train_seed.json)
生成数据集：[eval_dataset.json](generate_sft_data.py)
训练：[train_lora.py](train_lora.py)
测试：[eval.py](eval.py)