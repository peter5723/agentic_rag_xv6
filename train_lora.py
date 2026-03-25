import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# === 配置区域 ===
MODEL_PATH = "models/qwen-7b" # 替换为你本地实际路径
DATA_PATH = "xv6_sft_train_data_v4_massive.jsonl"
OUTPUT_DIR = "./qwen-xv6-lora"

print("🚀 [1/5] 加载 Tokenizer 和 数据集...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# Qwen 的特殊 eos_token 通常是 <|im_end|>，确保 padding 正确
tokenizer.pad_token = tokenizer.eos_token 

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# 【核心修改区】：严格使用 ChatML 格式，并动态映射多任务的 System Prompt
def format_chat_template(example):
    # example['instruction'] 存放的是我们设定的具体人设（裁判/生成器）
    # example['input'] 存放的是 问题 + 检索到的代码
    # example['output'] 存放的是 yes/no 或者 思考过程
    
    chatml_text = (
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>user\n{example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    )
    return {"text": chatml_text}

formatted_dataset = dataset.map(format_chat_template)

print("🚀 [2/5] 加载基础模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

print("🚀 [3/5] 注入 LoRA 适配器 (旁路微调)...")
lora_config = LoraConfig(
    r=16,               
    lora_alpha=32,      
    # 保持全线性层，这是多任务 Agent 微调成功的关键
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() 

print("🚀 [4/5] 配置训练参数并启动训练...")
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,  
    gradient_accumulation_steps=4,  
    learning_rate=2e-4,             
    num_train_epochs=3,             # 建议设为 3 轮，5 轮对这种微调容易过拟合
    logging_steps=5,
    save_strategy="epoch",
    optim="adamw_torch",
    fp16=False,
    bf16=True,                      
    report_to="none",
    dataset_text_field="text",      
    # max_seq_length=2048             # 建议加上截断长度，防止 OOM
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    args=training_args,
    processing_class=tokenizer      
)

trainer.train()

print("🚀 [5/5] 训练完成！保存 LoRA 权重...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ 模型已成功保存在: {OUTPUT_DIR}")