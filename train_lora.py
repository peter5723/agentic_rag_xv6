import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
# 【关键修复 1】：引入 SFTConfig 替代 TrainingArguments
from trl import SFTTrainer, SFTConfig

# === 配置区域 ===
MODEL_PATH = "models/qwen-7b" # 替换为你本地实际路径
DATA_PATH = "xv6_sft_train_data.jsonl"
OUTPUT_DIR = "./qwen-xv6-lora"

print("🚀 [1/5] 加载 Tokenizer 和 数据集...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def format_chat_template(example):
    prompt = f"System: 你是一个严谨的 xv6 内核专家。\nUser: {example['instruction']}\nContext: {example['input']}\nAssistant: {example['output']}"
    return {"text": prompt}

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
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() 

print("🚀 [4/5] 配置训练参数并启动训练...")
# 【关键修复 2】：使用 SFTConfig，并把报错的两个参数挪到这里
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,  
    gradient_accumulation_steps=4,  
    learning_rate=2e-4,             
    num_train_epochs=5,             
    logging_steps=5,
    save_strategy="epoch",
    optim="adamw_torch",
    fp16=False,
    bf16=True,                      
    report_to="none",
    dataset_text_field="text",      # <--- 从 Trainer 移到了这里
)

# 【关键修复 3】：Trainer 初始化变得非常干净，使用 processing_class 替代 tokenizer
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    args=training_args,
    processing_class=tokenizer      # <--- 新版 trl 的要求
)

trainer.train()

print("🚀 [5/5] 训练完成！保存 LoRA 权重...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ 模型已成功保存在: {OUTPUT_DIR}")