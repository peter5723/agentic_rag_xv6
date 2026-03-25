import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

# ==========================================
# 1. 核心路径与基础配置
# ==========================================
MODEL_ID = "models/qwen-7b"  # 你的底座模型路径
DATASET_PATH = "dpo_dataset.jsonl"     # 你的三元组数据集
OUTPUT_DIR = "../out"                    # 训练过程输出目录
FINAL_SAVE_PATH = "../out/dpo_512.pth" # 最终权重保存的文件夹路径

# 提前构建保存目录，极其关键的工程防御！
# 防止在训练了几个小时后，最后 save_pretrained 时因为找不到上级目录而直接抛出 FileNotFoundError
os.makedirs(FINAL_SAVE_PATH, exist_ok=True)

# ==========================================
# 2. 加载模型与分词器 (注意数据类型)
# ==========================================
print("🔥 正在加载分词器和底座模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# 强制设置 pad_token，Qwen 系列通常将 eos_token 作为 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载底座模型。为了保证混合精度计算的稳定性，建议计算精度保持在 bfloat16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16 
)

# ==========================================
# 3. 数据集准备
# ==========================================
# DPO 数据集必须包含三列：'prompt', 'chosen', 'rejected'
print("📚 正在加载 DPO 数据集...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# ==========================================
# 4. LoRA 炼丹炉配置 (注入低秩旁路)
# ==========================================
peft_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # 覆盖 Attention 核心
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ==========================================
# 5. 训练参数配置
# ==========================================
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,      
    gradient_accumulation_steps=4,      
    learning_rate=5e-6,                 
    num_train_epochs=3,                 
    logging_steps=10,
    save_steps=100,
    optim="adamw_torch",
    bf16=True,                          
    remove_unused_columns=False,        
    beta=0.1                            # beta 乖乖放在这里
)

# ==========================================
# 6. 召唤 DPOTrainer 并启动
# ==========================================
print("⚔️ 正在初始化 DPOTrainer...")
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,               # 传 None，trl 会利用 LoRA 自动克隆一个冻结的参考模型
    args=training_args,                        
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

print("🚀 开始进行偏好对齐训练...")
dpo_trainer.train()

# ==========================================
# 7. 优雅地保存最终的 LoRA 权重
# ==========================================
print(f"💾 训练完成！正在将最终权重保存至 {FINAL_SAVE_PATH} ...")
dpo_trainer.model.save_pretrained(FINAL_SAVE_PATH)
tokenizer.save_pretrained(FINAL_SAVE_PATH)
print("✅ 所有流程已顺利结束！")
