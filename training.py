
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, OFTConfig
from trl import SFTTrainer
import re

model_name = "Qwen/Qwen3-0.6B"
USE_OFT = True   # True = OFT, False = LoRA
output_dir = "./outputs"


train_dataset = load_dataset("gsm8k", "main")["train"]  # main split 里的 train
test_dataset = load_dataset("gsm8k", "main")["test"]   # main split 里的 test

def extract_answer(text):
    match = re.search(r"####\s*(-?\d+)", text)
    if match:
        return match.group(1)
    return None

def format_example(example):
    return {
        "text": f"### Question:\n{example['question']}\n\n### Answer:\n{example['answer']}"
    }

full_dataset = train_dataset.map(format_example)

train_test = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test['train']  
eval_dataset = train_test['test']  

print(f"训练集: {len(train_dataset)}")
print(f"验证集: {len(eval_dataset)}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

if USE_OFT:
    peft_config = OFTConfig(
        oft_block_size=32,
        target_modules="all-linear",
        use_cayley_neumann=True,
        bias="none",
        task_type="CAUSAL_LM",
    )
    save_name = "oft_model"
else:
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    save_name = "lora_model"


training_args = TrainingArguments(
    output_dir=f"{output_dir}/{save_name}",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=200,
    bf16=True,
    report_to="none",

    eval_strategy="steps",
    eval_steps=200,
    logging_dir=f"{output_dir}/{save_name}/logs",
)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args,
)


trainer.train()


trainer.model.save_pretrained(f"{output_dir}/{save_name}")
tokenizer.save_pretrained(f"{output_dir}/{save_name}")


logs = trainer.state.log_history

train_steps = []
train_losses = []
eval_steps = []
eval_losses = []
lrs = []

for log in logs:
    if "loss" in log:
        train_steps.append(log["step"])
        train_losses.append(log["loss"])
    if "eval_loss" in log:
        eval_steps.append(log["step"])
        eval_losses.append(log["eval_loss"])
    if "learning_rate" in log:
        lrs.append(log["learning_rate"])


plt.figure()
plt.plot(train_steps, train_losses, label="Train Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title(f"Train Loss ({save_name})")
plt.legend()
plt.grid()
plt.savefig(f"{output_dir}/{save_name}/train_loss.png")


if len(eval_losses) > 0:
    plt.figure()
    plt.plot(eval_steps, eval_losses, label="Eval Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"Eval Loss ({save_name})")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/{save_name}/eval_loss.png")

if len(lrs) > 0:
    plt.figure()
    plt.plot(range(len(lrs)), lrs, label="Learning Rate")
    plt.xlabel("Logging Steps")
    plt.ylabel("LR")
    plt.title(f"LR Schedule ({save_name})")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/{save_name}/lr_curve.png")

print("✅ Training & plotting done!")
