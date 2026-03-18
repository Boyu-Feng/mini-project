import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, OFTConfig
from trl import SFTTrainer


model_name = "Qwen/Qwen3-0.6B"
USE_OFT = True   # 👉 True = OFT, False = LoRA
output_dir = "./outputs"

dataset = load_dataset("tatsu-lab/alpaca")

def format_example(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }

dataset = dataset["train"].map(format_example)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
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
    report_to="none"
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    dataset_text_field="text",
    args=training_args,
)


trainer.train()

trainer.model.save_pretrained(f"{output_dir}/{save_name}")
tokenizer.save_pretrained(f"{output_dir}/{save_name}")