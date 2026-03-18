import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, "./outputs/oft_model")
tokenizer = AutoTokenizer.from_pretrained("./outputs/oft_model")

prompt = "### Instruction:\nExplain why the sky is blue.\n\n### Response:\n"

# Tokenize and move tensors to the model device (can't call .to() on the dict directly)
inputs = tokenizer(prompt, return_tensors="pt")
device = next(model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Ensure evaluation mode and no grad for inference
model.eval()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))