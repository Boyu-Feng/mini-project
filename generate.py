import re
from datasets import load_dataset
from tqdm.auto import tqdm
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

test_dataset = load_dataset("gsm8k", "main")["test"]

def extract_answer(text):

    if not text:
        return None
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1)
    all_nums = re.findall(r"(-?\d+(?:\.\d+)?)", text)
    return all_nums[-1] if all_nums else None

def numbers_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    try:

        as_clean = str(a).replace(",", "").strip()
        bs_clean = str(b).replace(",", "").strip()
        fa = float(as_clean)
        fb = float(bs_clean)
        return abs(fa - fb) <= tol
    except Exception:
        return False

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, "./outputs/oft_model_128", device_map="auto")

tokenizer = AutoTokenizer.from_pretrained("./outputs/oft_model_128", trust_remote_code=True)

def generate_answer(question, max_new_tokens=128, temperature=0.0):

    prompt = (
        "You are an assistant. When answering, think step-by-step and show the full reasoning process.\n"
        "IMPORTANT: The entire model output is considered the answer, and the LAST LINE must be ONLY the numeric answer in the exact format:\n"
        "#### <number>\n"
        "Do not include any additional text after this line.\n\n"
        f"Question: {question}\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    parsed = extract_answer(decoded)
    return decoded, parsed

if __name__ == "__main__":

    n_eval = 300
    n_eval = min(n_eval, len(test_dataset))

    correct = 0
    pbar = tqdm(test_dataset.select(range(n_eval)), desc="Evaluating")
    for i, example in enumerate(pbar):
        question = example.get("question") or example.get("Problem") or ""
        gold = extract_answer(example.get("answer", "")) 
        gen_text, pred = generate_answer(question)
  
        if numbers_equal(pred, gold):
            correct += 1

    acc = correct / n_eval if n_eval > 0 else 0.0
    print(f"\nEvaluated {n_eval} examples. Accuracy: {acc:.4f}")