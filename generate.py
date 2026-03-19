import re
from datasets import load_dataset
from tqdm.auto import tqdm
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# load test split
test_dataset = load_dataset("gsm8k", "main")["test"]

def extract_answer(text):
    # 优先匹配格式化答案 "#### 42" 或 "#### 42.0"，否则返回最后出现的数值（整数或浮点，作为后备）
    if not text:
        return None
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1)
    all_nums = re.findall(r"(-?\d+(?:\.\d+)?)", text)
    return all_nums[-1] if all_nums else None

# 新增：按数值比较两个字符串表示的数字（支持 "48" 和 "48.0" 相等）
def numbers_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    try:
        # 去掉千分位逗号和空白
        as_clean = str(a).replace(",", "").strip()
        bs_clean = str(b).replace(",", "").strip()
        fa = float(as_clean)
        fb = float(bs_clean)
        return abs(fa - fb) <= tol
    except Exception:
        return False

# Load base model and apply PEFT weights
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, "./outputs/oft_model", device_map="auto")
# 如果你保存了 tokenizer 在 outputs/oft_model，否则改为基础 tokenizer 名称
tokenizer = AutoTokenizer.from_pretrained("./outputs/oft_model", trust_remote_code=True)

# 简单的生成函数：给定问题返回模型生成的响应文本和最终解析出的数值（如果有）
def generate_answer(question, max_new_tokens=128, temperature=0.0):
    # 添加明确指令：最后一行只输出答案，格式严格为 "#### <数字>"，不要其他文字
    prompt = (
        f"### Instruction:\n{question}\n\n"
        "### Response:\n"
        "Please answer the question step by step if needed, but IMPORTANT: the final line of your response must be ONLY the numeric answer in the following exact format:\n"
        "#### <number>\n"
        "Do not include any other text after this line.\n"
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
    # 尝试从生成的响应里解析数值答案
    # 先尝试只看 "### Response:" 之后的内容
    gen_part = decoded.split("### Response:")[-1].strip()
    parsed = extract_answer(gen_part) or extract_answer(decoded)
    return gen_part, parsed

if __name__ == "__main__":

    n_eval = 300
    n_eval = min(n_eval, len(test_dataset))

    correct = 0
    pbar = tqdm(test_dataset.select(range(n_eval)), desc="Evaluating")
    for i, example in enumerate(pbar):
        question = example.get("question") or example.get("Problem") or ""
        gold = extract_answer(example.get("answer", ""))  # gsm8k 的答案字段通常是 'answer'
        gen_text, pred = generate_answer(question)

        # 使用数值比较：48 和 48.0 视为相同
        if numbers_equal(pred, gold):
            correct += 1

        # 可选：快速打印前几条示例用于调试（保持进度条整洁）
        if i < 5:
            tqdm.write(f"\n--- Example {i} ---")
            tqdm.write(f"Q: {question}")
            tqdm.write(f"Generated: {gen_text}")
            tqdm.write(f"Parsed: {pred}  Gold: {gold}")

        # 更新进度条后缀显示当前准确率
        pbar.set_postfix({"acc": f"{correct/(i+1):.4f}"})
 
    acc = correct / n_eval if n_eval > 0 else 0.0
    print(f"\nEvaluated {n_eval} examples. Accuracy: {acc:.4f}")