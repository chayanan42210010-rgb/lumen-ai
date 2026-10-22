import sys
# บังคับให้ Python อ่านไลบรารีจากไดรฟ์ D: เท่านั้น
sys.path.insert(0, r"D:\PythonLibs")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

print("🔄 กำลังโหลด Neural Network เข้าสู่หน่วยความจำ...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

print("✨ Lumen AI พร้อมทำงานแล้ว!")

def ask_ai(prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150)

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

if __name__ == "__main__":
    question = "สวัสดี Lumen AI รายงานตัวหน่อย"
    print(f"\n👤 ผู้ใช้: {question}")
    answer = ask_ai(question)
    print(f"🤖 AI: {answer}")
