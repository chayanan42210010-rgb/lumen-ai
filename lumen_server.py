import sys
# บังคับให้ระบบอ่านไลบรารีจากไดรฟ์ D:
sys.path.insert(0, r"D:\PythonLibs")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # อนุญาตให้หน้าเว็บส่งข้อความเข้ามาคุยได้

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

print("🔄 กำลังเตรียมระบบสมองกล Lumen AI (โหมดประหยัดแรม)...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# ใช้ float16 และ low_cpu_mem_usage เพื่อแก้ปัญหา MemoryError แรมเต็ม
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
)

print("✨ Lumen AI Server พร้อมเชื่อมต่อเว็บแล้ว!")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({"response": "กรุณาส่งข้อความหา Lumen AI ครับ"})

    messages = [{"role": "user", "content": user_message}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
    ai_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return jsonify({"response": ai_response})

if __name__ == '__main__':
    # รัน Local Server บนเครื่องตัวเองที่พอร์ต 5000
    app.run(host='127.0.0.1', port=5000, debug=False)
