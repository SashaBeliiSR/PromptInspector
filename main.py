from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Папка для кэша модели
CACHE_DIR = "./model_cache"
# Модель, которая умеет следовать инструкциям
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Загружаем мозг: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype="auto", 
    device_map="auto", 
    cache_dir=CACHE_DIR
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/build_prompt", methods=["POST"])
def build_prompt():
    """
    Эта функция берет куски от пользователя и превращает их в ИДЕАЛЬНЫЙ ПРОМПТ.
    """
    data = request.json
    blocks = data.get("blocks", [])

    # Собираем "грязный" черновик из блоков пользователя
    raw_input = ""
    for block in blocks:
        raw_input += f"[{block['title']}]: {block['content']}\n"

    # Это МЕТА-ПРОМПТ. Мы говорим ИИ, как быть промпт-инженером.
    system_instruction = (
        "You are an expert AI Prompt Engineer. Your goal is to rewrite the user's raw notes "
        "into a ONE PERFECT, STRUCTURED SYSTEM PROMPT for an LLM.\n"
        "Rules:\n"
        "1. Organize the prompt into numbered sections (1. Role, 2. Objective, 3. Context, 4. Constraints).\n"
        "2. Add specific details in parentheses explaining HOW to do it.\n"
        "3. Make it strict and professional.\n"
        "4. Output ONLY the optimized prompt, nothing else."
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"Here are the raw notes:\n{raw_input}\n\nCreate the perfect prompt now:"}
    ]

    # Генерируем "Идеальный промпт"
    optimized_prompt = generate_text(messages, max_new_tokens=400, temp=0.7)
    
    return jsonify({"optimized_prompt": optimized_prompt})

@app.route("/execute_prompt", methods=["POST"])
def execute_prompt():
    """
    Эта функция выполняет уже созданный идеальный промпт.
    """
    data = request.json
    final_prompt = data.get("prompt", "")

    messages = [
        {"role": "system", "content": "You are a precise AI engine. Follow the instructions below strictly."},
        {"role": "user", "content": final_prompt}
    ]

    result = generate_text(messages, max_new_tokens=500, temp=0.8)
    return jsonify({"result": result})

def generate_text(messages, max_new_tokens, temp):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temp,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)