from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import requests

app = Flask(__name__)

# Cargar el modelo pre-entrenado y el tokenizer
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model_gpt2")
DEEPL_API_KEY = '6063fdbe-462f-461b-9039-cc6ab873adfa:fx'

# Parámetros de generación ajustables
generation_params = {
    "max_length": 150,
    "num_return_sequences": 1,
    "no_repeat_ngram_size": 2,
    "early_stopping": True,
    "top_k": 50,
    "temperature": 0.7
}

def translate_text(text, target_lang):
    url = "https://api-free.deepl.com/v2/translate"
    params = {
        "auth_key": DEEPL_API_KEY,
        "text": text,
        "target_lang": target_lang
    }
    response = requests.post(url, data=params)
    if response.status_code == 200:
        result = response.json()
        return result['translations'][0]['text']
    else:
        print("Error:", response.status_code, response.text)
        return None

# Función para generar respuestas basadas en una entrada dada
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=300, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, **generation_params)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route("/generate_quote", methods=["GET"])
def generate_quote():
    # Obtener parámetros de la query string
    furniture_type = request.args.get("furniture_type")
    wood_type = request.args.get("wood_type")
    paint_type = request.args.get("paint_type")
    dimensions = request.args.get("dimensions")
    quantity = request.args.get("quantity")
    additional = request.args.get("aditionals")
    
    # Crear el prompt con las características proporcionadas
    prompt = f"Generate a quotation for {quantity} {furniture_type} made of {wood_type} with {paint_type} measuring {dimensions} and {additional} details additionals."
    
    # Generar la respuesta
    response = generate_response(prompt)
    final_response = response.replace(prompt, "").strip()
    
    # Traducir la respuesta a español
    prompt_translated = translate_text(prompt,"es")
    translated_response = translate_text(final_response, "es")
    
    # Devolver la respuesta como JSON
    return jsonify({
        'Prompt': prompt_translated,
        'Cotizacion': translated_response
    })
    
@app.route("/generate_quote_telegram", methods=["POST"])
def generate_quote_telegram():
    data = request.json
    prompt = data.get("prompt")
    input_translated = translate_text(prompt, "en")
    # Generar la respuesta
    response = generate_response(input_translated)
    final_response = response.replace(input_translated, "").strip()
    
    # Traducir la respuesta a español
    prompt_translated = translate_text(prompt, "es")
    translated_response = translate_text(final_response, "es")
    
    # Devolver la respuesta como JSON
    return jsonify({
        'Prompt': prompt_translated,
        'Cotizacion': translated_response
    })

if __name__ == "__main__":
    app.run(debug=True)
