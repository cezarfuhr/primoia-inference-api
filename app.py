from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

app = Flask(__name__)

# Mapeamento de índice para nome da categoria
category_names = {
    0: "Automação Residencial",
    1: "Operações na Binance"
}

# Carregar o tokenizer e o modelo de categoria
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased')
category_model = BertForSequenceClassification.from_pretrained("./modelos_treinados/modelo_categoria",
                                                               num_labels=len(category_names))
category_model.eval()  # Colocar o modelo de categoria em modo de avaliação

# Dicionário para armazenar modelos de intenção por categoria
intent_models = {}

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in nltk.corpus.stopwords.words('portuguese')]
    return " ".join(words)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data['text']

    # Pré-processamento
    processed_text = preprocess(input_text)

    # Tokenização
    input_encoding = tokenizer(processed_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Inferência para a categoria
    with torch.no_grad():
        output = category_model(**input_encoding)
        category_prediction = torch.argmax(output.logits, dim=-1).item()

    predicted_category = category_names[category_prediction]

    # Carregar o modelo de intenção específico da categoria, se ainda não estiver carregado
    if predicted_category not in intent_models:
        intent_model_path = f"./modelos_treinados/modelo_intencao_{predicted_category.replace(' ', '_').lower()}"
        intent_model = BertForSequenceClassification.from_pretrained(intent_model_path)
        intent_model.eval()  # Colocar o modelo de intenção em modo de avaliação
        intent_models[predicted_category] = intent_model
    else:
        intent_model = intent_models[predicted_category]

    # Inferência para a intenção
    with torch.no_grad():
        intent_output = intent_model(**input_encoding)
        intent_prediction = torch.argmax(intent_output.logits, dim=-1).item()

    # Mapeamento de intenções
    intent_names = {
        "Automação Residencial": {
            0: "Ligar Luz", 1: "Desligar Luz", 2: "Ligar Ar Condicionado", 3: "Desligar Ar Condicionado",
            4: "Abrir Janela", 5: "Fechar Janela", 6: "Ligar Televisão", 7: "Desligar Televisão",
            8: "Aumentar Volume", 9: "Diminuir Volume", 10: "Ligar Cafeteira", 11: "Desligar Cafeteira"
        },
        "Operações na Binance": {
            0: "Comprar Ripple na Binance", 1: "Comprar Solana na Binance",
            2: "Comprar Cardano na Binance", 3: "Comprar Bitcoin na Binance",
            4: "Vender Ripple na Binance", 5: "Vender Solana na Binance",
            6: "Vender Cardano na Binance", 7: "Vender Bitcoin na Binance",
        }
    }

    predicted_intent = intent_names[predicted_category][intent_prediction]

    def replace_strings(text):
        text = re.sub(r'ripley', 'Ripple', text, flags=re.IGNORECASE)
        text = re.sub(r'rifle', 'Ripple', text, flags=re.IGNORECASE)
        text = re.sub(r'xlp', 'XRP', text, flags=re.IGNORECASE)
        return text

    result = {
        'inference': replace_strings(input_text),
        'category': {
            'index': category_prediction,
            'description': predicted_category
        },
        'intention': {
            'index': intent_prediction,
            'description': predicted_intent
        }
    }

    print(result)
    return jsonify(result)


def start():
    app.run(host='0.0.0.0', threaded=True, port=8000)


start()