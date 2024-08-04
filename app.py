from flask import Flask, request, jsonify, render_template
import torch
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering

app = Flask(__name__)

model_name = 'finetuned_distilbert_squad'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def answer_question(question, context):
    inputs = tokenizer.encode_plus(
        question,
        context,
        add_special_tokens=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    answer_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_index:end_index + 1])
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/answer', methods=['POST'])
def get_answer():
    data = request.json
    question = data.get('question')
    context = data.get('context')
    if not question or not context:
        return jsonify({'error': 'Please provide both question and context'}), 400

    answer = answer_question(question, context)
    return jsonify({'answer': answer})


if __name__ == "__main__":
    app.run(debug=True)
