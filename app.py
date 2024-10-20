from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import json
import nltk
from nltk_utils import bag_of_words, tokenize, lemmatize
import pickle

# Load the model, words, and classes
model = tf.keras.models.load_model('model/chatbot_model.h5')
with open('model/words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('model/classes.pkl', 'rb') as f:
    classes = pickle.load(f)
with open('data/intents.json') as file:
    intents = json.load(file)

app = Flask(__name__)

def predict_class(sentence):
    sentence_words = tokenize(sentence)
    bag = bag_of_words(sentence_words, words)
    res = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return np.random.choice(i['responses'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def chatbot_response():
    msg = request.form['msg']
    ints = predict_class(msg)
    res = get_response(ints, intents)
    return jsonify({"response": res})

if __name__ == "__main__":
    app.run(debug=True)
