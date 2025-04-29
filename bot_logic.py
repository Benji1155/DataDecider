# bot_logic.py
import random
import json
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import spacy

# Ensure 'punkt' is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load trained model
model = load_model('chatbot_model.h5')

# Load words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load intents
with open('intents.json') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

# Clean and preprocess user input
# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def clean_up_sentence(sentence):
    sentence_words = sentence.split()  # Tokenize by space
    sentence_words = [nlp(word)[0].lemma_ for word in sentence_words]  # Lemmatize using spaCy
    return sentence_words

# Create bag-of-words from sentence
def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Predict intent
def predict_class(sentence, model):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Get response based on intent
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        for intent in intents_json['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    return "I'm not sure how to respond to that. Can you rephrase?"

# Final function to use
def get_bot_response(user_input):
    intents_list = predict_class(user_input, model)
    response = get_response(intents_list, intents)
    return response
