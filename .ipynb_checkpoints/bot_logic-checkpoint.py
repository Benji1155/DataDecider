# bot_logic.py
import random
import json
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Ensure 'punkt' is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Lazy-load the model
model = None
def load_bot_model():
    global model
    if model is None:
        print("[INFO] Loading model...")
        model = load_model('chatbot_model.h5')
        print("[INFO] Model loaded.")

# Load words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load intents
with open('intents.json') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

# Clean and preprocess user input
def clean_up_sentence(sentence):
    sentence_words = sentence.split()  # Simple tokenization
    sentence_words = [word.lower() for word in sentence_words]
    return sentence_words

# Create bag-of-words from sentence
def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    word_to_index = {w: i for i, w in enumerate(words)}  # Faster lookup
    for s in sentence_words:
        index = word_to_index.get(s)
        if index is not None:
            bag[index] = 1
    return np.array(bag)

# Predict intent
def predict_class(sentence):
    load_bot_model()
    bow = bag_of_words(sentence, words)
    print("[DEBUG] Bag of words created")

    res = model.predict(np.array([bow]), verbose=0)[0]
    print("[DEBUG] Prediction done")

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# Get response
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        for intent in intents_json['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    return "I'm not sure how to respond to that. Can you rephrase?"

# Final function that drives the response
def get_bot_response(user_input):
    print(f"[INFO] User input: {user_input}")
    try:
        intents_list = predict_class(user_input)
        response = get_response(intents_list, intents)
    except Exception as e:
        print(f"[ERROR] {e}")
        response = "Oops, something went wrong. Try again soon!"

    return response
