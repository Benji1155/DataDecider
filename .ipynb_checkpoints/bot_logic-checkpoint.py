# bot_logic.py

import numpy as np
from tensorflow.keras.models import load_model
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer

# Load your trained model and related files
model = load_model('model.h5')  # adjust path if needed
with open('intents.json') as file:
    intents = json.load(file)

# Load tokenizer and label encoder
import pickle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

lemmatizer = WordNetLemmatizer()

def preprocess_input(user_input):
    # Tokenize and lemmatize input
    tokens = nltk.word_tokenize(user_input)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    return ' '.join(tokens)

def get_bot_response(user_input):
    processed_input = preprocess_input(user_input)
    
    # Tokenize for the model
    sequences = tokenizer.texts_to_sequences([processed_input])
    padded_sequences = np.array(sequences)
    
    # Predict intent
    predictions = model.predict(padded_sequences)
    intent_idx = np.argmax(predictions)
    intent_label = lbl_encoder.inverse_transform([intent_idx])[0]

    # Find a matching response
    for intent in intents['intents']:
        if intent['tag'] == intent_label:
            return random.choice(intent['responses'])

    return "I'm not sure how to respond to that. Can you rephrase?"
