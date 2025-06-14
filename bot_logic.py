import random
import json
import numpy as np
import nltk
import pickle
import os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model

# --- Setup ---
# NLTK will automatically find its data from the NLTK_DATA environment variable.
# We no longer need to manually set paths in the code.
lemmatizer = WordNetLemmatizer()

# --- Lazy-load resources ---
model = None
words = None
classes = None
intents = None
resources_loaded = False

def load_nlu_resources():
    """Loads all necessary NLU files. It only runs once."""
    global model, words, classes, intents, resources_loaded
    if resources_loaded:
        return

    print("[INFO] Attempting to load NLTK model and data files...")
    try:
        # Get the absolute path to the directory where this script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct absolute paths to the resource files
        model_path = os.path.join(base_dir, 'chatbot_model.h5')
        words_path = os.path.join(base_dir, 'words.pkl')
        classes_path = os.path.join(base_dir, 'classes.pkl')
        intents_path = os.path.join(base_dir, 'intents.json')

        model = load_model(model_path)
        words = pickle.load(open(words_path, 'rb'))
        classes = pickle.load(open(classes_path, 'rb'))
        with open(intents_path, 'r', encoding='utf-8') as file:
            intents = json.load(file)
            
        resources_loaded = True
        print("[INFO] NLU resources loaded successfully.")
    except Exception as e:
        print(f"[ERROR] A critical error occurred while loading NLU resources: {e}")
        # Keep resources as None so the bot can report the issue
        model, words, classes, intents = None, None, None, None

# --- Core NLU Functions ---
def clean_up_sentence(sentence):
    """Tokenizes and lemmatizes the sentence."""
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    """Creates a bag-of-words array."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """Predicts the intent for a given sentence."""
    if not all([model, words, classes]):
        print("[ERROR] NLU resources not loaded. Cannot predict class.")
        return [] 

    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]), verbose=0)[0]
    
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    """Gets a response from the intents file."""
    if not intents_list or not intents_json:
        return "I'm not sure how to respond to that. Can you rephrase?"

    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that. Can you rephrase?"

# --- Main Function to be Called by app.py ---
def get_bot_response(user_input):
    """Drives the NLU response generation."""
    # Ensure resources are loaded, but only once.
    load_nlu_resources()
    
    if not resources_loaded:
        return "Sorry, my core components failed to load. Please check the server logs for errors."
        
    try:
        intents_list = predict_class(user_input)
        response = get_response(intents_list, intents)
    except Exception as e:
        print(f"[ERROR] An error occurred during prediction: {e}")
        response = "Oops, something went wrong on my end. Please try again."

    return response
