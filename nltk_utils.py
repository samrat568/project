import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import json

# Download NLTK dependencies
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def lemmatize(word):
    return lemmatizer.lemmatize(word.lower())

def bag_of_words(tokenized_sentence, words):
    tokenized_sentence = [lemmatize(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in tokenized_sentence:
            bag[idx] = 1
    return bag

def preprocess_data(intents):
    words = []
    classes = []
    documents = []
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # Tokenize each word
            w = tokenize(pattern)
            words.extend(w)
            # Add to documents
            documents.append((w, intent['tag']))
        # Add the tag to classes if it's not there already
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
    
    # Lemmatize and remove duplicates
    words = sorted(set([lemmatize(w) for w in words]))
    classes = sorted(set(classes))

    return words, classes, documents
