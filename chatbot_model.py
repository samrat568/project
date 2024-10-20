import os
import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

nltk.download('punkt')
nltk.download('wordnet')

# Define ignore words
ignore_words = ['?', '!', '.', ',', 'a', 'the', 'is', 'it', 'that', 'and', 'or', 'to', 'for']  # Add more common words as needed

# Load intents file
with open('intents.json') as file:
    intents = json.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Prepare data
words = []
classes = []
documents = []

# Iterate over intents and extract words and classes
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add to documents in the corpus
        documents.append((word_list, intent['tag']))
        # Add to classes if it's not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lower, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

# Prepare training data
X_train = []
y_train = []

# Create training data
for doc in documents:
    bag = []
    pattern_words = doc[0]
    # Lemmatize and lower each word and create the bag of words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1 if w in pattern_words else 0)

    X_train.append(bag)
    y_train.append(classes.index(doc[1]))

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=len(classes))

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dense(len(classes), activation='softmax'))  # Match output layer to the number of classes

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Create model directory if it doesn't exist
model_dir = 'model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Created directory: {model_dir}")  # Confirm directory creation

# Fit model
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

# Save the model
model_path = os.path.join(model_dir, 'chatbot_model.keras')
model.save(model_path)
print("Model training complete and saved to:", model_path)
