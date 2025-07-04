from flask import Flask, request, jsonify, render_template
import numpy as np
import re
import pickle
from gensim.models import KeyedVectors

# Initialize Flask app
app = Flask(__name__)

# Load ML model and Word2Vec vectorizer
model = pickle.load(open("model.pkl", "rb"))
word2vec = KeyedVectors.load("word2vec.kv", mmap='r')  # Ensure this file is in .kv format

# --- Enhanced Preprocessing Function ---
def preprocess_text(text):
    # Basic cleaning: remove URLs, mentions, special characters
    text = re.sub(r"http\S+|@\S+|[^A-Za-z\s]", "", text.lower())
    tokens = text.split()

    # Handle negations: "not good" → "not_good"
    processed = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "not" and i + 1 < len(tokens):
            combined = f"not_{tokens[i + 1]}"
            if combined in word2vec.key_to_index:
                processed.append(combined)
            else:
                # If combined form doesn't exist, keep separate if they exist
                if tokens[i] in word2vec.key_to_index:
                    processed.append(tokens[i])
                if tokens[i + 1] in word2vec.key_to_index:
                    processed.append(tokens[i + 1])
            i += 2
        else:
            if tokens[i] in word2vec.key_to_index:
                processed.append(tokens[i])
            i += 1

    return processed

# Convert tokens to averaged vector
def vectorize_text(tokens):
    if not tokens:
        return np.zeros(word2vec.vector_size,)  # Zero vector if no known tokens
    vectors = [word2vec[word] for word in tokens]
    return np.mean(vectors, axis=0)

# --- Flask Routes ---

@app.route('/')
def home():
    return render_template("index.html")  # Make sure this template exists

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        tweet = data.get("tweet", "")

        print(f"[INPUT] Raw tweet: {tweet}")

        tokens = preprocess_text(tweet)
        print(f"[PREPROCESS] Tokens: {tokens}")

        vector = vectorize_text(tokens).reshape(1, -1)
        print(f"[VECTORIZER] Vector shape: {vector.shape}")

        prediction = model.predict(vector)[0]
        print(f"[PREDICTION] Raw: {prediction}")

        sentiment = "Negative 😢" if prediction == 1 else "Positive 😊"
        return jsonify({"sentiment": sentiment})

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": "Something went wrong"}), 500

if __name__ == '__main__':
    app.run(debug=True)
