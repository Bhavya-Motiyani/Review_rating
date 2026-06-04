from flask import Flask, render_template, request
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("classification_review_rating_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load length of each sentence
with open("max_length.pkl", "rb") as f:
    max_length = pickle.load(f)


def predict_rating(review):

    sequence = tokenizer.texts_to_sequences([review])

    padded_sequence = pad_sequences(sequence,maxlen=max_length,padding="post",truncating="post")

    prediction = model.predict(padded_sequence,verbose=0)

    predicted_rating = prediction.argmax(axis=1)[0] + 1

    return predicted_rating


@app.route("/", methods=["GET", "POST"])
def home():

    predicted_rating = None

    if request.method == "POST":
        review_text = request.form.get("review", "").strip()
        if review_text:
            predicted_rating = predict_rating(review_text)

    return render_template("index.html",prediction=predicted_rating)


if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )