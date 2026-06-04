# Review Rating Prediction using LSTM

> An NLP-powered deep learning project that predicts customer ratings from review text using an LSTM neural network. Built with TensorFlow, Keras, and Flask, it demonstrates the complete workflow from data preprocessing and model training to web deployment.

---

## 📌 Overview

This project uses a Long Short-Term Memory (LSTM) Neural Network to predict customer review ratings on a scale of 1 to 5 stars.

The model is trained on customer review data and learns patterns between textual feedback and user ratings. Once trained, it can predict ratings for previously unseen reviews through a Flask-based web application.

The project demonstrates an end-to-end machine learning workflow, including:

- Data preprocessing
- Text tokenization
- Sequence padding
- Model training
- Model evaluation
- Model deployment

---

## 🚀 Features

- Predicts ratings from 1–5 stars
- LSTM-based deep learning model
- Text preprocessing and tokenization
- Sequence padding for variable-length reviews
- Flask web application for real-time predictions
- Responsive frontend using HTML and CSS
- Saved model and tokenizer for deployment

---

## 📂 Project Structure

```text
Review-Rating-Prediction/
│
├── jupyter Notebooks/
│   ├── Project_model.ipynb
│   └── Review_rating_explanation (Understanding).ipynb
│
├── classification_review_rating_model.keras
├── tokenizer.pkl
├── max_length.pkl
│
├── review_dataset.csv
│
├── rating_app.py
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── README.md
```

---

## 🔄 Workflow

### 1. Data Preparation

The dataset consists of customer reviews paired with ratings ranging from 1 to 5 stars.

The review text is processed using:

- Tokenization
- Vocabulary generation
- Text-to-sequence conversion
- Sequence padding

### 2. Label Encoding

Ratings are converted from:

```text
1, 2, 3, 4, 5
```

to:

```text
0, 1, 2, 3, 4
```

to make them compatible with multiclass classification.

### 3. Model Training

The processed review sequences are used to train an LSTM-based neural network.

The model learns relationships between review text and customer ratings, enabling it to predict ratings for unseen reviews.

### 4. Evaluation

The model is evaluated using:

- Accuracy
- Confusion Matrix
- Classification Report

### 5. Deployment

The trained model, tokenizer, and maximum sequence length are saved and loaded into a Flask application for real-time rating prediction.

---

## 📊 Dataset Distribution

```text
Rating 1 : 11,945
Rating 2 : 1,042
Rating 3 : 1,074
Rating 4 : 2,144
Rating 5 : 22,195
```

The dataset is highly imbalanced, with most reviews belonging to the 1-star and 5-star categories.

---

## ⚠️ Current Limitation

A major challenge in this project is the prediction of **2-star** and **3-star** reviews.

While the model achieves strong overall performance, prediction quality for these middle-rating classes remains noticeably lower than for 1-star and 5-star reviews.

Possible reasons include:

- Significant class imbalance
- Mixed-sentiment reviews that contain both positive and negative opinions
- The ordinal nature of ratings being treated as a standard classification problem

Several approaches were explored to improve these classes, including:

- Hyperparameter tuning
- Different LSTM configurations
- Bidirectional LSTM architectures
- Additional Dense layers
- Dropout and recurrent dropout tuning
- Class weighting
- Oversampling minority classes
- Epoch and batch size experimentation

Despite these experiments, the model continues to struggle with consistently identifying 2-star and 3-star reviews. Improving the prediction quality of these middle-rating classes remains the primary challenge for future work.

---

## 🎯 Future Goal

The primary goal of this project is not only to achieve high overall accuracy but also to improve performance on the underrepresented middle-rating classes, particularly **2-star** and **3-star** reviews.

Future work will focus on exploring more advanced NLP techniques and architectures that can better capture subtle sentiment differences and mixed-opinion reviews.

Potential directions include:

- GRU-based architectures
- Pretrained word embeddings (Word2Vec, GloVe, FastText)
- Transformer-based models such as BERT, DistilBERT, and RoBERTa
- Approaches that better account for the ordinal nature of review ratings

Improving the prediction quality of these middle-rating classes remains the main objective for future iterations of this project.

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/review-rating-prediction.git

cd review-rating-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python rating_app.py
```

Open:

```text
http://localhost:5000
```

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Flask
- HTML
- CSS

---

## 👩‍💻 Author

**Bhavya Motiyani**  
B.Tech in Computer Science and Engineering (Data Science Specialization)  
Gujarat Technological University — VGEC  
📧 *bhavyamotiyani68@gmail.com*
🔗 [LinkedIn Profile](https://www.linkedin.com/in/bhavya-motiyani-059544306)

---
