# ⭐ Review Rating Prediction using LSTM with Synthetic Data Augmentation

> An end-to-end Natural Language Processing (NLP) project that predicts customer review ratings (1–5 stars) using a Long Short-Term Memory (LSTM) neural network. The project also addresses severe class imbalance by generating synthetic customer reviews using **Ollama (Gemma 3 4B)**, significantly improving the model's ability to classify minority rating classes.

---

## 📌 Overview

Customer reviews are one of the richest sources of user feedback for businesses. Automatically predicting the rating associated with a review enables companies to analyze customer satisfaction without requiring explicit ratings.

This project builds an end-to-end deep learning pipeline capable of predicting review ratings from raw review text.

The project covers the complete machine learning workflow:

- Data preprocessing
- Text tokenization
- Sequence padding
- LSTM model training
- Model evaluation
- Class imbalance handling
- Synthetic data generation using a Large Language Model (LLM)
- Flask deployment for real-time predictions

Unlike a traditional text classification project, this work focuses heavily on solving the real-world problem of **imbalanced multiclass datasets**, where middle-rating reviews (2★, 3★ and 4★) are significantly underrepresented.

---

# 🚀 Features

- Predicts customer ratings from **1–5 stars**
- LSTM-based deep learning architecture
- Automatic text preprocessing pipeline
- Tokenization and sequence padding
- Synthetic review generation using **Ollama + Gemma 3 4B**
- Automatic duplicate removal
- Resume generation after interruption
- Real-time prediction through Flask
- Responsive HTML/CSS frontend
- Saved tokenizer and trained model for deployment
- Detailed evaluation using multiple metrics

---

# 📂 Project Structure

```text
Review-Rating-Prediction/
│
├── Jupyter Notebooks/
│   ├── Project_model.ipynb
│   ├── Review_rating_explanation.ipynb
│   └── Synthetic_Review_Generator.ipynb
│
├── classification_review_rating_model.keras
├── tokenizer.pkl
├── max_length.pkl
│
├── review_dataset.csv
├── synthetic_reviews.csv
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

# 🔄 Project Workflow

The project follows the complete NLP pipeline shown below.

```text
Customer Reviews
        │
        ▼
Text Cleaning
        │
        ▼
Tokenization
        │
        ▼
Text to Integer Sequences
        │
        ▼
Sequence Padding
        │
        ▼
LSTM Neural Network
        │
        ▼
Rating Prediction (1–5 Stars)
```

To improve minority class prediction, an additional data augmentation pipeline was developed.

```text
Original Reviews
        │
        ▼
Random Sampling
        │
        ▼
Ollama (Gemma 3 4B)
        │
        ▼
Synthetic Review Generation
        │
        ▼
Duplicate Removal
        │
        ▼
Balanced Dataset
        │
        ▼
Model Retraining
```

---

# 📚 Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Flask
- Ollama
- Gemma 3 4B
- HTML
- CSS

---

# 🧠 Model Architecture

The review text is converted into padded integer sequences before being passed to the neural network.

Architecture:

```text
Embedding Layer
        │
        ▼
LSTM Layer
        │
        ▼
Output Layer (Softmax)
```

The model is trained using:

- Sparse Categorical Crossentropy Loss
- Adam Optimizer
- Accuracy as the primary optimization metric

---

# 📊 Dataset

The project uses a dataset consisting of customer reviews paired with their corresponding ratings ranging from **1 to 5 stars**.

Each record contains:

- Customer review text
- Associated numerical rating

The task is formulated as a **5-class text classification** problem where the objective is to predict the rating solely from the review text.

---

# ⚠️ Initial Challenge: Severe Class Imbalance

The original dataset was highly imbalanced.

```text
Rating 1 : 11,945
Rating 2 : 1,042
Rating 3 : 1,074
Rating 4 : 2,144
Rating 5 : 22,195
```

Most customer reviews belonged to the **1-star** and **5-star** categories, while **2-star**, **3-star**, and **4-star** reviews represented only a small portion of the dataset.

This caused the model to become heavily biased toward predicting the majority classes, resulting in very poor performance on the middle ratings.

---

# 📉 Why Class Imbalance Matters

Although the initial model achieved a high overall accuracy, a detailed analysis revealed that the model almost ignored the minority classes.

The classifier frequently predicted:

- 1★ instead of 2★
- 5★ instead of 4★
- 2★ instead of 3★
- 4★ instead of 5★

This demonstrated that **accuracy alone was not an appropriate evaluation metric** for this problem.

Instead, class-wise Precision, Recall, F1-score, and Macro F1-score became much more informative.

---

# 🤖 Synthetic Data Generation using Ollama

To improve the minority classes, a synthetic review generation pipeline was developed.

Instead of duplicating existing reviews or applying traditional oversampling techniques, a Large Language Model (LLM) was used to generate entirely new customer reviews.

The project uses:

- **Ollama**
- **Gemma 3 4B**

to generate realistic customer reviews for the underrepresented rating classes.

Each generated review preserves the overall sentiment of the target rating while introducing new wording, sentence structures, and vocabulary.

---

# 🔄 Synthetic Review Generation Pipeline

The synthetic data generation notebook was designed to automate the entire augmentation process.

Pipeline:

```text
Original Reviews (from 2,3 and 4 stars)
        │
        ▼
Randomly Sample Real Reviews
        │
        ▼
Prompt Gemma 3 4B
        │
        ▼
Generate New Reviews
        │
        ▼
Append to Synthetic Dataset
        │
        ▼
Repeat Until Target Size Reached
```

Unlike manual prompting, the notebook automatically continues generating reviews until the desired number of samples for each class is reached.

---

# ⚙️ Prompt Engineering

For every generation request, the model receives a randomly selected subset of authentic reviews from the target rating.

The prompt instructs the language model to:

- Preserve the target sentiment
- Avoid copying existing reviews
- Generate diverse writing styles
- Produce varying sentence lengths
- Output only customer reviews
- Avoid numbering or quotation marks
- Generate natural sounding language

Random sampling of seed reviews ensures that each generation request produces diverse outputs instead of repeating similar phrases.

---

# 🛡️ Data Quality Measures

Several safeguards were incorporated into the generation pipeline to improve dataset quality.

### Duplicate Detection

Every generated review is converted to lowercase and compared against:

- Original dataset
- Previously generated reviews

Exact duplicates are automatically discarded.

---

### Automatic Resume

The notebook can resume generation after interruption.

Whenever the generation process starts, it:

- Loads the previously generated synthetic dataset
- Counts the existing synthetic reviews
- Calculates the remaining reviews required
- Continues generation without repeating completed work

This feature proved especially useful during long-running generation sessions.

---

# 📈 Dataset After Augmentation

The minority classes were expanded using LLM-generated synthetic reviews until approximately **10,000 reviews** were available for each class.

Final class distribution:

```text
Rating 1 : 11,945
Rating 2 : 10,000
Rating 3 : 10,000
Rating 4 : 10,000
Rating 5 : 22,195
```

The resulting dataset is substantially more balanced while retaining all authentic customer reviews.

This augmented dataset was then used to retrain the LSTM model and evaluate the impact of synthetic data generation on minority-class prediction performance.

---

# 📊 Model Evaluation

The model was evaluated using multiple performance metrics rather than relying solely on overall accuracy.

Evaluation metrics include:

- Accuracy
- Mean Absolute Error (MAE)
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

Since the dataset is imbalanced, **Macro F1-score** is considered a more informative metric than overall accuracy because it assigns equal importance to every rating class.

---

# 🧪 Experiment 1 — Original Dataset

The initial model was trained using the original imbalanced dataset.

### Results

**Accuracy**

```text
85.97%
```

**Mean Absolute Error**

```text
0.218
```

### Confusion Matrix

```text
[[3527   26   78   66   35]
 [ 247    7   25   33   14]
 [ 138    6   46   93   53]
 [  57    6   43  129  435]
 [  62    6   50  211 6607]]
```

### Classification Report

```text
              precision    recall   f1-score   support

1-Star          0.87       0.95       0.91      3732
2-Star          0.14       0.02       0.04       326
3-Star          0.19       0.14       0.16       336
4-Star          0.24       0.19       0.21       670
5-Star          0.92       0.95       0.94      6936

Accuracy                              0.86

Macro Avg                             0.45

Weighted Avg                          0.84
```

---

## Observation

Although the overall accuracy was high, the model almost completely failed to identify **2-star**, **3-star**, and **4-star** reviews.

The classifier learned to predict the majority classes effectively but struggled to distinguish the minority classes because of the severe imbalance in the training data.

This motivated the use of synthetic data augmentation.

---

# 🧪 Experiment 2 — Dataset After Synthetic Data Augmentation

After generating synthetic reviews using **Ollama (Gemma 3 4B)**, the model was retrained using the augmented dataset.

### Results

**Accuracy**

```text
83.58%
```

**Mean Absolute Error**

```text
0.228
```

### Confusion Matrix

```text
[[3491   99   45   12   86]
 [ 285 1529  118   25   43]
 [ 118  362 1281  165   74]
 [  45   24  307 1071  553]
 [  81   10   59  226 6560]]
```

### Classification Report

```text
              precision    recall   f1-score   support

1-Star          0.87       0.94       0.90      3733
2-Star          0.76       0.76       0.76      2000
3-Star          0.71       0.64       0.67      2000
4-Star          0.71       0.54       0.61      2000
5-Star          0.90       0.95       0.92      6936

Accuracy                              0.84

Macro Avg                             0.77

Weighted Avg                          0.83
```

---

# 📈 Performance Comparison

| Metric | Before Augmentation | After Augmentation |
|---------|--------------------:|-------------------:|
| Accuracy | **85.97%** | 83.58% |
| Macro F1-score | **0.45** | **0.77** |
| Weighted F1-score | **0.84** | 0.83 |
| 2★ F1-score | **0.04** | **0.76** |
| 3★ F1-score | **0.16** | **0.67** |
| 4★ F1-score | **0.21** | **0.61** |

---

# 🔍 Analysis

The augmented model achieved a slightly lower overall accuracy while dramatically improving its ability to classify the minority rating classes.

The most significant improvements were observed for the underrepresented ratings:

| Rating | Before F1 | After F1 |
|---------|----------:|---------:|
| ⭐ 2 | 0.04 | **0.76** |
| ⭐ 3 | 0.16 | **0.67** |
| ⭐ 4 | 0.21 | **0.61** |

These improvements indicate that the synthetic reviews helped the model learn meaningful linguistic patterns for the previously underrepresented classes.

The confusion matrix also shows that prediction errors are now concentrated between **adjacent ratings** (for example, 3★ ↔ 4★ and 4★ ↔ 5★), which is expected because neighboring ratings often contain similar language and sentiment.

---

# 💡 Key Takeaways

This project demonstrates that:

- High overall accuracy can hide poor minority-class performance.
- Macro F1-score provides a better measure of balanced multiclass classification.
- Large Language Models can be effectively used for synthetic text data augmentation.
- Synthetic reviews substantially improved prediction performance for underrepresented rating classes.
- Evaluating models using class-wise metrics provides much deeper insights than accuracy alone.

Rather than optimizing only for overall accuracy, this project focuses on developing a classifier that performs consistently across all five rating categories.

---

# 🌐 Web Application

To make the trained model easily accessible, a Flask-based web application was developed.

The application loads the trained model, tokenizer, and maximum sequence length, allowing users to enter any customer review and receive an instant predicted rating.

### Features

- Real-time review rating prediction
- Clean and responsive user interface
- Loads pre-trained model without retraining
- Automatic text preprocessing
- Fast inference using the saved LSTM model

Workflow:

```text
User Review
      │
      ▼
Text Preprocessing
      │
      ▼
Tokenization
      │
      ▼
Sequence Padding
      │
      ▼
Trained LSTM Model
      │
      ▼
Predicted Rating (1–5 Stars)
```

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/review-rating-prediction.git

cd review-rating-prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Start the Flask server:

```bash
python rating_app.py
```

Open your browser and visit:

```text
http://localhost:5000
```

Enter any customer review and the application will predict its rating.

---

# 🖥️ Example Predictions

### Example 1

**Input**

```text
The product quality exceeded my expectations and delivery was very fast.
```

**Prediction**

```text
★★★★★ (5 Stars)
```

---

### Example 2

**Input**

```text
The product works fine but the packaging was damaged when it arrived.
```

**Prediction**

```text
★★★★☆ (4 Stars)
```

---

### Example 3

**Input**

```text
Average quality. Nothing particularly good or bad about it.
```

**Prediction**

```text
★★★☆☆ (3 Stars)
```

---

### Example 4

**Input**

```text
The product is usable, but I expected much better quality for the price.
```

**Prediction**

```text
★★☆☆☆ (2 Stars)
```

---

### Example 5

**Input**

```text
Completely disappointed. The product stopped working within two days.
```

**Prediction**

```text
★☆☆☆☆ (1 Star)
```

---

# 📚 Libraries Used

- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Flask
- Ollama
- Requests
- OpenPyXL
- HTML
- CSS

---


# 🎓 Learning Outcomes

This project provided practical experience in:

- Natural Language Processing (NLP)
- Text preprocessing
- Deep Learning with LSTM
- Sequence modeling
- Handling imbalanced datasets
- Synthetic data generation using Large Language Models
- Prompt engineering
- Model evaluation
- Flask deployment
- End-to-end machine learning workflows

---

# 📄 Conclusion

This project demonstrates how deep learning and modern Large Language Models can be combined to build more robust NLP systems.

Rather than relying solely on overall accuracy, the project focuses on understanding and improving minority-class performance through careful evaluation and data augmentation.

By generating realistic synthetic reviews using **Ollama (Gemma 3 4B)**, the model became substantially better at identifying underrepresented rating classes while maintaining strong overall predictive performance.

The project highlights the importance of thoughtful data engineering, balanced evaluation metrics, and iterative model improvement in solving real-world machine learning problems.

---

# 👨‍💻 Author

**Bhavya Motiyani**

B.Tech in Computer Science and Engineering  
(Data Science Specialization)

Gujarat Technological University (GTU)

Vishwakarma Government Engineering College (VGEC)

📧 **Email:** bhavyamotiyani68@gmail.com

🔗 **LinkedIn:** https://www.linkedin.com/in/bhavya-motiyani-059544306

---

## ⭐ If you found this project interesting, consider giving it a star!
