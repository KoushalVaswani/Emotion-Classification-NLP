# Emotion Classification using NLP 🚀

This repository contains an end-to-end Natural Language Processing (NLP) project that classifies text sentences into six different emotional categories (e.g., joy, sadness, anger, love, surprise, fear). 

The project evaluates multiple text vectorization techniques (Bag of Words, TF-IDF, N-Grams) and machine learning classifiers to determine the most accurate, context-aware, and balanced pipeline.

## 📊 Project Workflow & Features

1. **Data Preprocessing & Cleaning**:
   * Converted text to lowercase.
   * Removed punctuations, numbers, and emojis using custom Python functions.
   * Tokenized text and filtered out English stopwords using `nltk`.

2. **Feature Extraction Techniques**:
   * **Bag of Words (BoW)** using `CountVectorizer` (Uni-grams)
   * **TF-IDF** using `TfidfVectorizer`
   * **Advanced N-Grams (Uni-grams + Bi-grams)** using `CountVectorizer(ngram_range=(1, 2))`

3. **Machine Learning Models Evaluated**:
   * **Multinomial Naive Bayes** (Baseline Model)
   * **Logistic Regression** (Tested with both BoW and TF-IDF)
   * **Support Vector Machines (LinearSVC)** (Tested with both Uni-grams and N-Grams)

---

## 📈 Performance & Results Comparison

We systematically optimized the pipeline step-by-step. Here is the final breakdown of the performance achieved across the test samples:

| Feature Extraction | Classifier Model | Accuracy | Status / Notes |
|--------------------|------------------|----------|----------------|
| Bag of Words (BoW) | Multinomial Naive Bayes | 77.16% | Weakest baseline; struggled heavily with minority classes. |
| TF-IDF Vectorizer  | Logistic Regression     | 85.03% | Good performance, but lower recall on specific emotions. |
| Bag of Words (BoW) | Logistic Regression     | 88.87% | Extremely strong and reliable performance. |
| Bag of Words (BoW) | Linear SVM (LinearSVC)   | 88.97% | Solid linear boundary approach. |
| **BoW + N-Grams (1, 2)** | **Linear SVM (LinearSVC)** | **90.78%** | 🏆 **Ultimate Winner (Best Accuracy & Context Aware)** |

### Key Insights:
* **The Winner**: **Linear SVM combined with Uni-grams + Bi-grams (N-grams)** emerged as the ultimate winner, pushing the accuracy past the **90%** mark (**90.78%**).
* **Context Matters**: Incorporating Bi-grams allowed the model to capture short word sequences (like "not happy" or "very sad"), which drastically improved the classification performance.
* **Class Balance**: The F1-score for the toughest minority class (Class 3) jumped from a poor **0.13** (in Naive Bayes) to a highly impressive **0.79** in the final N-gram SVM model.

---

## 💻 Sample Prediction Pipeline

The finalized context-aware Linear SVM pipeline is fully capable of processing raw text inputs and accurately predicting the underlying emotion:

```python
# Example Prediction using the top-performing N-gram pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

# Input raw text
my_text = ["I feel so sad and depressed today."]
my_text_ngram = cv_ngram.transform(my_text)

# Model Prediction
predicted_class = model_svm_ngram.predict(my_text_ngram)
print(f"Predicted Emotion: {reverse_emotion_numbers[predicted_class[0]]}")

# Output: Predicted Emotion: sadness
