# Emotion Classification using NLP 🚀

This repository contains an end-to-end Natural Language Processing (NLP) project that classifies text sentences into six different emotional categories (e.g., joy, sadness, anger, love, surprise, fear). 

The project evaluates multiple text vectorization techniques (Bag of Words, TF-IDF) and machine learning classifiers to determine the most accurate and balanced pipeline.

## 📊 Project Workflow & Features

1. **Data Preprocessing & Cleaning**:
   * Converted text to lowercase.
   * Removed punctuations, numbers, and emojis using custom Python functions.
   * Tokenized text and filtered out English stopwords using `nltk`.

2. **Feature Extraction Techniques**:
   * **Bag of Words (BoW)** using `CountVectorizer`
   * **TF-IDF** using `TfidfVectorizer`

3. **Machine Learning Models Evaluated**:
   * **Multinomial Naive Bayes** (Baseline Model)
   * **Logistic Regression** (Tested with both BoW and TF-IDF)
   * **Support Vector Machines (LinearSVC)**

---

## 📈 Performance & Results Comparison

We systematically compared different combinations of Vectorizers and Classifiers. Here is the final breakdown of the performance achieved across 3,200 test samples:

| Feature Extraction | Classifier Model | Accuracy | Status / Notes |
|--------------------|------------------|----------|----------------|
| Bag of Words (BoW) | Multinomial Naive Bayes | 77.16% | Weakest baseline; struggled heavily with minority classes. |
| TF-IDF Vectorizer  | Logistic Regression     | 85.03% | Good performance, but lower recall on specific emotions. |
| Bag of Words (BoW) | Logistic Regression     | 88.87% | Extremely strong and reliable performance. |
| **Bag of Words (BoW)** | **Linear SVM (LinearSVC)** | **88.97%** | 🏆 **Ultimate Winner (Best Accuracy & Balanced F1-Scores)** |

### Key Insights:
* **The Winner**: **Linear SVM with Bag of Words** achieved the highest accuracy of **88.97%**.
* **Class Balance**: Moving from Naive Bayes to Linear SVM dramatically resolved the class imbalance issue. For instance, the F1-score for the toughest minority class (Class 3) jumped from a poor **0.13** to a highly impressive **0.76**.

---

## 💻 Sample Prediction Pipeline

The trained Linear SVM pipeline is fully capable of processing raw text inputs and accurately predicting the underlying emotion:

```python
# Example Prediction using the top-performing pipeline
from sklearn.svm import LinearSVC

# Input raw text
my_text = ["I feel so sad and depressed today."]
my_text_bow = cv.transform(my_text)

# Model Prediction
predicted_class = model_svm.predict(my_text_bow)
print(f"Predicted Emotion: {reverse_emotion_numbers[predicted_class[0]]}")

# Output: Predicted Emotion: sadness
