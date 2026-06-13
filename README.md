# Emotion Classification using NLP 🚀

This repository contains an end-to-end Natural Language Processing (NLP) project that classifies text sentences into different emotional categories (e.g., joy, sadness, anger, love, etc.). 

The project evaluates multiple text vectorization techniques and machine learning classifiers to find the best performing pipeline.

## 📊 Project Workflow & Features

1. **Data Preprocessing & Cleaning**:
   * Handled missing values (if any).
   * Converted text to lowercase.
   * Removed punctuations, numbers, HTML tags, URLs, and emojis.
   * Tokenized text and removed English stopwords using `nltk`.

2. **Feature Extraction Techniques**:
   * **Bag of Words (BoW)** using `CountVectorizer`
   * **TF-IDF** using `TfidfVectorizer`

3. **Machine Learning Models Evaluated**:
   * **Multinomial Naive Bayes** (Baseline Model)
   * **Logistic Regression** (Optimized with higher iterations)

---

## 📈 Performance & Results

We compared different combinations of Vectorizers and Classifiers. Here is the breakdown of the accuracy achieved:

| Vectorizer | Classifier | Accuracy | Status |
|------------|------------|----------|--------|
| Bag of Words (BoW) | Multinomial Naive Bayes | 77.16% | decent baseline |
| **Bag of Words (BoW)** | **Logistic Regression** | **88.87%** | 🏆 **Best Performer** |
| TF-IDF | Logistic Regression | 85.03% | Good balance |

### Key Insights:
* **Logistic Regression + BoW** emerged as the winner with **~89% accuracy**.
* Transitioning from Naive Bayes to Logistic Regression significantly improved the recall for minority/complex emotion classes (specifically Class 3).

---

## 💻 Sample Prediction Pipeline

The final trained model is capable of predicting emotions from completely raw/new text inputs. 

```python
# Example Prediction using TF-IDF + Logistic Regression
my_text = ["I feel so sad and depressed today."]
my_text_tfidf = tfidf.transform(my_text)

new_pred = model_lr_tfidf.predict(my_text_tfidf)
print(f"Predicted emotion: {reverse_emotion_numbers[new_pred[0]]}")
# Output: Predicted emotion: sadness
