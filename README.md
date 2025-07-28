# Sentiment-Analysis-Of-Film-Review
# 🎬 Sentiment Analysis of Movie Reviews

This project performs **Sentiment Analysis** on movie reviews using **Natural Language Processing (NLP)** and **Machine Learning** techniques. It classifies each review as either **Positive** or **Negative** using the **Multinomial Naive Bayes** algorithm and **TF-IDF vectorization**.

---

## 📁 Project Structure

- `dataset/` - Contains the movie review dataset (CSV or TXT format).
- `notebook.ipynb` - Jupyter Notebook with full implementation.
- `README.md` - Project overview and instructions.

---

## 🎯 Objectives

- Preprocess textual movie review data.
- Convert text into numerical features using TF-IDF.
- Train a Naive Bayes model for sentiment classification.
- Evaluate performance using accuracy, confusion matrix, and classification report.
- Visualize word clouds and confusion matrix.

---

## 📚 Technologies Used

- **Language**: Python
- **Libraries**:
  - `pandas`, `numpy` – Data processing
  - `matplotlib`, `seaborn` – Data visualization
  - `sklearn` – ML modeling and evaluation
  - `wordcloud` – Word cloud visualization

---

## 🧰 Libraries Imported

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from wordcloud import WordCloud

