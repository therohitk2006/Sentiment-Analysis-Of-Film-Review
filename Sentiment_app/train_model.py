import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Load the IMDB dataset
df = pd.read_csv(r"C:\Users\Administrator\Desktop\Sentiment_app\IMDB Dataset.csv")

# 2. Preprocess the data
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 3. Train-test split
X_train, _, y_train, _ = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# 4. Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

# 5. Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 6. Save vectorizer and model
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model and vectorizer trained and saved successfully from IMDB Dataset!")
