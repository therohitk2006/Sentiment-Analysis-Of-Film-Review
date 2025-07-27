import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Streamlit app
st.title("💬 Sentiment Analysis App")
st.markdown("Type a review to check if it's Positive or Negative.")

review = st.text_input("Enter your review")

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        review_vec = vectorizer.transform([review]).toarray()
        result = model.predict(review_vec)[0]
        st.success("✅ Positive" if result == 1 else "❌ Negative")
