import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# App title
st.title("📰 Fake News Detector")
st.write("Enter a news headline to check if it's Real or Fake.")

# Input box
user_input = st.text_area("News Headline", "")

# Predict button
if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a news headline.")
    else:
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]
        label = "✅ Real News" if prediction == 1 else "❌ Fake News"
        st.subheader("Prediction:")
        st.success(label)
