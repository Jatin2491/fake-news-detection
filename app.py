import streamlit as st
import pickle
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def predict(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    return model.predict(vector)[0]

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("Fake News Detection")

st.warning("⚠️ This model does not verify factual correctness, only text patterns.")

news_text = st.text_area("Enter News Text")

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(news_text)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]

        if prediction == 0:
            st.error(f"Fake News ❌ (Confidence: {prob[0]*100:.2f}%)")
        else:
            st.success(f"Real News ✅ (Confidence: {prob[1]*100:.2f}%)")