import streamlit as st
import pickle
import nltk
import re
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from newspaper import Article

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def predict_news(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0]
    return prediction, prob

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection")

st.warning("⚠️ This model does not verify factual correctness, only text patterns.")

url = st.text_input("🔗 Enter News URL")

import requests
from bs4 import BeautifulSoup

url = st.text_input("🔗 Enter News URL")

if st.button("Fetch & Predict"):
    if url.strip() == "":
        st.warning("Please enter a valid URL")
    else:
        try:
            with st.spinner("Fetching article..."):
                article = Article(url)
                article.download()
                article.parse()
                news_text = article.text

                # Fallback
                if news_text.strip() == "":
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, "html.parser")
                    paragraphs = soup.find_all("p")
                    news_text = " ".join([p.get_text() for p in paragraphs])

            
            st.text_area("Extracted News", news_text, height=200)

            prediction, prob = predict_news(news_text)

            if prediction == 0:
                st.error(f"Fake News ❌ (Confidence: {prob[0]*100:.2f}%)")
            else:
                st.success(f"Real News ✅ (Confidence: {prob[1]*100:.2f}%)")

        except:
            st.error("Failed to fetch article. Try another URL.")