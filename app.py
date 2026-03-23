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

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

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

st.title("📰 Fake News Detection")
st.write("Enter a news article to check if it's **Fake or Real**")

news = st.text_area("Enter News Text")

if st.button("Predict"):
    if news.strip() == "":
        st.warning("Please enter some text")
    else:
        result = predict(news)
        if result == 0:
            st.error("Fake News ❌")
        else:
            st.success("Real News ✅")