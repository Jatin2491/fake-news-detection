
import streamlit as st
import pickle
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Fake News Detection System")

st.write("Enter a news article below to check whether it is Fake or Real.")

news_text = st.text_area("Enter News Text")

if st.button("Predict"):

    vect_text = vectorizer.transform([news_text])

    prediction = model.predict(vect_text)

    if prediction[0] == 0:
        st.error("This News is Fake")
    else:
        st.success("This News is Real")