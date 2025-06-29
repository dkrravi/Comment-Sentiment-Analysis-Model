import streamlit as st
import numpy as np
import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("bilstm_sentiment_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_length = 50  # Same as training

def clean_comment(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def decode_sentiment(score):
    if score < -0.2:
        return "Negative 😡"
    elif score > 0.2:
        return "Positive 😊"
    else:
        return "Neutral 😐"

st.title("Comment Sentiment Analyzer")

user_input = st.text_area("Enter a comment to analyze:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a comment!")
    else:
        cleaned = clean_comment(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=max_length, padding='post')
        prediction = model.predict(padded)[0][0]
        sentiment = decode_sentiment(prediction)
        
        st.markdown(f"**Sentiment:** `{sentiment}`")
        st.markdown(f"**Score:** `{prediction:.2f}`")
