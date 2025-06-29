import pandas as pd
import re
from textblob import TextBlob

def clean_comment(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_sentiment_encoded(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 1
    elif polarity < -0.1:
        return -1
    else:
        return 0

df = pd.read_csv("comments.csv")
df = df[['comment']]
df['comment'] = df['comment'].apply(clean_comment)
df = df[df['comment'].str.strip() != ""]
df['encoded_label'] = df['comment'].apply(get_sentiment_encoded)
df.to_csv("comment.csv", index=False)
