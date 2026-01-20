import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
st.set_page_config(
    page_title="AI Echo – Sentiment Analysis",
    layout="wide"
)
st.title("📊 AI Echo: Sentiment Analysis Dashboard")
st.write("Exploratory Data Analysis of ChatGPT User Reviews")
df = pd.read_csv("data/cleaned_reviews.csv")
st.subheader("1️⃣ Rating Distribution")
fig, ax = plt.subplots()
df['rating'].value_counts().sort_index().plot(kind='bar', ax=ax)
ax.set_xlabel("Rating")
ax.set_ylabel("Number of Reviews")
st.pyplot(fig)
st.subheader("2️⃣ Sentiment Distribution")
fig, ax = plt.subplots()
df['sentiment'].value_counts().plot(kind='bar', ax=ax)
ax.set_xlabel("Sentiment")
ax.set_ylabel("Count")
st.pyplot(fig)
st.subheader("3️⃣ Common Words in Negative Reviews")
negative_text = " ".join(
    df[df['sentiment'] == "Negative"]['clean_review']
)
wc = WordCloud(
    background_color="white",
    width=800,
    height=400
).generate(negative_text)
fig, ax = plt.subplots()
ax.imshow(wc)
ax.axis("off")
st.pyplot(fig)
st.subheader("4️⃣ Average Rating by Platform")
fig, ax = plt.subplots()
df.groupby('platform')['rating'].mean().plot(kind='bar', ax=ax)
ax.set_xlabel("Platform")
ax.set_ylabel("Average Rating")
st.pyplot(fig)
st.subheader("5️⃣ Verified vs Non-Verified User Ratings")
fig, ax = plt.subplots()
df.groupby('verified_purchase')['rating'].mean().plot(kind='bar', ax=ax)
ax.set_xlabel("Verified Purchase")
ax.set_ylabel("Average Rating")
st.pyplot(fig)
st.subheader("6️⃣ Average Rating by Location (Top 10 Countries)")
top_locations = df['location'].value_counts().head(10).index
fig, ax = plt.subplots()
df[df['location'].isin(top_locations)] \
    .groupby('location')['rating'].mean() \
    .plot(kind='bar', ax=ax)
ax.set_xlabel("Location")
ax.set_ylabel("Average Rating")
st.pyplot(fig)
st.subheader("7️⃣ Average Review Length per Rating")
fig, ax = plt.subplots()
df.groupby('rating')['review_length'].mean().plot(kind='bar', ax=ax)
ax.set_xlabel("Rating")
ax.set_ylabel("Average Review Length")
st.pyplot(fig)
st.subheader("8️⃣ Average Rating by ChatGPT Version")
fig, ax = plt.subplots()
df.groupby('version')['rating'].mean().plot(kind='bar', ax=ax)
ax.set_xlabel("ChatGPT Version")
ax.set_ylabel("Average Rating")
st.pyplot(fig)
