import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Echo – Sentiment Analysis",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("📊 AI Echo: Sentiment Analysis Dashboard")
st.write("Exploratory Data Analysis of ChatGPT User Reviews")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_reviews.csv")

df_original = load_data()
df = df_original.copy()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🔍 Filter Options")

platform_filter = st.sidebar.selectbox(
    "Select Platform",
    ["All"] + sorted(df["platform"].dropna().unique().tolist())
)

sentiment_filter = st.sidebar.selectbox(
    "Select Sentiment",
    ["All"] + sorted(df["sentiment"].dropna().unique().tolist())
)

if platform_filter != "All":
    df = df[df["platform"] == platform_filter]

if sentiment_filter != "All":
    df = df[df["sentiment"] == sentiment_filter]

# ---------------- DATA PREVIEW ----------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# =========================================================
# 1️⃣ Rating Distribution
# =========================================================
st.subheader("1️⃣ Rating Distribution")

fig, ax = plt.subplots()
df["rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_xlabel("Rating")
ax.set_ylabel("Number of Reviews")
st.pyplot(fig)

# =========================================================
# 2️⃣ Sentiment Distribution
# =========================================================
st.subheader("2️⃣ Sentiment Distribution")

fig, ax = plt.subplots()
df["sentiment"].value_counts().plot(kind="bar", ax=ax)
ax.set_xlabel("Sentiment")
ax.set_ylabel("Count")
st.pyplot(fig)

# =========================================================
# 3️⃣ Positive vs Negative Keywords (Word Cloud)
# =========================================================
st.subheader("3️⃣ Positive vs Negative Review Keywords")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Positive Reviews**")
    positive_reviews = df[df["sentiment"] == "Positive"]["clean_review"].dropna()
    if len(positive_reviews) > 0:
        text = " ".join(positive_reviews)
        wc = WordCloud(background_color="white", width=400, height=300).generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("No positive reviews available.")

with col2:
    st.markdown("**Negative Reviews**")
    negative_reviews = df[df["sentiment"] == "Negative"]["clean_review"].dropna()
    if len(negative_reviews) > 0:
        text = " ".join(negative_reviews)
        wc = WordCloud(background_color="white", width=400, height=300).generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("No negative reviews available.")

# =========================================================
# 4️⃣ Average Rating Over Time
# =========================================================
st.subheader("4️⃣ Average Rating Over Time")

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    trend = df.groupby(df["date"].dt.to_period("M"))["rating"].mean()
    st.line_chart(trend)
else:
    st.info("Date column not available.")

# =========================================================
# 5️⃣ Ratings by Location
# =========================================================
st.subheader("5️⃣ Average Rating by Location (Top 10 Countries)")

top_locations = df["location"].value_counts().head(10).index
fig, ax = plt.subplots()
df[df["location"].isin(top_locations)] \
    .groupby("location")["rating"].mean() \
    .sort_values(ascending=False) \
    .plot(kind="bar", ax=ax)

ax.set_xlabel("Location")
ax.set_ylabel("Average Rating")
st.pyplot(fig)

# =========================================================
# 6️⃣ Platform Comparison (Web vs Mobile)
# =========================================================
st.subheader("6️⃣ Average Rating by Platform")

fig, ax = plt.subplots()
df.groupby("platform")["rating"].mean().plot(kind="bar", ax=ax)
ax.set_xlabel("Platform")
ax.set_ylabel("Average Rating")
st.pyplot(fig)

# =========================================================
# 7️⃣ Verified vs Non-Verified Users
# =========================================================
st.subheader("7️⃣ Verified vs Non-Verified User Ratings")

fig, ax = plt.subplots()
df.groupby("verified_purchase")["rating"].mean().plot(kind="bar", ax=ax)
ax.set_xlabel("Verified Purchase")
ax.set_ylabel("Average Rating")
st.pyplot(fig)

# =========================================================
# 8️⃣ Review Length vs Rating
# =========================================================
st.subheader("8️⃣ Average Review Length per Rating")

fig, ax = plt.subplots()
df.groupby("rating")["review_length"].mean().plot(kind="bar", ax=ax)
ax.set_xlabel("Rating")
ax.set_ylabel("Average Review Length")
st.pyplot(fig)

# =========================================================
# 9️⃣ ChatGPT Version vs Sentiment
# =========================================================
st.subheader("9️⃣ ChatGPT Version vs Sentiment")

if df["version"].nunique() > 1:
    version_sentiment = (
        df.groupby(["version", "sentiment"])
        .size()
        .unstack(fill_value=0)
    )
    st.bar_chart(version_sentiment)
else:
    st.info("Not enough version data available for comparison.")
# =========================================================
# 🔟 Common Negative Feedback Themes
# =========================================================
st.subheader("🔟 Common Negative Feedback Themes")

negative_reviews = df[df["sentiment"] == "Negative"]["clean_review"].dropna()

if len(negative_reviews) > 0:
    all_words = " ".join(negative_reviews).split()
    word_freq = Counter(all_words)
    common_words = dict(word_freq.most_common(10))
    st.bar_chart(common_words)
else:
    st.info("No negative feedback available for selected filters.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("📌 **AI Echo Project** | NLP-based Sentiment Analysis using Streamlit")
