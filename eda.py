import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/cleaned_reviews.csv")

# ================================
# 1. Rating Distribution
# ================================
plt.figure(figsize=(6,4))
df["rating"].value_counts().sort_index().plot(kind="bar")
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.show()

# ================================
# 2. Sentiment Distribution
# ================================
plt.figure(figsize=(6,4))
df["sentiment"].value_counts().plot(kind="bar")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ================================
# 3. Negative Reviews Word Cloud
# ================================
negative_reviews = df[df["sentiment"] == "Negative"]["clean_review"].dropna()

if len(negative_reviews) > 0:
    text = " ".join(negative_reviews)

    wc = WordCloud(
        background_color="white",
        width=800,
        height=400
    ).generate(text)

    plt.figure(figsize=(8,4))
    plt.imshow(wc)
    plt.axis("off")
    plt.title("Negative Reviews Word Cloud")
    plt.tight_layout()
    plt.show()
else:
    print("No negative reviews available for word cloud.")

# ================================
# 4. Average Rating by Platform
# ================================
plt.figure(figsize=(6,4))
df.groupby("platform")["rating"].mean().plot(kind="bar")
plt.title("Average Rating by Platform")
plt.xlabel("Platform")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.show()

# ================================
# 5. Verified vs Non-Verified Users
# ================================
plt.figure(figsize=(6,4))
df.groupby("verified_purchase")["rating"].mean().plot(kind="bar")
plt.title("Average Rating: Verified vs Non-Verified Users")
plt.xlabel("Verified Purchase")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.show()

# ================================
# 6. Average Rating by Location (Top 10)
# ================================
top_locations = df["location"].value_counts().head(10).index

plt.figure(figsize=(8,4))
df[df["location"].isin(top_locations)] \
    .groupby("location")["rating"].mean() \
    .sort_values(ascending=False) \
    .plot(kind="bar")

plt.title("Average Rating by Location (Top 10 Countries)")
plt.xlabel("Location")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.show()

# ================================
# 7. Review Length vs Rating
# ================================
plt.figure(figsize=(6,4))
df.groupby("rating")["review_length"].mean().plot(kind="bar")
plt.title("Average Review Length per Rating")
plt.xlabel("Rating")
plt.ylabel("Average Review Length")
plt.tight_layout()
plt.show()
# ================================
# 8. Average Rating by ChatGPT Version
# ================================
plt.figure(figsize=(6,4))
df.groupby("version")["rating"].mean().sort_index().plot(kind="bar")
plt.title("Average Rating by ChatGPT Version")
plt.xlabel("ChatGPT Version")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.show()
# ================================
# 9. ChatGPT Version vs Sentiment
# ================================

# Average rating by version
plt.figure(figsize=(6,4))
df.groupby("version")["rating"].mean().sort_index().plot(kind="bar")
plt.title("Average Rating by ChatGPT Version")
plt.xlabel("ChatGPT Version")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.show()

# Sentiment distribution by version
version_sentiment = (
    df.groupby(["version", "sentiment"])
    .size()
    .unstack(fill_value=0)
)
plt.figure(figsize=(8,4))
version_sentiment.plot(kind="bar")
plt.title("Sentiment Distribution by ChatGPT Version")
plt.xlabel("ChatGPT Version")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.show()
# ================================
# 10. Common Negative Feedback Themes
# ================================
from collections import Counter
negative_reviews = df[df["sentiment"] == "Negative"]["clean_review"].dropna()
if len(negative_reviews) > 0:
    all_words = " ".join(negative_reviews).split()
    word_freq = Counter(all_words)
    top_negative_words = dict(word_freq.most_common(10))
    plt.figure(figsize=(6,4))
    plt.bar(top_negative_words.keys(), top_negative_words.values())
    plt.title("Top Negative Feedback Keywords")
    plt.xlabel("Keywords")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("No negative reviews available.")
