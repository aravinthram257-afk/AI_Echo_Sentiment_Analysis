import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- DOWNLOAD NLTK RESOURCES (RUN ONCE) ----------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/chatgpt_style_reviews.csv")

# ---------------- HANDLE MISSING VALUES ----------------
df = df.dropna(subset=["review", "rating"])

# ---------------- TEXT PREPROCESSING SETUP ----------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()                              # Convert to lowercase
    text = re.sub(r"[^a-z\s]", "", text)             # Remove special characters
    words = text.split()                             # Tokenization
    words = [
        lemmatizer.lemmatize(word)                   # Lemmatization
        for word in words
        if word not in stop_words                    # Stopword removal
    ]
    return " ".join(words).strip()

# ---------------- APPLY CLEANING ----------------
df["clean_review"] = df["review"].apply(clean_text)

# ---------------- SENTIMENT LABELING ----------------
def rating_to_sentiment(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"
df["sentiment"] = df["rating"].apply(rating_to_sentiment)
# ---------------- SAVE CLEANED DATA ----------------
df.to_csv("data/cleaned_reviews.csv", index=False)
print("✅ Data preprocessing completed successfully")
