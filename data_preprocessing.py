import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
df = pd.read_csv("D:\AI_Echo_Sentiment_Analysis\data\chatgpt_style_reviews.csv")
df = df.dropna(subset=['review', 'rating'])
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)
df['clean_review'] = df['review'].apply(clean_text)
def rating_to_sentiment(r):
    if r <= 2:
        return "Negative"
    elif r == 3:
        return "Neutral"
    else:
        return "Positive"
df['sentiment'] = df['rating'].apply(rating_to_sentiment)
df.to_csv("data/cleaned_reviews.csv", index=False)
print("✅ Data preprocessing completed")
