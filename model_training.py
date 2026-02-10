import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/cleaned_reviews.csv")

# Remove empty reviews
df = df[df["clean_review"].str.strip() != ""]

X = df["clean_review"]
y = df["sentiment"]

# ---------------- TF-IDF ----------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)
X_vec = vectorizer.fit_transform(X)

# ---------------- TRAIN-TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- MODEL ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
