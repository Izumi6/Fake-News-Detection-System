import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset (Kaggle-style: title, text, label)
df = pd.read_csv("news.csv")  # make sure this file exists

# Combine title and text into one feature
df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()

# Map numeric labels to strings (optional, for readability)
if df["label"].dtype != "object":
    df["label"] = df["label"].map({0: "real", 1: "fake"})

# Drop rows with missing data
df = df.dropna(subset=["text", "label"])
df = df[df["text"].str.len() > 0]

X = df["text"]
y = df["label"]

# 2. Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF vectorization
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_df=0.95,
    min_df=5
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 4. Train model (Logistic Regression)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_tfidf, y_train)

# 5. Evaluation
y_pred = log_reg.predict(X_test_tfidf)

print("=== Fake News Detection (Logistic Regression) ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Helper function to classify new news text
def predict_news(text: str) -> str:
    vec = tfidf.transform([text])
    label = log_reg.predict(vec)[0]
    return label

# Example usage
sample_news = "Breaking: Government announces new policy changes in education sector."
print("\nSample prediction:", predict_news(sample_news))
