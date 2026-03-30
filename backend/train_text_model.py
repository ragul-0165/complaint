import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("🚀 Loading dataset...")

# ✅ LOAD CORRECT DATASET
data = pd.read_csv("dataset/final_bank_data_v3.csv")

print("✅ Dataset loaded:", data.shape)

# Input & Output
X = data['complaint_text']
y = data['issue_type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("📊 Data split done")

# 🔥 IMPROVED TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=7000,
    ngram_range=(1, 2)   # 🔥 BIGRAMS ADDED
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("🔢 Text vectorized")

# 🔥 IMPROVED MODEL
model = LogisticRegression(
    max_iter=400,
    class_weight='balanced'
)

model.fit(X_train_vec, y_train)

print("✅ Model trained")

# Predictions
y_pred = model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Accuracy:", accuracy)

# Detailed report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/text_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("💾 Model saved successfully!")