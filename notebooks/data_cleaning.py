import pandas as pd
import os
import re

print("🚀 Starting data cleaning...")

# Load dataset
data = pd.read_csv("dataset/complaints.csv")

print("Original shape:", data.shape)

# ✅ Keep required columns
data = data[['Consumer complaint narrative', 'Product', 'Issue', 'Sub-issue']]

# Rename columns
data.columns = ['complaint_text', 'product', 'issue', 'sub_issue']

# Remove empty complaints
data = data.dropna(subset=['complaint_text'])

# -------------------------------
# 🔥 TEXT CLEANING (IMPORTANT)
# -------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data['complaint_text'] = data['complaint_text'].apply(clean_text)

# -------------------------------
# 🔥 FILTER BANKING DATA
# -------------------------------

banking_products = [
    'Checking or savings account',
    'Credit card',
    'Money transfers'
]

data = data[data['product'].isin(banking_products)]

print("After filtering banking data:", data.shape)

# -------------------------------
# 🔥 ISSUE TYPE MAPPING (IMPROVED)
# -------------------------------

def map_issue(issue, sub_issue):
    text = str(issue) + " " + str(sub_issue)
    text = text.lower()

    if "transaction" in text or "payment" in text or "charged" in text:
        return "Transaction Issue"
    elif "account" in text:
        return "Account Management"
    elif "card" in text:
        return "Card Issue"
    elif "fee" in text or "interest" in text:
        return "Charges & Fees"
    elif "atm" in text or "technical" in text:
        return "Technical Issue"
    else:
        return "Other"   # 🔥 merged small classes

# Apply mapping
data['issue_type'] = data.apply(lambda x: map_issue(x['issue'], x['sub_issue']), axis=1)

# Add department column
data['department'] = 'Banking'

# -------------------------------
# 🔥 CHECK DISTRIBUTION
# -------------------------------

print("\nClass distribution:")
print(data['issue_type'].value_counts())

# -------------------------------
# 🔥 FINAL SAMPLING
# -------------------------------

data = data.sample(4000, random_state=42)

print("Final shape:", data.shape)

# -------------------------------
# SAVE FILE
# -------------------------------

output_path = "dataset/final_bank_data_v3.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

data.to_csv(output_path, index=False)

print("✅ Dataset cleaned (final version) and saved!")