"""
Clean and standardize the training CSV.

Expected input:
    data/emails_large.csv

Expected output:
    data/clean_emails.csv
"""

import os
import re
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "emails_large.csv")
OUT = os.path.join(BASE, "data", "clean_emails.csv")

TEXT_ALIASES = ["body", "text", "email_text", "email_content", "content", "message", "text_combined", "email"]
SUBJECT_ALIASES = ["subject", "title"]
LABEL_ALIASES = ["label", "class", "target", "category", "type"]

def find_column(df, aliases):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    return None

def normalize_label(value):
    s = str(value).strip().lower()
    if s in {"1", "phishing", "phish", "spam", "scam", "malicious", "fraud"}:
        return 1
    if s in {"0", "safe", "legitimate", "ham", "benign", "normal"}:
        return 0
    if "phish" in s or "spam" in s or "scam" in s or "malicious" in s or "fraud" in s:
        return 1
    if "safe" in s or "legit" in s or "ham" in s or "benign" in s:
        return 0
    return np.nan

def normalize_text(text):
    text = "" if pd.isna(text) else str(text)
    return re.sub(r"\s+", " ", text).strip()

def main():
    if not os.path.exists(RAW):
        raise FileNotFoundError("Put your larger CSV at data/emails_large.csv.")

    df = pd.read_csv(RAW, on_bad_lines="skip", low_memory=False)
    body_col = find_column(df, TEXT_ALIASES)
    subject_col = find_column(df, SUBJECT_ALIASES)
    label_col = find_column(df, LABEL_ALIASES)

    if not body_col or not label_col:
        raise ValueError(f"Could not identify text/label columns. Found: {list(df.columns)}")

    subject = df[subject_col].map(normalize_text) if subject_col else ""
    body = df[body_col].map(normalize_text)

    out = pd.DataFrame({
        "subject": subject,
        "body": body,
        "label": df[label_col].map(normalize_label)
    })

    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    out["body"] = out["body"].fillna("")
    out = out[(out["subject"].str.len() + out["body"].str.len()) >= 5]
    out["combined"] = (out["subject"] + " " + out["body"]).str.strip()
    out = out.drop_duplicates(subset=["combined"]).drop(columns=["combined"])
    out = out.reset_index(drop=True)

    if out["label"].nunique() != 2:
        raise ValueError("Cleaned dataset must contain both Safe and Phishing classes.")

    out.to_csv(OUT, index=False)
    print(f"Saved: {OUT}")
    print(f"Rows: {len(out):,}")
    print(out["label"].map({0: "Safe", 1: "Phishing"}).value_counts())

if __name__ == "__main__":
    main()
