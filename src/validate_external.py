"""Validate the saved model on an independent, unseen CSV dataset."""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

sys.path.append(os.path.dirname(__file__))
from feature_extraction import combine_email_text, build_numeric_features

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "independent_validation.csv")
MODEL_PATH = os.path.join(BASE, "models", "phishing_model.joblib")
OUT_DIR = os.path.join(BASE, "outputs")
OUT = os.path.join(OUT_DIR, "independent_validation_results.csv")
os.makedirs(OUT_DIR, exist_ok=True)

TEXT_ALIASES = ["body", "text", "email_text", "email_content", "content", "message", "email", "text_combined"]


def find_col(df, aliases):
    low = {str(c).lower().strip(): c for c in df.columns}
    return next((low[a] for a in aliases if a in low), None)


def label_value(x):
    s = str(x).strip().lower()
    if s in {"1", "phishing", "phish", "spam", "scam", "malicious", "fraud"} or "phish" in s or "spam" in s:
        return 1
    if s in {"0", "safe", "legitimate", "ham", "benign", "normal"} or "legit" in s or "benign" in s:
        return 0
    return np.nan


def read_csv_robust(path):
    for sep in [None, ",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python" if sep is None else None, on_bad_lines="skip")
            if len(df.columns) >= 2:
                return df
        except Exception:
            pass
    raise ValueError("Could not parse the independent validation CSV.")


def main():
    if not os.path.exists(DATA):
        raise FileNotFoundError(
            "Add an independent unseen CSV as data/independent_validation.csv first."
        )
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Train the model first with: python src\\train_model.py")

    artifact = joblib.load(MODEL_PATH)
    df = read_csv_robust(DATA)
    text_col = find_col(df, TEXT_ALIASES)
    label_col = find_col(df, ["label", "class", "target", "category", "type"])
    if not text_col or not label_col:
        raise ValueError(f"Need text and label columns. Found: {list(df.columns)}")

    subject_col = find_col(df, ["subject", "title"])
    out = pd.DataFrame({
        "subject": df[subject_col].fillna("").astype(str) if subject_col else "",
        "body": df[text_col].fillna("").astype(str),
        "label": df[label_col].map(label_value)
    }).dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    if out["label"].nunique() < 2:
        raise ValueError("Independent validation data must contain both Safe and Phishing labels.")

    word_vectorizer = artifact.get("word_vectorizer", artifact.get("vectorizer"))
    X_word = word_vectorizer.transform(combine_email_text(out))
    if artifact.get("char_vectorizer") is not None:
        X_text = hstack([X_word, artifact["char_vectorizer"].transform(combine_email_text(out))], format="csr")
    else:
        X_text = X_word

    if artifact.get("uses_numeric_features", True):
        num = build_numeric_features(out)
        X = hstack([X_text, csr_matrix(artifact["scaler"].transform(num))], format="csr")
    else:
        X = X_text

    model = artifact["model"]
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
    else:
        decision = model.decision_function(X)
        score = 1 / (1 + np.exp(-np.clip(decision, -20, 20)))

    metrics = {
        "dataset": os.path.basename(DATA),
        "samples": len(out),
        "safe": int((out["label"] == 0).sum()),
        "phishing": int((out["label"] == 1).sum()),
        "accuracy": accuracy_score(out["label"], pred),
        "precision": precision_score(out["label"], pred, zero_division=0),
        "recall": recall_score(out["label"], pred, zero_division=0),
        "f1": f1_score(out["label"], pred, zero_division=0),
        "roc_auc": roc_auc_score(out["label"], score),
        "pr_auc": average_precision_score(out["label"], score)
    }
    pd.DataFrame([metrics]).to_csv(OUT, index=False)
    print("\nIndependent dataset:", os.path.basename(DATA))
    print("Samples:", len(out), "| Safe:", metrics["safe"], "| Phishing:", metrics["phishing"])
    print(pd.DataFrame([metrics]).to_string(index=False))
    print("\nConfusion matrix [rows=actual, columns=predicted]:")
    print(confusion_matrix(out["label"], pred))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
