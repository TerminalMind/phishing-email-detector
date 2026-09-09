"""Prediction, risk scoring and explainability for the v4 model artifact."""

import os
import math
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix

from .feature_extraction import combine_email_text, build_numeric_features

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "phishing_model.joblib")


def risk_level(score):
    if score < 0.25:
        return "Low"
    if score < 0.50:
        return "Medium"
    if score < 0.75:
        return "High"
    return "Critical"


def predict_email(subject="", body="", sender="", reply_to="", html="",
                  attachments="", spf="", dkim="", dmarc=""):
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    word_vectorizer = artifact.get("word_vectorizer", artifact.get("vectorizer"))
    char_vectorizer = artifact.get("char_vectorizer")
    scaler = artifact["scaler"]

    df = pd.DataFrame([{
        "subject": subject, "body": body, "sender": sender,
        "reply_to": reply_to, "html": html, "attachments": attachments,
        "spf": spf, "dkim": dkim, "dmarc": dmarc
    }])
    text = combine_email_text(df)
    X_word = word_vectorizer.transform(text)
    X_text = X_word if char_vectorizer is None else hstack(
        [X_word, char_vectorizer.transform(text)], format="csr"
    )
    numeric = build_numeric_features(df)

    if artifact.get("uses_numeric_features", True):
        X_num = csr_matrix(scaler.transform(numeric))
        X = hstack([X_text, X_num], format="csr")
    else:
        X_num = None
        X = X_text

    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(X)[0, 1])
    else:
        decision = float(model.decision_function(X)[0])
        score = 1.0 / (1.0 + math.exp(-max(min(decision, 20), -20)))

    label = "Phishing" if score >= 0.5 else "Safe"
    explanations = []
    feature_names = artifact.get("feature_names", [])

    if hasattr(model, "coef_"):
        coef = model.coef_[0]
        values = X.toarray()[0]
        contributions = values * coef
        for i in np.argsort(np.abs(contributions))[::-1][:12]:
            if i < len(feature_names) and abs(contributions[i]) > 0:
                explanations.append({
                    "feature": feature_names[i],
                    "contribution": float(contributions[i]),
                    "direction": "Phishing" if contributions[i] > 0 else "Safe"
                })
    elif hasattr(model, "feature_log_prob_"):
        values = X_text.toarray()[0]
        coef = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        for i in np.argsort(np.abs(values * coef))[::-1][:12]:
            contribution = values[i] * coef[i]
            if i < len(feature_names) and abs(contribution) > 0:
                explanations.append({
                    "feature": feature_names[i],
                    "contribution": float(contribution),
                    "direction": "Phishing" if contribution > 0 else "Safe"
                })

    return {
        "label": label,
        "probability": score,
        "risk_level": risk_level(score),
        "risk_score": round(score * 100, 2),
        "score_type": artifact.get("score_type", "probability"),
        "features": numeric.iloc[0].to_dict(),
        "explanations": explanations
    }
