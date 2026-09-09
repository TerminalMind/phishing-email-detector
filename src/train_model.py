"""Training, 5-fold CV, model comparison, explainability and final persistence."""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import hstack, csr_matrix
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

sys.path.append(os.path.dirname(__file__))
from feature_extraction import combine_email_text, build_numeric_features

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "clean_emails.csv")
MODEL_DIR = os.path.join(BASE, "models")
OUT_DIR = os.path.join(BASE, "outputs")
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_model.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA)
    df["subject"] = df.get("subject", "").fillna("").astype(str)
    df["body"] = df.get("body", "").fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    df = df[(df["subject"] + df["body"]).str.len() >= 5]
    df["dedup_text"] = (df["subject"] + " " + df["body"]).str.lower().str.replace(r"\s+", " ", regex=True)
    return df.drop_duplicates("dedup_text").drop(columns=["dedup_text"]).reset_index(drop=True)


def make_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2500, class_weight="balanced", C=2.0, random_state=RANDOM_STATE
        ),
        "Complement Naive Bayes": ComplementNB(alpha=0.1),
        "Linear SVM": LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE)
    }


def score_model(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def build_features(train_df, eval_df):
    # Word n-grams capture semantic phrases; character n-grams capture spelling,
    # obfuscation and cross-language patterns more robustly.
    word_vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
        min_df=2, max_df=0.98, sublinear_tf=True, max_features=50000,
        dtype=np.float32
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb", lowercase=True, ngram_range=(3, 5),
        min_df=2, max_df=0.995, sublinear_tf=True, max_features=50000,
        dtype=np.float32
    )

    train_text = combine_email_text(train_df)
    eval_text = combine_email_text(eval_df)

    Xtr_word = word_vectorizer.fit_transform(train_text)
    Xev_word = word_vectorizer.transform(eval_text)
    Xtr_char = char_vectorizer.fit_transform(train_text)
    Xev_char = char_vectorizer.transform(eval_text)

    Xtr_text = hstack([Xtr_word, Xtr_char], format="csr")
    Xev_text = hstack([Xev_word, Xev_char], format="csr")

    scaler = StandardScaler()
    tr_num = build_numeric_features(train_df)
    ev_num = build_numeric_features(eval_df)
    Xtr_num = csr_matrix(scaler.fit_transform(tr_num))
    Xev_num = csr_matrix(scaler.transform(ev_num))

    return {
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "scaler": scaler,
        "Xtr_text": Xtr_text,
        "Xev_text": Xev_text,
        "Xtr_num": Xtr_num,
        "Xev_num": Xev_num,
        "num_columns": list(tr_num.columns),
        "word_count": Xtr_word.shape[1],
        "char_count": Xtr_char.shape[1]
    }


def cross_validate(df):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    records = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, df["label"]), start=1):
        tr, va = df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()
        ytr, yva = tr["label"].values, va["label"].values
        f = build_features(tr, va)
        Xtr = hstack([f["Xtr_text"], f["Xtr_num"]], format="csr")
        Xva = hstack([f["Xev_text"], f["Xev_num"]], format="csr")

        ros = RandomOverSampler(random_state=RANDOM_STATE)
        Xres, yres = ros.fit_resample(Xtr, ytr)

        for name, base_model in make_models().items():
            model = clone(base_model)
            if name == "Complement Naive Bayes":
                model.fit(Xres[:, :f["Xtr_text"].shape[1]], yres)
                Xeval = f["Xev_text"]
            else:
                model.fit(Xres, yres)
                Xeval = Xva

            pred = model.predict(Xeval)
            score = score_model(model, Xeval)
            records.append({
                "fold": fold, "model": name,
                "accuracy": accuracy_score(yva, pred),
                "precision": precision_score(yva, pred, zero_division=0),
                "recall": recall_score(yva, pred, zero_division=0),
                "f1": f1_score(yva, pred, zero_division=0),
                "roc_auc": roc_auc_score(yva, score),
                "pr_auc": average_precision_score(yva, score)
            })

    cv_df = pd.DataFrame(records)
    cv_df.to_csv(os.path.join(OUT_DIR, "cross_validation_results.csv"), index=False)
    summary = cv_df.groupby("model").agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std")
    ).reset_index().sort_values(["f1_mean", "roc_auc_mean"], ascending=False)
    summary.to_csv(os.path.join(OUT_DIR, "cross_validation_summary.csv"), index=False)
    return summary


def feature_names_for(f, uses_numeric):
    names = list(f["word_vectorizer"].get_feature_names_out())
    names += [f"char:{x}" for x in f["char_vectorizer"].get_feature_names_out()]
    if uses_numeric:
        names += list(f["num_columns"])
    return names


def save_feature_importance(model, model_name, feature_names):
    importance = None
    if model_name in {"Logistic Regression", "Linear SVM"}:
        importance = model.coef_[0]
    elif model_name == "Complement Naive Bayes":
        importance = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        feature_names = feature_names[:len(importance)]

    if importance is None:
        return
    fi = pd.DataFrame({"feature": feature_names, "importance": importance})
    fi["abs_importance"] = fi["importance"].abs()
    fi.sort_values("abs_importance", ascending=False).head(75).to_csv(
        os.path.join(OUT_DIR, "top_features.csv"), index=False
    )


def fit_final_and_test(df):
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=RANDOM_STATE, stratify=df["label"]
    )
    f = build_features(train_df, test_df)
    Xtr = hstack([f["Xtr_text"], f["Xtr_num"]], format="csr")
    Xte = hstack([f["Xev_text"], f["Xev_num"]], format="csr")
    ytr, yte = train_df["label"].values, test_df["label"].values

    ros = RandomOverSampler(random_state=RANDOM_STATE)
    Xres, yres = ros.fit_resample(Xtr, ytr)

    results, curves = [], {}
    for name, model in make_models().items():
        if name == "Complement Naive Bayes":
            model.fit(Xres[:, :f["Xtr_text"].shape[1]], yres)
            Xeval = f["Xev_text"]
        else:
            model.fit(Xres, yres)
            Xeval = Xte
        pred = model.predict(Xeval)
        score = score_model(model, Xeval)
        results.append({
            "model": name,
            "accuracy": accuracy_score(yte, pred),
            "precision": precision_score(yte, pred, zero_division=0),
            "recall": recall_score(yte, pred, zero_division=0),
            "f1": f1_score(yte, pred, zero_division=0),
            "roc_auc": roc_auc_score(yte, score),
            "pr_auc": average_precision_score(yte, score)
        })
        curves[name] = (roc_curve(yte, score), precision_recall_curve(yte, score))

        cm = confusion_matrix(yte, pred)
        plt.figure(figsize=(5.5, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Safe", "Phishing"], yticklabels=["Safe", "Phishing"])
        plt.title(f"Confusion Matrix — {name}")
        plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"confusion_matrix_{name.lower().replace(' ', '_')}.png"), dpi=160)
        plt.close()

    results_df = pd.DataFrame(results).sort_values(["f1", "roc_auc"], ascending=False)
    results_df.to_csv(os.path.join(OUT_DIR, "holdout_model_comparison.csv"), index=False)

    plt.figure(figsize=(7, 5))
    for name, ((fpr, tpr, _), _) in curves.items():
        auc = results_df.loc[results_df["model"] == name, "roc_auc"].iloc[0]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_curves.png"), dpi=160); plt.close()

    plt.figure(figsize=(7, 5))
    for name, (_, (precision, recall, _)) in curves.items():
        ap = results_df.loc[results_df["model"] == name, "pr_auc"].iloc[0]
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curves"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "precision_recall_curves.png"), dpi=160); plt.close()

    cv_summary = cross_validate(df)
    best_name = cv_summary.iloc[0]["model"]

    # Refit the selected model on the complete training corpus.
    full = build_features(df, df)
    Xfull = hstack([full["Xtr_text"], full["Xtr_num"]], format="csr")
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    Xfull_res, yfull_res = ros.fit_resample(Xfull, df["label"].values)
    best_model = make_models()[best_name]

    if best_name == "Complement Naive Bayes":
        best_model.fit(Xfull_res[:, :full["Xtr_text"].shape[1]], yfull_res)
        uses_numeric = False
    else:
        best_model.fit(Xfull_res, yfull_res)
        uses_numeric = True

    feature_names = feature_names_for(full, uses_numeric)
    save_feature_importance(best_model, best_name, feature_names)

    # Store a transparent score description. LinearSVC uses a decision score;
    # the dashboard should call it a risk score, not a calibrated probability.
    artifact = {
        "model": best_model,
        "word_vectorizer": full["word_vectorizer"],
        "char_vectorizer": full["char_vectorizer"],
        "scaler": full["scaler"],
        "numeric_columns": full["num_columns"],
        "uses_numeric_features": uses_numeric,
        "model_name": best_name,
        "risk_thresholds": {"low": 0.25, "medium": 0.50, "high": 0.75},
        "feature_names": feature_names,
        "word_feature_count": full["word_count"],
        "char_feature_count": full["char_count"],
        "score_type": "probability" if hasattr(best_model, "predict_proba") else "sigmoid_decision_score",
        "version": "4.0"
    }
    # Backward-compatible alias for older scripts.
    artifact["vectorizer"] = full["word_vectorizer"]
    joblib.dump(artifact, MODEL_PATH)

    return results_df, cv_summary, best_name


def main():
    df = load_data()
    print(f"Dataset rows: {len(df):,}")
    print(df["label"].map({0: "Safe", 1: "Phishing"}).value_counts())
    results_df, cv_summary, best_name = fit_final_and_test(df)
    print("\n=== 5-Fold CV Summary ===")
    print(cv_summary.to_string(index=False))
    print("\n=== Holdout Test ===")
    print(results_df.to_string(index=False))
    print(f"\nBest deployment model: {best_name}")
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
