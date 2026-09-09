import os
import io
import pandas as pd
import streamlit as st

from src.predict import predict_email
from src.report import build_pdf

st.set_page_config(page_title="Phishing Email Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ Phishing Email Detection System")
st.caption("Machine-learning email analysis with explainability, risk scoring and evaluation.")

MODEL_PATH = os.path.join("models", "phishing_model.joblib")
if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Train it first with: python src\\train_model.py")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Email Analyzer", "📊 Evaluation", "📈 Dataset & Explainability", "ℹ️ About"
])

with tab1:
    st.subheader("Analyze an email")
    with st.form("email_form"):
        c1, c2 = st.columns(2)
        with c1:
            subject = st.text_input("Subject")
            sender = st.text_input("Sender")
            reply_to = st.text_input("Reply-To")
        with c2:
            spf = st.selectbox("SPF", ["", "pass", "fail"])
            dkim = st.selectbox("DKIM", ["", "pass", "fail"])
            dmarc = st.selectbox("DMARC", ["", "pass", "fail"])

        body = st.text_area("Email body", height=240)
        html = st.text_area("HTML content (optional)", height=100)
        attachments = st.text_input("Attachments (optional)")
        submitted = st.form_submit_button("🚀 Analyze Email", use_container_width=True)

    if submitted:
        if not subject.strip() and not body.strip():
            st.warning("Enter at least a subject or body.")
        else:
            result = predict_email(
                subject=subject, body=body, sender=sender, reply_to=reply_to,
                html=html, attachments=attachments, spf=spf, dkim=dkim, dmarc=dmarc
            )
            st.session_state["last_result"] = result
            st.session_state["last_email"] = {
                "subject": subject, "sender": sender, "reply_to": reply_to
            }

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        email = st.session_state["last_email"]

        a, b, c = st.columns(3)
        a.metric("Classification", result["label"])
        b.metric("Risk Level", result["risk_level"])
        c.metric("Risk Score", f'{result["risk_score"]:.1f}%')

        if result["risk_level"] == "Critical":
            st.error("🚨 CRITICAL: Treat this email as highly suspicious. Do not click links or provide credentials.")
        elif result["risk_level"] == "High":
            st.error("⚠️ HIGH: Verify the sender and links using an independent channel.")
        elif result["risk_level"] == "Medium":
            st.warning("⚠️ MEDIUM: Review indicators before taking action.")
        else:
            st.success("✅ LOW: No strong phishing signal was detected by the model.")

        st.subheader("Security indicators")
        f = result["features"]
        indicator_df = pd.DataFrame([
            ["URLs", f["url_count"]], ["IP-based URLs", f["ip_url_count"]],
            ["Suspicious keywords", f["suspicious_keyword_count"]],
            ["Urgency words", f["urgency_word_count"]],
            ["Email addresses", f["email_address_count"]],
            ["Sender/Reply-To mismatch", f["sender_reply_mismatch"]],
            ["HTML indicator", f["html_indicator"]],
            ["Attachments", f["attachment_indicator"]],
            ["SPF score", f["spf_score"]], ["DKIM score", f["dkim_score"]], ["DMARC score", f["dmarc_score"]]
        ], columns=["Indicator", "Value"])
        st.dataframe(indicator_df, use_container_width=True, hide_index=True)

        st.subheader("Why the model flagged this email")
        if result["explanations"]:
            exp = pd.DataFrame(result["explanations"])
            exp["contribution"] = exp["contribution"].round(4)
            st.dataframe(exp[["feature", "direction", "contribution"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("Feature-level explanation is not available for the selected model.")

        report_pdf = build_pdf(result, email["subject"], email["sender"], email["reply_to"])
        st.download_button(
            "📄 Download PDF Report", data=report_pdf,
            file_name="phishing_detection_report.pdf", mime="application/pdf"
        )

        report_csv = pd.DataFrame([{
            "subject": email["subject"], "sender": email["sender"],
            "reply_to": email["reply_to"], "classification": result["label"],
            "risk_level": result["risk_level"], "risk_score": result["risk_score"],
            **result["features"]
        }]).to_csv(index=False)
        st.download_button(
            "📥 Download CSV Report", data=report_csv,
            file_name="phishing_detection_report.csv", mime="text/csv"
        )

with tab2:
    st.subheader("Model evaluation")
    cv_path = os.path.join("outputs", "cross_validation_summary.csv")
    holdout_path = os.path.join("outputs", "holdout_model_comparison.csv")

    if os.path.exists(cv_path):
        cv = pd.read_csv(cv_path)
        st.write("### 5-fold cross-validation")
        st.dataframe(cv.style.format(
            {c: "{:.3f}" for c in cv.columns if c != "model"}
        ), use_container_width=True, hide_index=True)
    else:
        st.info("Run the training pipeline to generate cross-validation results.")

    if os.path.exists(holdout_path):
        st.write("### Independent holdout test from the training dataset")
        holdout = pd.read_csv(holdout_path)
        st.dataframe(holdout.style.format(
            {c: "{:.3f}" for c in holdout.columns if c != "model"}
        ), use_container_width=True, hide_index=True)

    for filename, title in [
        ("roc_curves.png", "ROC Curves"),
        ("precision_recall_curves.png", "Precision-Recall Curves"),
        ("confusion_matrix_logistic_regression.png", "Logistic Regression Confusion Matrix"),
        ("confusion_matrix_complement_naive_bayes.png", "Complement Naive Bayes Confusion Matrix"),
        ("confusion_matrix_linear_svm.png", "Linear SVM Confusion Matrix"),
    ]:
        path = os.path.join("outputs", filename)
        if os.path.exists(path):
            st.image(path, caption=title, use_container_width=True)

    ext_path = os.path.join("outputs", "independent_validation_results.csv")
    if os.path.exists(ext_path):
        st.write("### Independent external validation")
        st.dataframe(pd.read_csv(ext_path), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Dataset statistics")
    data_path = os.path.join("data", "clean_emails.csv")
    if os.path.exists(data_path):
        ds = pd.read_csv(data_path)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total emails", f"{len(ds):,}")
        c2.metric("Safe", f"{(ds['label'] == 0).sum():,}")
        c3.metric("Phishing", f"{(ds['label'] == 1).sum():,}")
        st.bar_chart(ds["label"].map({0: "Safe", 1: "Phishing"}).value_counts())
    else:
        st.info("Run dataset cleaning first.")

    fi_path = os.path.join("outputs", "top_features.csv")
    if os.path.exists(fi_path):
        st.subheader("Top model features")
        fi = pd.read_csv(fi_path)
        st.dataframe(fi.head(25), use_container_width=True, hide_index=True)

with tab4:
    st.markdown("""
### Project capabilities

- Dataset cleaning and duplicate removal
- Stratified 80/20 holdout evaluation
- 5-fold stratified cross-validation
- Logistic Regression, Complement Naive Bayes and Linear SVM
- TF-IDF word/bigram features
- URL, IP URL, urgency, keyword and structural indicators
- Sender/Reply-To mismatch
- SPF/DKIM/DMARC indicators
- HTML and attachment indicators
- ROC-AUC and PR-AUC
- Confusion matrices
- Model feature explainability
- Low / Medium / High / Critical risk levels
- PDF and CSV detection reports
- Dataset statistics
- Optional independent validation dataset

**Important:** Model scores are probabilistic assessments. They are not proof that an email is malicious.
""")
