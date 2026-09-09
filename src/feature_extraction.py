"""Feature engineering for the phishing email detector.

Includes language-agnostic structural features so the model can generalize
better across datasets and languages.
"""

import math
import re
import numpy as np
import pandas as pd

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>'\"]+", re.I)
IP_URL_RE = re.compile(r"https?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/|$)", re.I)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
DOMAIN_RE = re.compile(r"@([a-z0-9.-]+\.[a-z]{2,})", re.I)
URL_DOMAIN_RE = re.compile(r"(?:https?://|www\.)([^/\s?#:'\"]+)", re.I)

SUSPICIOUS_KEYWORDS = [
    "urgent", "verify", "verification", "account suspended", "password",
    "login", "click here", "confirm", "security alert", "payment",
    "invoice", "bank", "winner", "prize", "gift card", "limited time",
    "act now", "reset", "credential", "update your account", "refund",
    "unusual activity", "locked", "suspend", "expire", "claim", "bitcoin",
    "wire transfer", "social security", "tax", "otp", "one time password"
]

URGENCY_WORDS = [
    "urgent", "immediately", "now", "asap", "within 24 hours",
    "action required", "last warning", "final notice", "expires", "deadline"
]

SUSPICIOUS_TLDS = {
    "zip", "mov", "click", "top", "xyz", "tk", "ml", "ga", "cf", "gq",
    "work", "support", "live", "buzz", "fit", "loan", "win", "rest"
}


def clean_text(value):
    return "" if pd.isna(value) else str(value).strip()


def combine_email_text(df):
    subject = df.get("subject", pd.Series("", index=df.index)).map(clean_text)
    body = df.get("body", pd.Series("", index=df.index)).map(clean_text)
    return (subject + " " + body).str.replace(r"\s+", " ", regex=True).str.strip()


def count_urls(text):
    return len(URL_RE.findall(text))


def extract_urls(text):
    return URL_RE.findall(text)


def count_ip_urls(text):
    return len(IP_URL_RE.findall(text))


def count_emails(text):
    return len(EMAIL_RE.findall(text))


def count_suspicious_keywords(text):
    low = text.lower()
    return sum(1 for word in SUSPICIOUS_KEYWORDS if word in low)


def count_urgency_words(text):
    low = text.lower()
    return sum(1 for word in URGENCY_WORDS if word in low)


def count_html_tags(value):
    return len(re.findall(r"<[a-z][^>]*>", clean_text(value).lower()))


def html_indicator(value):
    text = clean_text(value).lower()
    return int(any(x in text for x in ["<html", "<a ", "href=", "<form", "<script", "<iframe"]))


def attachment_indicator(value):
    text = clean_text(value).lower()
    return int(bool(text) and text not in {"none", "no", "false", "0", "[]", "nan"})


def domain_from_email(value):
    match = DOMAIN_RE.search(clean_text(value).lower())
    return match.group(1) if match else ""


def sender_reply_mismatch(sender, reply_to):
    s = domain_from_email(sender)
    r = domain_from_email(reply_to)
    return int(bool(s and r and s != r))


def auth_score(value):
    text = clean_text(value).lower()
    if text in {"pass", "passed", "true", "yes", "1"}:
        return 1
    if text in {"fail", "failed", "false", "no", "0"}:
        return -1
    return 0


def url_features(text):
    urls = extract_urls(text)
    if not urls:
        return {
            "avg_url_length": 0.0,
            "max_url_length": 0.0,
            "url_domain_count": 0,
            "url_subdomain_count": 0,
            "url_query_count": 0,
            "url_at_symbol_count": 0,
            "url_hyphen_count": 0,
            "url_encoded_count": 0,
            "suspicious_tld_count": 0,
            "https_url_ratio": 0.0,
            "url_digit_ratio": 0.0,
            "url_entropy_mean": 0.0,
        }

    lengths = [len(u) for u in urls]
    domains = []
    for u in urls:
        match = URL_DOMAIN_RE.search(u)
        if match:
            domains.append(match.group(1).lower())

    def entropy(value):
        if not value:
            return 0.0
        counts = np.bincount(np.frombuffer(value.encode("utf-8", errors="ignore"), dtype=np.uint8), minlength=256)
        probs = counts[counts > 0] / len(value.encode("utf-8", errors="ignore"))
        return float(-(probs * np.log2(probs)).sum())

    tlds = []
    for domain in domains:
        parts = domain.split(".")
        if parts:
            tlds.append(parts[-1])

    return {
        "avg_url_length": float(np.mean(lengths)),
        "max_url_length": float(np.max(lengths)),
        "url_domain_count": len(set(domains)),
        "url_subdomain_count": sum(max(len(d.split(".")) - 2, 0) for d in domains),
        "url_query_count": sum("?" in u or "=" in u for u in urls),
        "url_at_symbol_count": sum("@" in u for u in urls),
        "url_hyphen_count": sum(u.count("-") for u in urls),
        "url_encoded_count": sum(len(re.findall(r"%[0-9a-f]{2}", u, re.I)) for u in urls),
        "suspicious_tld_count": sum(t in SUSPICIOUS_TLDS for t in tlds),
        "https_url_ratio": sum(u.lower().startswith("https://") for u in urls) / len(urls),
        "url_digit_ratio": sum(c.isdigit() for u in urls for c in u) / max(sum(len(u) for u in urls), 1),
        "url_entropy_mean": float(np.mean([entropy(u) for u in urls])),
    }


def build_numeric_features(df):
    text = combine_email_text(df)
    out = pd.DataFrame(index=df.index)

    out["url_count"] = text.map(count_urls)
    out["ip_url_count"] = text.map(count_ip_urls)
    out["email_address_count"] = text.map(count_emails)
    out["suspicious_keyword_count"] = text.map(count_suspicious_keywords)
    out["urgency_word_count"] = text.map(count_urgency_words)
    out["exclamation_count"] = text.str.count("!")
    out["question_count"] = text.str.count(r"\?")
    out["dollar_sign_count"] = text.str.count(r"\$")
    out["text_length"] = text.str.len()
    out["word_count"] = text.str.split().str.len()
    out["uppercase_ratio"] = text.map(
        lambda x: sum(c.isupper() for c in x) / max(sum(c.isalpha() for c in x), 1)
    )
    out["digit_ratio"] = text.map(
        lambda x: sum(c.isdigit() for c in x) / max(len(x), 1)
    )
    out["non_ascii_ratio"] = text.map(
        lambda x: sum(ord(c) > 127 for c in x) / max(len(x), 1)
    )
    out["punctuation_ratio"] = text.map(
        lambda x: sum(not c.isalnum() and not c.isspace() for c in x) / max(len(x), 1)
    )
    out["unique_word_ratio"] = text.map(
        lambda x: len(set(x.lower().split())) / max(len(x.split()), 1)
    )

    url_df = pd.DataFrame([url_features(x) for x in text], index=df.index)
    out = pd.concat([out, url_df], axis=1)

    html = df.get("html", pd.Series("", index=df.index))
    out["html_indicator"] = html.map(html_indicator)
    out["html_tag_count"] = html.map(count_html_tags)
    out["attachment_indicator"] = df.get("attachments", pd.Series("", index=df.index)).map(attachment_indicator)
    out["sender_reply_mismatch"] = [
        sender_reply_mismatch(s, r)
        for s, r in zip(
            df.get("sender", pd.Series("", index=df.index)),
            df.get("reply_to", pd.Series("", index=df.index)),
        )
    ]

    for field in ["spf", "dkim", "dmarc"]:
        out[f"{field}_score"] = df.get(field, pd.Series("", index=df.index)).map(auth_score)

    return out.replace([np.inf, -np.inf], 0).fillna(0)
