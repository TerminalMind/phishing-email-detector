# 🔐 Security Policy

## Supported Versions

This project is primarily an educational cybersecurity and machine learning project.

At this time, security updates are provided for the latest version available in the repository.

| Version        | Supported |
| -------------- | --------- |
| Latest         | ✅ Yes     |
| Older versions | ❌ No      |

---

## 🐛 Reporting a Security Vulnerability

If you discover a security vulnerability, security weakness, or other security-related issue in this project, please report it responsibly.

### Please do not

* Publicly disclose the vulnerability before it has been reviewed.
* Use the vulnerability to access systems or data that do not belong to you.
* Submit malicious code or intentionally harmful payloads.
* Attempt to compromise third-party systems using this project.

### How to Report

Please report security issues privately through the **GitHub Security Advisories** feature, if enabled for this repository.

If GitHub Security Advisories are not available, please contact the repository owner through the contact information provided on the GitHub profile.

When reporting an issue, please include:

* A clear description of the vulnerability.
* Steps required to reproduce the issue.
* The affected file or component.
* Potential security impact.
* Screenshots or logs, if applicable.
* A suggested mitigation or fix, if available.

---

## 🛡️ Security Considerations

This project analyzes email content to identify potential phishing messages.

Users should be aware that:

* The model can produce false positives.
* The model can produce false negatives.
* Machine learning predictions should not be treated as definitive security decisions.
* Suspicious emails should be verified using additional security controls.
* Real passwords, authentication tokens, API keys, or other sensitive information should never be included in test data.
* Do not upload confidential emails or personal information to an untrusted deployment of this project.

---

## 🔑 Sensitive Information

Never commit sensitive information to this repository, including:

* Passwords
* API keys
* Access tokens
* Private keys
* Database credentials
* Authentication cookies
* Personal email data
* Confidential organizational information

Use environment variables or secure secret-management solutions when credentials are required.

---

## 🧪 Responsible Testing

Testing should only be performed on systems, applications, datasets, and email accounts that you own or have explicit permission to test.

This project must not be used to:

* Conduct unauthorized phishing campaigns.
* Steal credentials or personal information.
* Bypass authentication.
* Attack third-party systems.
* Distribute malicious content.

The project is intended for **education, research, and defensive cybersecurity purposes**.

---

## 🔄 Security Updates

Security-related fixes may be released through:

* GitHub commits
* Pull requests
* GitHub Security Advisories
* Updated documentation

Users are encouraged to keep their local copy of the project and its dependencies up to date.

---

## 📦 Dependency Security

The project relies on Python libraries such as Scikit-learn, Pandas, NumPy, and Matplotlib.

Dependencies should be kept reasonably up to date to reduce exposure to known vulnerabilities.

Before installing dependencies, review the project's `requirements.txt` file and use a virtual environment where possible.

---

## ⚠️ Disclaimer

This project is provided for educational and research purposes. No guarantee is made that the phishing detection model will identify every malicious email or prevent phishing attacks.

The author is not responsible for misuse of this software or for damage resulting from unauthorized use.

---

## 📬 Responsible Disclosure

Thank you for helping improve the security of this project.

Security researchers and contributors are encouraged to report vulnerabilities responsibly and provide enough information to reproduce and address the issue.
