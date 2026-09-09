# 🛡️ Phishing Email Detection using Machine Learning

A machine learning-based cybersecurity project that detects whether an email is **Phishing** or **Safe** by analyzing email content and extracting security-related features.

The project uses **Python and Scikit-learn** to train a classification model on a dataset containing phishing and legitimate emails.

---
# 🛡️ Phishing Email Detection System

🚀 Live Demo (https://phishing-email-detector-exngbwvrmklerggxixrr2r.streamlit.app/)

An ML-powered phishing email detection system built with Python,
Scikit-learn and Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://phishing-email-detector-exngbwvrmklerggxixrr2r.streamlit.app/)

---

## 🚀 Features

* 📧 Detects **Phishing** and **Safe** emails
* 🤖 Machine learning-based classification
* 🔍 Analyzes email text and security-related patterns
* 🔗 Detects suspicious URLs
* ⚠️ Identifies common phishing keywords and patterns
* 📊 Displays model accuracy
* 📈 Generates a confusion matrix
* 📋 Provides classification performance metrics
* 💾 Saves the trained machine learning model
* 🧪 Supports testing with new email messages
* 🛡️ Designed as an educational cybersecurity project

---

## 🧠 How It Works

The system follows a simple machine learning pipeline:

```text
Email Dataset
     ↓
Data Preprocessing
     ↓
Feature Extraction
     ↓
Train/Test Split
     ↓
Machine Learning Model
     ↓
Model Evaluation
     ↓
Phishing / Safe Prediction
```

The model learns patterns from previously classified emails and uses those patterns to classify new email messages.

---

## 📂 Project Structure

```text
phishing-email-detector/
│
├── data/
│   └── phishing_email.csv
│
├── models/
│   └── phishing_model.pkl
│
├── src/
│   ├── train_model.py
│   └── predict.py
│
├── results/
│   └── confusion_matrix.png
│
├── requirements.txt
├── README.md
├── LICENSE
├── CONTRIBUTING.md
└── SECURITY.md
```

> File and folder names may vary depending on your final project structure.

---

## 🛠️ Technologies Used

* **Python**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Natural Language Processing (NLP)**
* **Machine Learning**
* **Cybersecurity**

---

## 📊 Dataset

The model is trained using a dataset containing both:

* **Safe/Legitimate emails**
* **Phishing emails**

The dataset used during development contains approximately **9,480 email records**.

Example distribution:

| Label     |   Records |
| --------- | --------: |
| Safe      |     4,772 |
| Phishing  |     4,708 |
| **Total** | **9,480** |

The relatively balanced dataset helps prevent the model from becoming heavily biased toward one class.

---

## 🔍 Features Analyzed

The project analyzes characteristics of email messages that can help identify phishing attempts.

### Text-Based Features

* Email subject/content
* Common phishing keywords
* Suspicious phrases
* Urgency-related language
* Credential/payment-related terminology

### URL-Based Features

* Number of URLs
* Suspicious URL patterns
* URL-related indicators
* Links commonly associated with phishing behavior

### Classification Features

The extracted features are converted into numerical representations that can be processed by the machine learning algorithm.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/phishing-email-detector.git
```

### 2. Navigate to the Project

```bash
cd phishing-email-detector
```

### 3. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

Run the training script:

```bash
python src/train_model.py
```

The script will:

1. Load the email dataset
2. Preprocess the data
3. Extract relevant features
4. Split the dataset into training and testing sets
5. Train the machine learning model
6. Evaluate the model
7. Display performance metrics
8. Generate a confusion matrix
9. Save the trained model

---

## 🧪 Test the Model

After training, you can use the prediction script to classify a new email:

```bash
python src/predict.py
```

Example:

```text
Enter email text:

URGENT! Your account has been suspended.
Click the link below to verify your account.

Prediction: PHISHING
```

Another example:

```text
Enter email text:

Hi John,

The meeting has been scheduled for tomorrow at 10 AM.

Prediction: SAFE
```

---

## 📈 Model Evaluation

The project evaluates the trained model using standard machine learning metrics.

### Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

### Confusion Matrix

The confusion matrix helps visualize:

```text
                 Predicted
              Safe   Phishing
Actual Safe
Actual Phishing
```

This is particularly useful for understanding false positives and false negatives.

---

## 🔐 Cybersecurity Relevance

Phishing is one of the most common social engineering techniques used by attackers to trick users into:

* Revealing passwords
* Sharing sensitive information
* Clicking malicious links
* Downloading malicious files
* Making fraudulent payments
* Visiting fake login pages

Machine learning can assist in identifying suspicious email patterns and provide an additional layer of email security.

---

## ⚠️ Limitations

This project is intended primarily for **educational and research purposes**.

The model may produce:

* False positives — legitimate emails classified as phishing
* False negatives — phishing emails classified as safe

Attackers can also modify email content to bypass detection systems.

Therefore, the prediction should **not be considered a guaranteed security verdict**.

---

## 🔮 Future Improvements

Possible future improvements include:

* 🌐 Web-based user interface
* 📧 Direct email analysis
* 🔗 Advanced URL reputation checking
* 🧠 Deep learning models
* 📝 Email header analysis
* 🌍 Domain reputation analysis
* 📎 Attachment analysis
* 🚨 Risk score generation
* 📊 Interactive security dashboard
* 🔄 Continuous model retraining
* ☁️ Cloud deployment
* 🔐 Integration with email security systems

---

## 🎯 Project Objectives

The main objectives of this project are:

1. Understand phishing email characteristics.
2. Apply machine learning to cybersecurity.
3. Extract useful features from email content.
4. Build a phishing classification model.
5. Evaluate the model using standard metrics.
6. Demonstrate how ML can support phishing detection.

---

## 👨‍💻 Author

**Shivam Kumar**

MCA – Cybersecurity

This project was developed as part of a cybersecurity and machine learning learning project.

---

## 🤝 Contributing

Contributions are welcome!

If you would like to improve this project:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Test your changes.
5. Submit a Pull Request.

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before contributing.

---

## 🔒 Security

If you discover a security issue or vulnerability related to this project, please follow the instructions in [`SECURITY.md`](SECURITY.md).

---

## 📜 License

This project is licensed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for details.

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

### 🛡️ Disclaimer

This project is created for **educational and cybersecurity research purposes**. It should not be used as the sole mechanism for determining whether an email is malicious. Always verify suspicious emails through additional security controls and trusted sources.
