# MergerGraphNet

### A Graph Neural Network Approach for Predicting Corporate Mergers from 10-K MD&A Text

**MergerGraphNet** is an AI-driven framework that combines **TF-IDF embeddings**, **Graph Neural Networks (GNNs)**, and **Natural Language Processing (NLP)** to predict corporate mergers using **10-K Management’s Discussion and Analysis (MD&A) sections**.

This project is designed as a **binary classification task**, predicting whether a company will **merge** or **not merge** based on financial disclosures.

## 📊 Result Table

| Model                  | Approach     | Accuracy   | Precision | Recall     | F1         | TP     | TN       | FP      | FN      |
|------------------------|--------------|------------|-----------|------------|------------|--------|----------|---------|---------|
| Gaussian NB            | Report       | 0.5195     | 0.1998    | 0.7252     | 0.3133     | 219    | 819      | 877     | 83      |
| Logistic Regression    | Report       | 0.8493     | 0.6667    | 0.0066     | 0.0131     | 2      | 1695     | 1       | 300     |
| Random Forest          | Report       | 0.8544     | 0.6667    | 0.0728     | 0.1313     | 22     | 1685     | 11      | 280     |
| XGBoost                | Report       | 0.8438     | 0.4242    | 0.0927     | 0.1522     | 28     | 1658     | 38      | 274     |
| GraphLSTMClassifier    | Sentence     | 0.8303     | 0.3238    | 0.1126     | 0.1671     | 34     | 1625     | 71      | 268     |
| **GraphClassifier**    | **Sentence** | **0.7893** | **0.272** | **0.2351** | **0.2522** | **71** | **1506** | **190** | **231** |
| SentenceClassifierLSTM | Sentence     | 0.8233     | 0.2821    | 0.1093     | 0.1575     | 33     | 1612     | 84      | 269     |


## 🚀 Features

- **📊 Graph-Based Representation:** Constructs company-entity relationships from financial reports.
- **📝 TF-IDF Embeddings:** Converts textual disclosures into structured numerical representations.
- **🧠 GNN-Powered Predictions:** Classifies companies as Merge or Not-Merge using graph-based learning.
- **⚡ Scalable & Efficient:** Handles large-scale financial text corpora.

## 🛠 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/CagataySavasli/MergerGraphNet.git
cd MergerGraphNet
```

## 📌 Research Context

MergerGraphNet aims to **bridge financial text mining with graph-based learning**, providing a novel AI-driven approach to **predict corporate mergers** from **structured financial disclosures**.

## 🤝 Contributions

Feel free to open an **issue**, submit a **pull request**, or discuss improvements!

📩 **For inquiries, reach out or create an issue.** 🚀

## 👨‍💻 Developers

- **Ahmet Çağatay Savaşlı** – Developer
- **Prof. Dr. Emre Sefer** – Advisor

*This project was developed within the [OzU Machine Learning in Finance and Bioinformatics Lab](https://ozu-mlfinbio-lab.github.io/).*
