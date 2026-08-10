<div align="center">

# 📰 Fake News Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*NLP-powered misinformation classifier using TF-IDF features and Logistic Regression to distinguish fake from real news.*

</div>

---

## 📋 Overview

A machine learning pipeline that combats misinformation by automatically classifying news articles as **real** or **fake**. The system combines article titles and body text, applies TF-IDF vectorization, and uses Logistic Regression for binary classification with high accuracy.

## 🏗️ Pipeline

```
News Article (Title + Body)
    ↓
Text Preprocessing (combine title & text, lowercase, clean)
    ↓
TF-IDF Vectorization (stop words removal, frequency filtering)
    ↓
Logistic Regression Classifier (max_iter=1000)
    ↓
Prediction: REAL / FAKE
```

## ✨ Features

- 🔍 **Title + Body Fusion** — Combines headline and article text for richer features
- 📊 **TF-IDF with Smart Filtering** — `max_df=0.95`, `min_df=5` to filter noise
- 🎯 **Stratified Train/Test Split** — Ensures balanced class distribution
- 📈 **Full Evaluation Suite** — Accuracy, Classification Report, Confusion Matrix
- ⚡ **Fast Inference** — Classify new articles in milliseconds

## 🚀 Quick Start

```bash
git clone https://github.com/Izumi6/Fake-News-Detection-System.git
cd Fake-News-Detection-System
pip install pandas scikit-learn
python fake_news_detection.py
```

## 🛠️ Tech Stack

`Python` · `Scikit-learn` · `TF-IDF` · `Logistic Regression` · `Pandas`

## 👤 Author

**Suyash Vakhariya** — [suyashvakhariya.com](https://suyashvakhariya.com) · [LinkedIn](https://www.linkedin.com/in/suyashvakhariya) · [GitHub](https://github.com/Izumi6)
