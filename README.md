# 🛡️ PhishGuard — Phishing URL Detection

A machine learning system that classifies URLs as **phishing** or **legitimate** using extracted URL-based features.

> Cybersecurity ML Project — Built with Python, scikit-learn, and Streamlit.

---

## 📁 Project Structure

```
phishguard/
├── data/
│   ├── raw/                  # Original downloaded datasets
│   ├── processed/            # Cleaned & split datasets
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── README.md             # Data dictionary
├── src/
│   ├── preprocessing/        # Data cleaning scripts
│   ├── feature_engineering/  # Feature extraction
│   ├── models/               # Model definitions
│   ├── training/             # Training scripts
│   ├── evaluation/           # Metrics & plots
│   └── utils/                # Helper functions
├── notebooks/
│   ├── EDA.ipynb
│   ├── model_training.ipynb
│   └── results_analysis.ipynb
├── model/
│   └── phishing_model.pkl    # Saved trained model (place here)
├── tests/                    # Unit tests
├── app.py                    # Streamlit demo app  ← YOU ARE HERE
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/phishguard.git
cd phishguard
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the demo app

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`

---

## 🤖 Using the Real Trained Model

By default the app runs in **demo mode** using a heuristic scorer.

To use the real trained model:

1. Get the file `phishing_model.pkl` from Person 4 (model training)
2. Place it inside the `model/` folder:
   ```
   model/phishing_model.pkl
   ```
3. Restart the app — it auto-detects the file on startup

The model must be a scikit-learn compatible estimator saved with `pickle` or `joblib`.

---

## 🔍 Features Extracted

The app extracts 16 URL-based features across three categories:

| Category | Features |
|---|---|
| Address bar | IP address in URL, URL length, `@` symbol, double slash redirect, dash in domain, subdomain count, HTTPS, redirect |
| Domain | Domain length, suspicious TLD, digit count, special characters |
| Path / Query | Path length, query length, login keywords, brand keywords |

---

## 📊 Datasets Used

| Dataset | Source | Purpose |
|---|---|---|
| Phishing URLs | [Kaggle — Phishing Dataset for ML](https://www.kaggle.com/) | Phishing samples |
| Legitimate URLs | [UNB URL-2016](https://www.unb.ca/cic/datasets/url-2016.html) | Benign samples |
| PhishTank | [phishtank.com](https://www.phishtank.com/) | Additional phishing validation |

---

## 👥 Team Contributions

| Person | Role | Contribution |
|---|---|---|
| Person 1 | Project Lead | Problem definition, literature review, report assembly |
| Person 2 | Data | Dataset collection, preprocessing, EDA setup |
| Person 3 | Features | Feature engineering, visualizations, EDA |
| Person 4 | Models | Model training, hyperparameter tuning, evaluation |
| Person 5 | Demo & Docs | Streamlit app, repo organization, README, demo video |

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

---

## 📝 License

For academic use only. Dataset licenses apply — see `data/README.md`.
