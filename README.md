# 💊 ADR & DDI Prediction System

> Predict **Adverse Drug Reactions (ADR)** and **Drug-Drug Interactions (DDI)** from plain-text drug names using machine learning and NLP.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/License-Academic-green)

---

## 🧠 What It Does

- 🔍 **Extracts drug names** from plain text using SciSpacy (BC5CDR NER model)
- ⚠️ **Predicts ADRs** — side effects a drug can cause (from SIDER + OFFSIDES)
- 💥 **Predicts DDIs** — dangerous interactions between drug pairs (from TWOSIDES)
- 📊 **Model**: Stacked Ensemble — Random Forest + XGBoost → LightGBM meta-learner

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 98.94% |
| Precision | 99.70% |
| Recall | 55.46% |
| F1-Score | 71.27% |

---

## 📦 Datasets Used

| Source | Purpose |
|---|---|
| SIDER 4.1 | Official drug side effects |
| OFFSIDES (TatonettiLab) | Off-label ADR signals |
| TWOSIDES (TatonettiLab) | Drug-Drug interaction signals |
| DrugBank Vocabulary | Drug name/ID bridge |

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/asthikbandari51/adr-ddi-prediction.git
cd adr-ddi-prediction
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

### 4. Download datasets + models
📁 **[Download from Google Drive](https://drive.google.com/drive/folders/1_sFdtfSrECZrxQ2FdunuM00YfOTe3aqX?usp=drive_link)**

Extract and place:
- All CSV files → `data/processed/`
- All `.pkl` files → `models/`

### 5. Run the app
```bash
streamlit run app.py
```

---

## 🗂️ Project Structure
"adr-ddi-prediction/
├── app.py # Streamlit web UI
├── predict.py # Core prediction engine
├── requirements.txt
├── src/
│ ├── clean_data.py # Data preprocessing
│ ├── merge_data.py # Dataset merging
│ ├── train_adr.py # ADR model training
│ └── train_ddi.py # DDI model training
├── data/processed/ # ← Download from Drive
└── models/ # ← Download from Drive"

---

## 👥 Team

**JBIET — CSE Major Project (2025–26)**  
Team 25 — ADR & DDI Prediction System
