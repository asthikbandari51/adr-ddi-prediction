# ADR & DDI Prediction System
Predicts Adverse Drug Reactions and Drug-Drug Interactions
using SIDER, OFFSIDES, TWOSIDES, and DrugBank.

Stack: Python · SciSpacy · XGBoost · LightGBM · Streamlit

## Setup Instructions

1. Clone: `git clone https://github.com/asthikbandari51/adr-ddi-prediction.git`
2. Install: `pip install -r requirements.txt`
3. Download processed data + models from Google Drive: **[https://drive.google.com/drive/folders/1_sFdtfSrECZrxQ2FdunuM00YfOTe3aqX?usp=sharing](https://drive.google.com/drive/folders/1_sFdtfSrECZrxQ2FdunuM00YfOTe3aqX?usp=sharing)**
4. Extract so your folder looks like:
   - `data/processed/*.csv`
   - `models/*.pkl`
5. Run: `streamlit run app.py`

## Model Performance

Accuracy: 98.94% | Precision: 99.70% | Recall: 55.46% | F1: 71.27%

## Dataset Sources

- SIDER 4.1, OFFSIDES, TWOSIDES (TatonettiLab)
- DrugBank Vocabulary (Open Data)
