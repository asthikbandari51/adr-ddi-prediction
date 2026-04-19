import pandas as pd
import numpy as np
import os
import joblib
import gc
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROCESSED = os.path.join('data', 'processed')
MODELS = os.path.join('models')
os.makedirs(MODELS, exist_ok=True)

print("=" * 60)
print("  ADR Model Training")
print("=" * 60)

# ─────────────────────────────────────────
# LOAD & PREPARE DATA
# ─────────────────────────────────────────
print("\n[1/6] Loading ADR training data...")
adr = pd.read_csv(os.path.join(PROCESSED, 'adr_training_data.csv'))

# Keep only top 50 most common side effects
# (avoids extremely rare classes that hurt model accuracy)
top_50_se = adr['side_effect'].value_counts().head(50).index.tolist()
adr = adr[adr['side_effect'].isin(top_50_se)]
print(f"    Rows after filtering top 50 side effects: {len(adr)}")

# Group side effects per drug — one row per drug with list of side effects
drug_features = adr.drop_duplicates(subset='drug_name')[
    ['drug_name', 'drugbank_id', 'mol_weight',
     'h_acceptors', 'h_donors', 'logp']
].reset_index(drop=True)

drug_se = adr.groupby('drug_name')['side_effect'].apply(list).reset_index()
drug_se.columns = ['drug_name', 'side_effects']

df = drug_features.merge(drug_se, on='drug_name', how='inner')
print(f"    Unique drugs for training: {len(df)}")

# ─────────────────────────────────────────
# ENCODE FEATURES & LABELS
# ─────────────────────────────────────────
print("\n[2/6] Encoding features and labels...")

# Encode drug name as numeric feature
le_drug = LabelEncoder()
df['drug_enc'] = le_drug.fit_transform(df['drug_name'])

# Multi-label binarize side effects
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['side_effects'])

# Feature matrix
X = df[['drug_enc', 'mol_weight', 'h_acceptors',
        'h_donors', 'logp']].values

print(f"    Feature matrix shape: {X.shape}")
print(f"    Label matrix shape: {y.shape}")
print(f"    Side effects modelled: {len(mlb.classes_)}")

# ─────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────
print("\n[3/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"    Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ─────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────
print("\n[4/6] Training stacked model (this may take a few minutes)...")

# Stacked ensemble: RF + XGB → LightGBM meta
base_models = [
    ('rf', RandomForestClassifier(
        n_estimators=100, class_weight='balanced',
        random_state=42, n_jobs=-1)),
    ('xgb', XGBClassifier(
        n_estimators=100, eval_metric='logloss',
        random_state=42, n_jobs=-1)),
]

meta_model = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)

stacked = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=3,
    n_jobs=-1
)

# Wrap in MultiOutputClassifier for multi-label prediction
adr_model = MultiOutputClassifier(stacked, n_jobs=-1)
adr_model.fit(X_train, y_train)
print("    Training complete!")

# ─────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────
print("\n[5/6] Evaluating model...")
y_pred = adr_model.predict(X_test)

f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)

print(f"    F1 Score (macro): {f1_macro:.4f}")
print(f"    F1 Score (micro): {f1_micro:.4f}")

# ─────────────────────────────────────────
# SAVE MODEL & ENCODERS
# ─────────────────────────────────────────
print("\n[6/6] Saving model and encoders...")
joblib.dump(adr_model,  os.path.join(MODELS, 'adr_model.pkl'))
joblib.dump(mlb,        os.path.join(MODELS, 'adr_mlb.pkl'))
joblib.dump(le_drug,    os.path.join(MODELS, 'adr_drug_encoder.pkl'))

print("    ✅ adr_model.pkl saved")
print("    ✅ adr_mlb.pkl saved")
print("    ✅ adr_drug_encoder.pkl saved")

print("\n" + "=" * 60)
print("  ADR MODEL TRAINING COMPLETE!")
print("=" * 60)