import pandas as pd
import numpy as np
import os
import joblib
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROCESSED = os.path.join('data', 'processed')
MODELS = os.path.join('models')
os.makedirs(MODELS, exist_ok=True)

print("=" * 60)
print("  DDI Model Training")
print("=" * 60)

# ─────────────────────────────────────────
# LOAD DATA — sample 500k rows (RAM safe)
# ─────────────────────────────────────────
print("\n[1/7] Loading DDI training data (500k sample)...")
ddi = pd.read_csv(
    os.path.join(PROCESSED, 'ddi_training_data.csv'),
    nrows=500_000
)
print(f"    Loaded: {len(ddi)} rows")
print(f"    Unique interaction effects: {ddi['interaction_effect'].nunique()}")

# ─────────────────────────────────────────
# FILTER — keep top 30 interaction types
# ─────────────────────────────────────────
print("\n[2/7] Filtering top 30 interaction types...")
top_30 = ddi['interaction_effect'].value_counts().head(30).index.tolist()
ddi = ddi[ddi['interaction_effect'].isin(top_30)]
print(f"    Rows after filter: {len(ddi)}")

# ─────────────────────────────────────────
# ENCODE
# ─────────────────────────────────────────
print("\n[3/7] Encoding features and labels...")

le_drug = LabelEncoder()
all_drugs = pd.concat([ddi['drug_1'], ddi['drug_2']]).unique()
le_drug.fit(all_drugs)
joblib.dump(le_drug, os.path.join(MODELS, 'ddi_drug_encoder.pkl'))

ddi['drug_1_enc'] = le_drug.transform(ddi['drug_1'])
ddi['drug_2_enc'] = le_drug.transform(ddi['drug_2'])

le_label = LabelEncoder()
ddi['label'] = le_label.fit_transform(ddi['interaction_effect'])
joblib.dump(le_label, os.path.join(MODELS, 'ddi_label_encoder.pkl'))

# Save label mapping for readable predictions
label_map = dict(enumerate(le_label.classes_))
import json
with open(os.path.join(MODELS, 'ddi_label_map.json'), 'w') as f:
    json.dump(label_map, f, indent=2)

# Feature columns
feature_cols = ['drug_1_enc', 'drug_2_enc', 'mol_weight_1', 'h_acceptors_1',
                'h_donors_1', 'logp_1', 'mol_weight_2', 'h_acceptors_2',
                'h_donors_2', 'logp_2', 'PRR', 'freq']

X = ddi[feature_cols].values
y = ddi['label'].values
print(f"    Features shape: {X.shape}")
print(f"    Labels: {len(set(y))} classes")

# ─────────────────────────────────────────
# SMOTE — fix class imbalance
# ─────────────────────────────────────────
print("\n[4/7] Applying SMOTE to fix class imbalance...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_res, y_res = smote.fit_resample(X, y)
print(f"    After SMOTE: {X_res.shape[0]} rows")

# ─────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────
print("\n[5/7] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)
print(f"    Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

del ddi, X_res, y_res
gc.collect()

# ─────────────────────────────────────────
# TRAIN STACKED MODEL
# ─────────────────────────────────────────
print("\n[6/7] Training stacked model...")

base_models = [
    ('rf', RandomForestClassifier(
        n_estimators=100, class_weight='balanced',
        random_state=42, n_jobs=-1)),
    ('xgb', XGBClassifier(
        n_estimators=100, eval_metric='mlogloss',
        random_state=42, n_jobs=-1)),
    ('lr', LogisticRegression(
        max_iter=1000, class_weight='balanced',
        random_state=42, n_jobs=-1)),
]

ddi_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LGBMClassifier(n_estimators=100, random_state=42),
    cv=3,
    n_jobs=-1
)

ddi_model.fit(X_train, y_train)
print("    Training complete!")

# ─────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────
print("\n[7/7] Evaluating model...")
y_pred = ddi_model.predict(X_test)

print(classification_report(y_test, y_pred,
      target_names=le_label.classes_, zero_division=0))

f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
print(f"    Weighted F1 Score: {f1:.4f}")

# ─────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────
joblib.dump(ddi_model, os.path.join(MODELS, 'ddi_model.pkl'))
print("\n    ✅ ddi_model.pkl saved")

print("\n" + "=" * 60)
print("  DDI MODEL TRAINING COMPLETE!")
print("=" * 60)