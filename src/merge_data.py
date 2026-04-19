import pandas as pd
import gc
import os

RAW = os.path.join('data', 'raw')
PROCESSED = os.path.join('data', 'processed')

print("=" * 60)
print("  ADR-DDI Project — Data Merging Script")
print("=" * 60)


# ─────────────────────────────────────────
# LOAD BRIDGE FILES
# ─────────────────────────────────────────
print("\n[Loading bridge files...]")

drug_names = pd.read_csv(os.path.join(PROCESSED, 'drug_names_clean.csv'))
vocab = pd.read_csv(os.path.join(PROCESSED, 'drugbank_vocab_clean.csv'))
lipinski = pd.read_csv(os.path.join(PROCESSED, 'lipinski_clean.csv'))

drug_names['drug_name'] = drug_names['drug_name'].str.lower().str.strip()
vocab['drug_name'] = vocab['drug_name'].str.lower().str.strip()

# ← FIX: rename stitch_id → stitch_flat_id to match all SIDER files
drug_names.rename(columns={'stitch_id': 'stitch_flat_id'}, inplace=True)

print(f"    drug_names: {len(drug_names)} entries")
print(f"    vocab:      {len(vocab)} entries")
print(f"    lipinski:   {len(lipinski)} entries")


# ─────────────────────────────────────────
# MERGE 1: ADR TRAINING DATA
# ─────────────────────────────────────────
print("\n[1/2] Building ADR training dataset...")

meddra_se   = pd.read_csv(os.path.join(PROCESSED, 'meddra_se_clean.csv'))
meddra_freq = pd.read_csv(os.path.join(PROCESSED, 'meddra_freq_clean.csv'))
offsides    = pd.read_csv(os.path.join(PROCESSED, 'offsides_clean.csv'))
indications = pd.read_csv(os.path.join(PROCESSED, 'indications_clean.csv'))

# ── Step 1: Join meddra_se with drug_names on stitch_flat_id
meddra_se = meddra_se.merge(drug_names, on='stitch_flat_id', how='inner')
meddra_se = meddra_se[['drug_name', 'side_effect_name']]
meddra_se.rename(columns={'side_effect_name': 'side_effect'}, inplace=True)
meddra_se['source'] = 'sider'
print(f"    meddra_se after join: {len(meddra_se)} rows")

# ── Step 2: Join freq data and merge into meddra_se
meddra_freq = meddra_freq.merge(drug_names, on='stitch_flat_id', how='inner')
meddra_freq = meddra_freq[['drug_name', 'freq_lower', 'side_effect_name']]
meddra_freq.rename(columns={'side_effect_name': 'side_effect',
                             'freq_lower': 'frequency'}, inplace=True)

meddra_se = meddra_se.merge(
    meddra_freq[['drug_name', 'side_effect', 'frequency']],
    on=['drug_name', 'side_effect'],
    how='left'
)
meddra_se['frequency'] = meddra_se['frequency'].fillna(0.0)

# ── Step 3: Prepare OFFSIDES
offsides = offsides[['drug_name', 'side_effect', 'freq']]
offsides.rename(columns={'freq': 'frequency'}, inplace=True)
offsides['source'] = 'offsides'
print(f"    offsides rows: {len(offsides)}")

# ── Step 4: Combine SIDER + OFFSIDES
adr_combined = pd.concat([meddra_se, offsides], ignore_index=True)
adr_combined.drop_duplicates(subset=['drug_name', 'side_effect'], inplace=True)
print(f"    Combined ADR rows: {len(adr_combined)}")

# ── Step 5: Add DrugBank ID
adr_combined = adr_combined.merge(
    vocab[['drug_name', 'drugbank_id']],
    on='drug_name', how='left'
)

# ── Step 6: Add Lipinski features
adr_combined = adr_combined.merge(
    lipinski[['drugbank_id', 'mol_weight', 'h_acceptors', 'h_donors', 'logp']],
    on='drugbank_id', how='left'
)

# ── Step 7: Add indications
indications = indications.merge(drug_names, on='stitch_flat_id', how='inner')
indications = indications[['drug_name', 'indication']].drop_duplicates()
indications_grouped = indications.groupby('drug_name')['indication'].apply(
    lambda x: '|'.join(x)).reset_index()
adr_combined = adr_combined.merge(indications_grouped, on='drug_name', how='left')

# ── Step 8: Final cleanup
adr_combined.dropna(subset=['drug_name', 'side_effect'], inplace=True)
adr_combined.drop_duplicates(subset=['drug_name', 'side_effect'], inplace=True)
adr_combined.fillna({
    'frequency': 0.0, 'mol_weight': 0.0,
    'h_acceptors': 0.0, 'h_donors': 0.0,
    'logp': 0.0, 'indication': 'unknown',
    'drugbank_id': 'unknown', 'source': 'unknown'
}, inplace=True)

adr_combined.to_csv(os.path.join(PROCESSED, 'adr_training_data.csv'), index=False)
print(f"    ✅ ADR training data saved — {len(adr_combined)} rows")
print(f"    Columns: {list(adr_combined.columns)}")

del meddra_se, meddra_freq, offsides, indications, adr_combined
gc.collect()


# ─────────────────────────────────────────
# MERGE 2: DDI TRAINING DATA
# ─────────────────────────────────────────
print("\n[2/2] Building DDI training dataset...")

output_path = os.path.join(PROCESSED, 'ddi_training_data.csv')
first_chunk = True
total