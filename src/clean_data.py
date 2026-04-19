import pandas as pd
import gc
import os

# ─────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────
RAW = os.path.join('data', 'raw')
PROCESSED = os.path.join('data', 'processed')
os.makedirs(PROCESSED, exist_ok=True)

print("="*60)
print("  ADR-DDI Project — Data Cleaning Script")
print("="*60)


# ─────────────────────────────────────────
# 1. drug_names.tsv  (34 KB — tiny)
# ─────────────────────────────────────────
print("\n[1/8] Cleaning drug_names.tsv ...")

drug_names = pd.read_csv(
    os.path.join(RAW, 'drug_names.tsv'),
    sep='\t',
    header=None,
    names=['stitch_id', 'drug_name']
)
drug_names['drug_name'] = drug_names['drug_name'].str.lower().str.strip()
drug_names.drop_duplicates(inplace=True)
drug_names.dropna(inplace=True)

drug_names.to_csv(os.path.join(PROCESSED, 'drug_names_clean.csv'), index=False)
print(f"    ✅ Done — {len(drug_names)} drugs saved")
del drug_names
gc.collect()


# ─────────────────────────────────────────
# 2. drugbank vocabulary.csv  (3 MB — tiny)
# ─────────────────────────────────────────
print("\n[2/8] Cleaning drugbank_vocabulary.csv ...")

vocab = pd.read_csv(
    os.path.join(RAW, 'drugbank vocabulary.csv'),
    usecols=['DrugBank ID', 'Common name', 'Synonyms']
)
vocab.columns = ['drugbank_id', 'drug_name', 'synonyms']
vocab['drug_name'] = vocab['drug_name'].str.lower().str.strip()
vocab.drop_duplicates(subset='drugbank_id', inplace=True)
vocab.dropna(subset=['drugbank_id', 'drug_name'], inplace=True)

vocab.to_csv(os.path.join(PROCESSED, 'drugbank_vocab_clean.csv'), index=False)
print(f"    ✅ Done — {len(vocab)} drugs saved")
del vocab
gc.collect()


# ─────────────────────────────────────────
# 3. DB_compounds_lipinski.csv  (308 KB — tiny)
# ─────────────────────────────────────────
print("\n[3/8] Cleaning DB_compounds_lipinski.csv ...")

lipinski = pd.read_csv(
    os.path.join(RAW, 'DB_compounds_lipinski.csv'),
    usecols=['ID', 'molecular_weight', 'n_hba', 'n_hbd', 'logp', 'ro5_fulfilled']
)
lipinski.columns = ['drugbank_id', 'mol_weight', 'h_acceptors', 'h_donors', 'logp', 'ro5']
lipinski.dropna(subset=['drugbank_id'], inplace=True)
lipinski.drop_duplicates(subset='drugbank_id', inplace=True)

# Convert ro5_fulfilled TRUE/FALSE → 1/0
lipinski['ro5'] = lipinski['ro5'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})

lipinski.to_csv(os.path.join(PROCESSED, 'lipinski_clean.csv'), index=False)
print(f"    ✅ Done — {len(lipinski)} compounds saved")
del lipinski
gc.collect()


# ─────────────────────────────────────────
# 4. meddra_all_se.tsv  (19 MB — small)
# ─────────────────────────────────────────
print("\n[4/8] Cleaning meddra_all_se.tsv ...")

meddra_se = pd.read_csv(
    os.path.join(RAW, 'meddra_all_se.tsv'),
    sep='\t',
    header=None,
    names=['stitch_flat_id', 'stitch_stereo_id', 'umls_id',
           'meddra_type', 'umls_meddra_id', 'side_effect_name']
)
# Keep only PT (Preferred Term) — removes duplicates
meddra_se = meddra_se[meddra_se['meddra_type'] == 'PT']
meddra_se = meddra_se[['stitch_flat_id', 'side_effect_name']]
meddra_se['side_effect_name'] = meddra_se['side_effect_name'].str.lower().str.strip()
meddra_se.drop_duplicates(inplace=True)
meddra_se.dropna(inplace=True)

meddra_se.to_csv(os.path.join(PROCESSED, 'meddra_se_clean.csv'), index=False)
print(f"    ✅ Done — {len(meddra_se)} drug-side_effect pairs saved")
del meddra_se
gc.collect()


# ─────────────────────────────────────────
# 5. meddra_freq.tsv  (23 MB — small)
# ─────────────────────────────────────────
print("\n[5/8] Cleaning meddra_freq.tsv ...")

meddra_freq = pd.read_csv(
    os.path.join(RAW, 'meddra_freq.tsv'),
    sep='\t',
    header=None,
    names=['stitch_flat_id', 'stitch_stereo_id', 'umls_id',
           'freq_label', 'freq_lower', 'freq_upper',
           'meddra_type', 'umls_meddra_id', 'side_effect_name']
)
# Remove placebo rows and keep only PT
meddra_freq = meddra_freq[
    (meddra_freq['meddra_type'] == 'PT') &
    (~meddra_freq['freq_label'].astype(str).str.contains('placebo', case=False, na=False))
]
meddra_freq = meddra_freq[['stitch_flat_id', 'freq_lower', 'side_effect_name']]
meddra_freq['side_effect_name'] = meddra_freq['side_effect_name'].str.lower().str.strip()
meddra_freq.dropna(inplace=True)
meddra_freq.drop_duplicates(inplace=True)

meddra_freq.to_csv(os.path.join(PROCESSED, 'meddra_freq_clean.csv'), index=False)
print(f"    ✅ Done — {len(meddra_freq)} freq records saved")
del meddra_freq
gc.collect()


# ─────────────────────────────────────────
# 6. meddra_all_indications.tsv  (2.5 MB — tiny)
# ─────────────────────────────────────────
print("\n[6/8] Cleaning meddra_all_indications.tsv ...")

indications = pd.read_csv(
    os.path.join(RAW, 'meddra_all_indications.tsv'),
    sep='\t',
    header=None,
    names=['stitch_flat_id', 'umls_id', 'method',
           'concept_name', 'meddra_type', 'umls_meddra_id', 'meddra_concept_name']
)
# Keep only PT rows
indications = indications[indications['meddra_type'] == 'PT']
indications = indications[['stitch_flat_id', 'concept_name']]
indications.columns = ['stitch_flat_id', 'indication']
indications['indication'] = indications['indication'].str.lower().str.strip()
indications.drop_duplicates(inplace=True)
indications.dropna(inplace=True)

indications.to_csv(os.path.join(PROCESSED, 'indications_clean.csv'), index=False)
print(f"    ✅ Done — {len(indications)} drug-indication pairs saved")
del indications
gc.collect()


# ─────────────────────────────────────────
# 7. OFFSIDES.csv  (286 MB — medium)
# ─────────────────────────────────────────
print("\n[7/8] Cleaning OFFSIDES.csv ...")

offsides = pd.read_csv(
    os.path.join(RAW, 'OFFSIDES.csv'),
    usecols=['drug_concept_name', 'condition_concept_name',
             'PRR', 'mean_reporting_frequency'],
    low_memory=False                      # ← fixes DtypeWarning
)
offsides.columns = ['drug_name', 'side_effect', 'PRR', 'freq']
offsides['drug_name'] = offsides['drug_name'].str.lower().str.strip()
offsides['side_effect'] = offsides['side_effect'].str.lower().str.strip()

# ← THIS IS THE KEY FIX: force PRR and freq to numeric
# any non-numeric values (strings, blanks) become NaN and get dropped
offsides['PRR'] = pd.to_numeric(offsides['PRR'], errors='coerce')
offsides['freq'] = pd.to_numeric(offsides['freq'], errors='coerce')

offsides.dropna(inplace=True)

# Keep only statistically significant signals
offsides = offsides[
    (offsides['PRR'] > 2.0) &
    (offsides['freq'] > 0.01)
]
offsides.drop_duplicates(inplace=True)

offsides.to_csv(os.path.join(PROCESSED, 'offsides_clean.csv'), index=False)
print(f"    ✅ Done — {len(offsides)} ADR records saved")
del offsides
gc.collect()

# ─────────────────────────────────────────
# 8. TWOSIDES.csv  (4.2 GB — CHUNKED)
# ─────────────────────────────────────────
print("\n[8/8] Cleaning TWOSIDES.csv (chunked — this will take a few minutes) ...")

CHUNK_SIZE = 100_000
output_path = os.path.join(PROCESSED, 'twosides_clean.csv')
first_chunk = True
total_rows = 0
chunk_count = 0

for chunk in pd.read_csv(
    os.path.join(RAW, 'TWOSIDES.csv'),
    usecols=['drug_1_concept_name', 'drug_2_concept_name',
             'condition_concept_name', 'PRR', 'mean_reporting_frequency'],
    chunksize=CHUNK_SIZE,
    low_memory=False                      # ← fixes DtypeWarning
):
    chunk_count += 1

    # Rename columns
    chunk.columns = ['drug_1', 'drug_2', 'interaction_effect', 'PRR', 'freq']

    # ← KEY FIX: force PRR and freq to numeric (handles "NA", "Inf", strings)
    chunk['PRR'] = pd.to_numeric(chunk['PRR'], errors='coerce')
    chunk['freq'] = pd.to_numeric(chunk['freq'], errors='coerce')

    # Lowercase drug and effect names
    chunk['drug_1'] = chunk['drug_1'].str.lower().str.strip()
    chunk['drug_2'] = chunk['drug_2'].str.lower().str.strip()
    chunk['interaction_effect'] = chunk['interaction_effect'].str.lower().str.strip()

    # Drop NaN (from coerce above + any missing values)
    chunk.dropna(inplace=True)

    # Filter: keep only strong, frequent interaction signals
    chunk = chunk[
        (chunk['PRR'] > 2.0) &
        (chunk['freq'] > 0.01)
    ]
    chunk.drop_duplicates(inplace=True)

    # Write to file (append after first chunk)
    if len(chunk) > 0:
        chunk.to_csv(
            output_path,
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            index=False
        )
        first_chunk = False

    total_rows += len(chunk)

    # ← Progress update every 10 chunks so you know it's still running
    if chunk_count % 10 == 0:
        print(f"    ... processed {chunk_count * CHUNK_SIZE:,} rows, "
              f"kept {total_rows:,} so far")

    del chunk
    gc.collect()

print(f"    ✅ Done — {total_rows} DDI records saved")

print("\n" + "="*60)
print("  ALL FILES CLEANED SUCCESSFULLY!")
print(f"  Saved to: data/processed/")
print("="*60)