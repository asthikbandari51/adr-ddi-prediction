import pandas as pd
import gc
import os

PROCESSED = os.path.join('data', 'processed')

vocab    = pd.read_csv(os.path.join(PROCESSED, 'drugbank_vocab_clean.csv'))
lipinski = pd.read_csv(os.path.join(PROCESSED, 'lipinski_clean.csv'))
vocab['drug_name'] = vocab['drug_name'].str.lower().str.strip()
# ─────────────────────────────────────────
# MERGE 2: DDI TRAINING DATA
# ─────────────────────────────────────────
print("\n[2/2] Building DDI training dataset...")

output_path = os.path.join(PROCESSED, 'ddi_training_data.csv')
first_chunk = True
total_rows = 0
chunk_count = 0

for chunk in pd.read_csv(
    os.path.join(PROCESSED, 'twosides_clean.csv'),
    chunksize=100_000
):
    chunk_count += 1
    chunk['drug_1'] = chunk['drug_1'].str.lower().str.strip()
    chunk['drug_2'] = chunk['drug_2'].str.lower().str.strip()

    # Add DrugBank ID for drug_1
    chunk = chunk.merge(
        vocab[['drug_name', 'drugbank_id']].rename(
            columns={'drug_name': 'drug_1',
                     'drugbank_id': 'drugbank_id_1'}),
        on='drug_1', how='left'
    )

    # Add DrugBank ID for drug_2
    chunk = chunk.merge(
        vocab[['drug_name', 'drugbank_id']].rename(
            columns={'drug_name': 'drug_2',
                     'drugbank_id': 'drugbank_id_2'}),
        on='drug_2', how='left'
    )

    # Add Lipinski features for drug_1
    chunk = chunk.merge(
        lipinski.rename(columns={
            'drugbank_id': 'drugbank_id_1',
            'mol_weight': 'mol_weight_1',
            'h_acceptors': 'h_acceptors_1',
            'h_donors': 'h_donors_1',
            'logp': 'logp_1'
        }),
        on='drugbank_id_1', how='left'
    )

    # Add Lipinski features for drug_2
    chunk = chunk.merge(
        lipinski.rename(columns={
            'drugbank_id': 'drugbank_id_2',
            'mol_weight': 'mol_weight_2',
            'h_acceptors': 'h_acceptors_2',
            'h_donors': 'h_donors_2',
            'logp': 'logp_2'
        }),
        on='drugbank_id_2', how='left'
    )

    chunk.fillna(0.0, inplace=True)
    chunk.drop_duplicates(
        subset=['drug_1', 'drug_2', 'interaction_effect'],
        inplace=True
    )

    chunk.to_csv(
        output_path,
        mode='w' if first_chunk else 'a',
        header=first_chunk,
        index=False
    )
    first_chunk = False
    total_rows += len(chunk)

    if chunk_count % 5 == 0:
        print(f"    ... processed {chunk_count * 100_000:,} rows, "
              f"kept {total_rows:,} so far")

    del chunk
    gc.collect()

print(f"    ✅ DDI training data saved — {total_rows} rows")

print("\n" + "=" * 60)
print("  MERGING COMPLETE!")
print("  Files saved to data/processed/:")
print("    → adr_training_data.csv  ✅ already done")
print("    → ddi_training_data.csv  ✅ just completed")
print("=" * 60)