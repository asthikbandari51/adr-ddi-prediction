import pandas as pd
import os

PROCESSED = os.path.join('data', 'processed')

files = [
    'drug_names_clean.csv',
    'meddra_se_clean.csv',
    'meddra_freq_clean.csv',
    'indications_clean.csv',
    'offsides_clean.csv',
]

for f in files:
    df = pd.read_csv(os.path.join(PROCESSED, f), nrows=2)
    print(f"\n{'='*40}")
    print(f"FILE: {f}")
    print(f"Columns: {list(df.columns)}")
    print(df)