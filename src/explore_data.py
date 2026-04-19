import pandas as pd
import os

PROCESSED = os.path.join('data', 'processed')

print("=" * 60)
print("ADR TRAINING DATA")
print("=" * 60)
adr = pd.read_csv(os.path.join(PROCESSED, 'adr_training_data.csv'))
print(f"Shape: {adr.shape}")
print(f"Columns: {list(adr.columns)}")
print(f"\nTop 10 most common side effects:")
print(adr['side_effect'].value_counts().head(10))
print(f"\nDrugs with most side effects:")
print(adr['drug_name'].value_counts().head(5))
print(f"\nMissing values:\n{adr.isnull().sum()}")
del adr

print("\n" + "=" * 60)
print("DDI TRAINING DATA (sample of 10,000 rows)")
print("=" * 60)
ddi = pd.read_csv(os.path.join(PROCESSED, 'ddi_training_data.csv'),
                  nrows=10000)
print(f"Columns: {list(ddi.columns)}")
print(f"\nTop 10 most common interactions:")
print(ddi['interaction_effect'].value_counts().head(10))
print(f"\nMissing values:\n{ddi.isnull().sum()}")
del ddi