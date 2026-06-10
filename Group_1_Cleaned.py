import pandas as pd
import re

df = pd.read_excel('Group_1_Data_Cleaned.xlsx')

def classify_note(row):
    notes = str(row['Notes']) if pd.notna(row['Notes']) else ''
    cleaned = str(row['Cleaned_Notes']) if pd.notna(row['Cleaned_Notes']) else ''
    
    # Check if any IDs are missing — Unclassified
    if pd.isna(row['ProductId']) or pd.isna(row['ProductVariantId']) or pd.isna(row['SkuId']):
        return 'Unclassified'
    
    # Over classified — notes has way more than just the product name
    over_keywords = [
        'apply', 'every', 'months', 'directions', 'directed',
        'gloves', 'eggs', 'chickens', 'batch', 'expires', 'expiry',
        'value £', 'free', 'per pump', 'per ml', 'per spray',
        'in consult', 'in house', 'in-house', 'procedure',
        'administered', 'treatment', 'application',
        'do not', 'order only', 'get from', 'see effipro',
        'use effipro', 'tinyurl', '0.25%', 'fipronil',
        'flea & tick', 'flea and tick', 'for flea',
        'cat/dog', 'cat or', 'small dog', 'large dog',
        'medium dog', 'puppy', 'kitten', 'per bottle',
        'lge', 'sml', 'reminder', 'irecall', 'special offer',
        'friends club', 'dogs trust', 'rspca', 'charity',
        'dispatch', 'bottle each', 'from pk', 'from pack',
    ]
    notes_lower = notes.lower()
    if any(kw in notes_lower for kw in over_keywords):
        return 'Over Classified'
    
    # Under classified — too vague to confirm the exact SKU
    under_patterns = [
        r'^frontline spray\s*$',           # just "Frontline Spray"
        r'^frontline\s*$',                  # just "Frontline"
        r'^frontline spray 100\s*$',        # missing unit
        r'^frontline spray\s+cat\s*$',      # species only
        r'^frontline spray\s+dog',          # species only
        r'^frontline\s+100ml\s*$',          # missing "spray"
        r'^frontline 100\s*ml\s*$',         # missing "spray"
        r'^spray.+frontline\s*$',           # vague reversed
        r'^tick spray.+frontline\s*$',      # purpose only
        r'^frontline spray 250',            # wrong size
        r'^frontline spray 500',            # wrong size
        r'^frontline 250',                  # wrong size
        r'^frontline 500',                  # wrong size
        r'^frontline homegard',             # different product
    ]
    if any(re.search(p, cleaned, re.IGNORECASE) for p in under_patterns):
        return 'Under Classified'
    
    # Correctly classified — everything else
    return 'Correctly Classified'

df['Classification'] = df.apply(classify_note, axis=1)

# Save
df.to_excel('Group_1_Data_Cleaned.xlsx', index=False)

print("Done!")
print(f"Total rows: {len(df)}")
print("\nClassification breakdown:")
print(df['Classification'].value_counts())
print("\nSample of each group:")
for group in df['Classification'].unique():
    print(f"\n--- {group} ---")
    print(df[df['Classification'] == group][['Notes', 'Cleaned_Notes', 'Classification']].head(3).to_string())


    