import pandas as pd

# Read all splits
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

for df in [train_df, val_df, test_df]:
    df.loc[df["ethnicity"] == 6, "ethnicity"] = 1
    
# Add split column
train_df['split'] = 0
val_df['split'] = 1
test_df['split'] = 2

# Combine into single metadata file
metadata_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Save combined metadata
metadata_df.to_csv("metadata.csv", index=False)
print("Created metadata.csv with splits:", metadata_df['split'].value_counts().sort_index())