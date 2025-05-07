import pandas as pd
import os

# Input user ID
user_id = input("Enter the user ID for which you want to extract features: ")
raw_data_file = f'data/{user_id}_raw.csv'

# Check if file exists
if not os.path.exists(raw_data_file):
    print(f"❌ File {raw_data_file} not found. Please collect data first.")
    exit()

# Load the raw keystroke data
df = pd.read_csv(raw_data_file)

# Define the expected password (same as in the dataset)
password = ".tie5Roanl"
key_sequence = list(password)

# Extract press/release times
press_times = {}
release_times = {}

for _, row in df.iterrows():
    k = row['key']
    t = row['time']
    if row['event'] == 'press' and k not in press_times:
        press_times[k] = t
    elif row['event'] == 'release' and k not in release_times:
        release_times[k] = t

# Compute features
row = {
    'subject': user_id,
    'sessionIndex': 1,
    'rep': 1
}

# Hold times (H.<key>)
for k in key_sequence:
    row[f'H.{k}'] = release_times.get(k, 0) - press_times.get(k, 0)

# Down-Down times (DD.<k1>.<k2>)
for i in range(len(key_sequence) - 1):
    k1 = key_sequence[i]
    k2 = key_sequence[i + 1]
    row[f'DD.{k1}.{k2}'] = press_times.get(k2, 0) - press_times.get(k1, 0)

# Up-Down times (UD.<k1>.<k2>)
for i in range(len(key_sequence) - 1):
    k1 = key_sequence[i]
    k2 = key_sequence[i + 1]
    row[f'UD.{k1}.{k2}'] = press_times.get(k2, 0) - release_times.get(k1, 0)

# Save the final feature row
feature_df = pd.DataFrame([row])
feature_df.to_csv(f'data/{user_id}_features.csv', index=False)
print(f"✅ DSL-compatible features extracted and saved to data/{user_id}_features.csv")
