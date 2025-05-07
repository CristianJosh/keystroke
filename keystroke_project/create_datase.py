import pandas as pd
import os

# Directory containing feature files
data_dir = 'data'
feature_files = [f for f in os.listdir(data_dir) if f.endswith('_features.csv')]

# Lists to store feature vectors and labels
all_features = []
labels = []

# Process each user's feature file
for file in feature_files:
    user_data = pd.read_csv(os.path.join(data_dir, file))
    
    # Identify feature columns (start with H., DD., or UD.)
    feature_cols = [col for col in user_data.columns if col.startswith(('H.', 'DD.', 'UD.'))]
    
    # Extract features and label
    features = user_data[feature_cols].values
    label = user_data['subject'].iloc[0]
    
    all_features.extend(features)
    labels.extend([label] * len(features))

# Create final dataset
dataset = pd.DataFrame(all_features, columns=feature_cols)
dataset['user_id'] = labels

# Save combined dataset
dataset.to_csv('data/keystroke_dataset.csv', index=False)
print("✅ DSL-style dataset compiled and saved to data/keystroke_dataset.csv")
