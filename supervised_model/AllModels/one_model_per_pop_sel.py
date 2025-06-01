import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import time

# Load dataset
df = pd.read_csv("/home/drosophila-lab/Documents/Genomics Project/snp-data/SNP_CSV_w_pvalues_with_thresholds.csv")
print('Finished loading.')

cols = [# 'Threshold0.1',
        # 'Threshold0.2',
        # 'Threshold0.30000000000000004',
        # 'Threshold0.4',
        # 'Threshold0.5',
        # 'Threshold0.6',
        # 'Threshold0.7'
        #  'Threshold0.7999999999999999',
        #  'Threshold0.8999999999999999',
         'Threshold0.9999999999999999'
        ]

models = {}

for col in cols:
    # Convert label to int
    df[col] = df[col].astype(int)

    # Define populations and initialize models
    selections = [1, 2, 3, 4, 5]
    populations = ['CACO', 'CAO', 'NACO', 'ANCO']

    for pop in populations:
        for sel in selections:
            print(f"\n=== Processing {pop}_{sel} population ===")
        
            # Filter data for current population
            # pop_df = df[df['Sel'] == sel and df['Pop'] == pop]
            pop_df = df[(df['Sel'] == sel) & (df['Pop'] == pop)]
        
            # Define features and apply one-hot encoding
            features = pop_df[['Pos', 'Freq1', 'Freq2', 'Freq3', 'Freq4']]
            # features = pd.get_dummies(features, columns=['Pop'], prefix='Pop')
        
            # Rest of processing remains the same
            labels = pop_df[col]
            features.replace([np.inf, -np.inf], np.nan, inplace=True)
            features.dropna(inplace=True)
            labels = labels.loc[features.index]
        
            # Split data (added validation)
            min_samples = 5  # Minimum samples per class
            if len(features) < min_samples:
                print(f"Skipping population {pop} - insufficient samples")
                continue
            
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels,
                test_size=0.01,
                random_state=42,
                stratify=labels  # Preserve class balance
            )
            print(f'Training samples: {len(X_train)}, Test samples: {len(X_test)}')
        
            # Train model
            models[f'{col}_{pop}_{sel}'] = RandomForestClassifier(
                n_estimators=50,
                class_weight='balanced',
                random_state=42,
                verbose=1,
                n_jobs=20
            )
            models[f'{col}_{pop}_{sel}'].fit(X_train, y_train)
        
            # Evaluate
            y_pred = models[f'{col}_{pop}_{sel}'].predict(X_test)
            with open(f'{pop}_{sel}_results.txt', 'w') as f:
                print(f"\nFeature importances ({pop}):", file=f)
                print(dict(zip(features.columns, models[f'{col}_{pop}_{sel}'].feature_importances_)), file=f)
                print(f"\nClassification Report ({pop}):", file=f)
                print(classification_report(y_test, y_pred), file=f)

            # save
            time.sleep(360)
            with open(f'{col}_{pop}_{sel}SNPs_model.pkl','wb') as f:
                pickle.dump(models[f'{col}_{pop}_{sel}'],f)
            time.sleep(360)
