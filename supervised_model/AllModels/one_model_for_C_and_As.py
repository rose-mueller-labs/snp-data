import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import time

# Load dataset
df = pd.read_csv("/home/drosophila-lab/Documents/Genomics Project/snp-data/DATA/SNP_CSV_w_pvalues_with_thresholds.csv")
print('Finished loading.')

# Convert label to int
cols = [
    # 'Threshold0.1',
    #     'Threshold0.2',
        # 'Threshold0.30000000000000004',
        # 'Threshold0.4',
        # 'Threshold0.5',
        # 'Threshold0.6',
        # 'Threshold0.7',
        # 'Threshold0.7999999999999999',
        # 'Threshold0.8999999999999999',
         'Threshold0.9999999999999999'
        ]

models = {}

for col in cols:
    # Convert label to int
    df[col] = df[col].astype(int)

    # Define populations and initialize models
    populations = {
        'C': ['CACO', 'CAO'], 
        'A': ['NACO', 'ANCO']
        } 
    #['CACO', 'CAO', 'NACO', 'ANCO']

    for pop in populations:
        print(f"\n=== Processing {pop} population ===")
    
        # Filter data for current population
        # pop_df = df[df['Pop'] == populations[pop][0] or df['Pop'] == populations[pop][1]]
        pop_df = df[df['Pop'].isin(populations[pop])]

        features = pop_df[['Pos', 'Freq1', 'Freq2', 'Freq3', 'Freq4']]
        labels = pop_df[col]
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.dropna(inplace=True)
        labels = labels.loc[features.index]
    
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.01, random_state=42
        )
        print(f'Training samples: {len(X_train)}, Test samples: {len(X_test)}')
    
        # Train model
        models[f'{col}_{pop}'] = RandomForestClassifier(
            n_estimators=50,
            class_weight='balanced',
            random_state=42,
            verbose=1,
            n_jobs=20
        )
        models[f'{col}_{pop}'].fit(X_train, y_train)
    
        # Evaluate
        y_pred = models[f'{col}_{pop}'].predict(X_test)
        with open(f'{pop}_results.txt', 'w') as f:
            print(f"\nFeature importances ({pop}):", file=f)
            print(dict(zip(features.columns, models[f'{col}_{pop}'].feature_importances_)), file=f)
            print(f"\nClassification Report ({pop}):", file=f)
            print(classification_report(y_test, y_pred), file=f)

        # save
        with open(f'{col}_{pop}_SNPs_model.pkl','wb') as f:
            pickle.dump(models[f'{col}_{pop}'],f)
        
        time.sleep(60)
