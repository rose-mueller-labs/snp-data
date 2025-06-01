import csv
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

cols = [
'Threshold0.1',
        'Threshold0.2',
        'Threshold0.30000000000000004',
        'Threshold0.4',
        'Threshold0.5',
        'Threshold0.6',
        'Threshold0.7',
        'Threshold0.7999999999999999',
        'Threshold0.8999999999999999',
        # 'Threshold0.9999999999999999'
        ]


for col in cols:
    # Convert label to int
    df[col] = df[col].astype(int)

    # Define features and labels
    features = df[['Pos', 'Sel', 'Freq1', 'Freq2', 'Freq3', 'Freq4']]
    labels = df[col]

    # Drop NaNs or Infs
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.dropna(inplace=True)
    labels = labels.loc[features.index]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    print('Finished splitting.')

    # Train model
    print('Started training.')
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', 
                                 random_state=42, verbose=1, n_jobs= 20)
    clf.fit(X_train, y_train)
    print('Finished training.')

    # Evaluate
    y_pred = clf.predict(X_test)
    print(clf.feature_importances_)
    print(classification_report(y_test, y_pred))

    with open(f'{col}_SNPs_model.pkl','wb') as f:
            pickle.dump(clf,f)

    time.sleep(240)