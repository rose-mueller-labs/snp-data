# best_thresh_per_sel_pop

import pickle
import csv
import pandas as pd

# goal: find the best threshold for each of the 20 populations (e.g. 0.1CAO_1...0.9CAO_1)
# we find this by testing on the same threshold but different population
# (e.g. 0.1CAO_1 gets tested on the data from Threshold: 0.1 and 
# Sel: 1 and Populations = [CACO, NACO, ANCO])

# get the f1 score and save it to a csv file with columns "Threshold", "Population", and "Sel"

import pandas as pd
import pickle
from sklearn.metrics import f1_score, accuracy_score

LOC_PATH = "/mnt/drosophila-lab/GenomicsProject/PopSelModels/ModelPerPopSel"

# Load dataset containing all populations and features
df = pd.read_csv("/home/drosophila-lab/Documents/Genomics Project/snp-data/DATA/SNP_CSV_w_pvalues_with_thresholds.csv")  # Contains Threshold, Pop, Sel, Freq1-4, Evolving

# Define model parameters
populations = ['CACO', 'CAO', 'NACO', 'ANCO']
selections = [1, 2, 3, 4, 5]
thresholds = [
                'Threshold0.1', 'Threshold0.2', 'Threshold0.30000000000000004',
              'Threshold0.4', 'Threshold0.5', 'Threshold0.6', 'Threshold0.7',
              'Threshold0.7000000000000000', 'Threshold0.8999999999999999',
              # 'Threshold0.9999999999999999'
            ]

results = []
best_results = []

for target_pop in populations:
    for sel in selections:
        best_performance = {'Threshold': None, 'F1': -1, 'Accuracy': -1}
        
        for thresh in thresholds:
            model_name = f"{LOC_PATH}/{thresh}_{target_pop}_{sel}SNPs_model.pkl"
            
            try:
                # Load model trained on specific population and selection
                with open(model_name, 'rb') as f:
                    model = pickle.load(f)
            except FileNotFoundError:
                continue

            # Prepare test data: same threshold/selection, different populations
            test_pops = [p for p in populations if p != target_pop]
            test_data = df[
                (df['Sel'] == sel) &
                (df['Pop'].isin(test_pops))
            ]

            if test_data.empty:
                continue

            # Extract features and labels
            X_test = test_data[['Freq1', 'Freq2', 'Freq3', 'Freq4']]
            y_test = test_data[thresh].astype(int)
            
            # Generate predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Update best threshold
            if f1 > best_performance['F1']:
                best_performance = {
                    'Threshold': thresh,
                    'F1': f1,
                    'Accuracy': accuracy
                }

            # Record all results
            results.append({
                'Population': target_pop,
                'Selection': sel,
                'Threshold': thresh,
                'F1 Score': f1,
                'Accuracy': accuracy
            })

        # Save best threshold for current population/selection
        if best_performance['Threshold'] is not None:
            best_results.append({
                'Population': target_pop,
                'Selection': sel,
                'Best Threshold': best_performance['Threshold'],
                'Best F1': best_performance['F1'],
                'Accuracy at Best': best_performance['Accuracy']
            })

# Save outputs
pd.DataFrame(results).to_csv('pop_sel_model_results.csv', index=False)
pd.DataFrame(best_results).to_csv('optimal_thresholds_pop_sel.csv', index=False)
