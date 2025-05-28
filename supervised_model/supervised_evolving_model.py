import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("SNP_CSV.csv")
print('Finished loading.')

# Convert label to int
df['Evolving'] = df['Evolving'].astype(int)

# Define features and labels
features = df[['Pos', 'Sel', 'Freq1', 'Freq2', 'Freq3', 'Freq4']]
labels = df['Evolving']

# Drop NaNs or Infs
features.replace([np.inf, -np.inf], np.nan, inplace=True)
features.dropna(inplace=True)
labels = labels.loc[features.index]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)
print('Finished splitting.')
#being able to have an interactive environment
#where you can have a variable enviornment
#is super duper useful
#because you can see what is potentially weird
#with any variables
#for example: you've trained using position and selection
#rather than just frequencies
#meaning whatever model might just say "all positions earlier than X are good"
#which isn't necessarily bad perse
#for example:
#you could use clf.feature_importances_
#to investigate the effects of including variables in it
#if the most important variable was "sel"
#that's a big indicator the model isn't great

# Train model
print('Started training.')
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, verbose=1, n_jobs= 20)
clf.fit(X_train, y_train)
print('Finished training.')

# Evaluate
y_pred = clf.predict(X_test)
print(clf.feature_importances_)
print(classification_report(y_test, y_pred))