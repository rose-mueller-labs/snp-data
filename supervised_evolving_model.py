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

# Train model
print('Started training.')
clf = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42, verbose=1)
clf.fit(X_train, y_train)
print('Finished training.')

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


# import csv
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

# df = pd.read_csv("SNP_CSV.csv")
# df.head()

# df['Evolving'] = df['Evolving'].astype(int)

# # Define features (X) and labels (y)
# features = df[['Pos', 'Sel', 'Freq1', 'Freq2', 'Freq3', 'Freq4']]  # Input variables
# labels = df['Evolving']  # Target variable

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     features, labels, test_size=0.2, random_state=42
# )

# # Train model with class weighting (if imbalance exists)
# clf = RandomForestClassifier(class_weight='balanced', random_state=42)
# clf.fit(X_train, y_train)

# from sklearn.metrics import classification_report

# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))
