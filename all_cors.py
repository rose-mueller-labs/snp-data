import pandas as pd
import re
import statsmodels.api as sm

# Load data
file_path = "2025_03_13_genomic_trajectory_SNP_table_withPvalues.csv"
df = pd.read_csv(file_path)

# Reshaping data into long format (helps with later analysis)
allele_data = []
for col in df.columns[5:-2]:  # Skip first few columns (chr, pos, ref, alt) and last two (convertedPosition, pvalue)
    match = re.match(r"([A-Z]+)(\d+)_(\d+)", col)  # Extract selection type, trajectory, and year
    if match:
        selection, traj, year = match.groups()
        for idx, freq in enumerate(df[col]):
            allele_data.append({
                "chr": df.loc[idx, "chr"],
                "pos": df.loc[idx, "pos"],
                "Selection": selection,   # CACO, CAO, NACO, etc, (Selection type)
                "Trajectory": traj,       # 1,2,3, as the trajectories (I think)
                "Generation": int(year),  # Yr is generation
                "Frequency": freq
            })

# Convert to df
long_df = pd.DataFrame(allele_data)

# Convert categorical variables to dummy variables
long_df = pd.get_dummies(long_df, columns=["Selection", "Trajectory"], drop_first=True)

print(long_df.dtypes)

# Define X and Y, which is where Frequency will be our target and generation our independent variable

long_df["Frequency"] = pd.to_numeric(long_df["Frequency"], errors="coerce")
long_df["Generation"] = pd.to_numeric(long_df["Generation"], errors="coerce")

X = long_df[["Generation"] + [col for col in long_df.columns if col.startswith("Selection_") or col.startswith("Trajectory_")]]
y = long_df["Frequency"]

# Add intercept for regression
X = sm.add_constant(X)

# Fit Linear Regression Model
model = sm.OLS(y, X).fit()

# Print coeffs/summary
print(model.summary())