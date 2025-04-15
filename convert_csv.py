import pandas as pd
import re
import statsmodels.api as sm

# Load data
file_path = "2025_03_13_genomic_trajectory_SNP_table_withPvalues.csv"
df = pd.read_csv(file_path)

