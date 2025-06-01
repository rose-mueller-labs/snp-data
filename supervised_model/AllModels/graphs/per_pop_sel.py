import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load
df = pd.read_csv("/home/drosophila-lab/Documents/Genomics Project/snp-data/supervised_model/AllModels/pop_sel_model_results.csv")

# Clean Threshold column to extract float value after the first 9 characters
def extract_threshold(val):
    if isinstance(val, str) and val.startswith("Threshold"):
        return float(val[9:])
    else:
        try:
            return float(val)
        except:
            return None

df['Threshold_float'] = df['Threshold'].apply(extract_threshold)
df = df.dropna(subset=['Threshold_float'])

# Set up seaborn style
sns.set(style="whitegrid", font_scale=1.2)

populations = df['Population'].unique()
metrics = ['F1 Score', 'Accuracy']

for metric in metrics:
    for pop in populations:
        plt.figure(figsize=(10, 6))
        sub = df[df['Population'] == pop]
        sns.lineplot(
            data=sub,
            x='Threshold_float',
            y=metric,
            hue='Selection',
            palette='tab10',
            marker='o',
            legend='full'
        )
        plt.title(f"{metric} vs Threshold for {pop}")
        plt.xlabel("Threshold")
        plt.ylabel(metric)
        plt.legend(title='Selection')
        plt.tight_layout()
        plt.savefig(f"{metric.replace(' ', '_').lower()}_vs_threshold_{pop}.png")
        plt.close()
