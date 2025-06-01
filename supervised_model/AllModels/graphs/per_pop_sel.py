import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("/home/drosophila-lab/Documents/Genomics Project/snp-data/supervised_model/AllModels/all_pop_model_results.csv")

colors = {'A': 'blue', 'C': 'orange'}
markers = {'F1 Score': 'o', 'Accuracy': 's'}

df['Threshold_decimal'] = round(df['Threshold'].str[9:].astype(float), 1)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

# F1 Score Plot
for pop in df['Population'].unique():
    df_pop = df[df['Population'] == pop]
    axes[0].scatter(df_pop['Threshold_decimal'], df_pop['F1 Score'],
                    color=colors[pop], marker=markers['F1 Score'],
                    label=f'{pop}', s=80, alpha=0.8)
axes[0].set_xlabel('Threshold')
axes[0].set_ylabel('F1 Score')
axes[0].set_title('F1 Score vs Threshold by Population')
axes[0].legend(title='Population')
axes[0].grid(True)

# Accuracy Plot
for pop in df['Population'].unique():
    df_pop = df[df['Population'] == pop]
    axes[1].scatter(df_pop['Threshold_decimal'], df_pop['Accuracy'],
                    color=colors[pop], marker=markers['Accuracy'],
                    label=f'{pop}', s=80, alpha=0.8)
axes[1].set_xlabel('Threshold')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy vs Threshold by Population')
axes[1].legend(title='Population')
axes[1].grid(True)

plt.tight_layout()
plt.savefig("per_pop_separate.png")
plt.show()
