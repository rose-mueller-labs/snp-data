import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap

# Load data
print('Before loading data.')
df = pd.read_csv("SNP_CSV.csv")
print('Finished loading data.')

# Feature engineering - create trajectory features
freq_cols = ['Freq1', 'Freq2', 'Freq3', 'Freq4']
X = df[freq_cols].values

print("here1")

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality Reduction with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("here2")

# Dimensionality Reduction with t-SNE (alternative) **
# tsne = TSNE(n_components=2, perplexity=2)
# X_tsne = tsne.fit_transform(X_scaled)

reducer = umap.UMAP()
embedding = reducer.fit_transform(X_scaled)
embedding.shape

print("here3")

# Clustering with K-Means ** (idk if this will be correct)
kmeans = KMeans(n_clusters=1, random_state=42)  # Assuming single population (CACO)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("here4")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

print("here5")

# PCA Plot
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Pop'], ax=axes[0])
axes[0].set_title('PCA Projection')

print("here6")

# t-SNE Plot
# sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df['Pop'], ax=axes[1])
# axes[1].set_title('t-SNE Projection')

# UMAP Plot
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    # c=[sns.color_palette()[x] for x in X.species.map({"Adelie":0, "Chinstrap":1, "Gentoo":2})])
)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the dataset', fontsize=24)


plt.savefig("clustervis1")

print("here7")

# Frequency Trajectory Plot (VISUAL)
for i, row in df.iterrows():
    axes[2].plot(freq_cols, row[freq_cols], label=f"Sample {i+1}")
axes[2].set_title('Frequency Trajectories')
axes[2].set_ylabel('Frequency')
axes[2].legend()

plt.tight_layout()
plt.savefig("clustervis2")

# Prediction function for new data (for inference)
def predict_population(new_frequencies):
    new_data = scaler.transform([new_frequencies])
    cluster = kmeans.predict(new_data)[0]
    # In a real multi-population scenario, I would map clusters to populations
    # For now, just return cluster ID
    return f"Cluster {cluster}"

# Example usage
new_sample = [0.85, 0.90, 0.91, 0.92]
print(f"Predicted cluster: {predict_population(new_sample)}")
