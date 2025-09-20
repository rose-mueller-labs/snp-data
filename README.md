# snp-data: SNP Evolution Analysis

This repository contains code and analysis from my research at the **Rose Lab (UC Irvine)**, where I investigated whether **drift or selection drives evolutionary change** in *Drosophila* populations. The project was funded by **Calit2**.

By analyzing SNP (single nucleotide polymorphism) data across multiple generations, I applied both supervised and unsupervised learning approaches to study evolutionary trajectories at the genomic scale.

---

## 🔬 Research Overview

* **Objective:** Determine whether patterns of SNP frequency change over time are best explained by neutral drift or selective pressure.
* **Data:** SNP frequencies across \~2.2 million base pairs in experimental *Drosophila* populations.
* **Supervised Learning:** Trained **220 Random Forest models**, iteratively refining the significance threshold for classifying evolutionary SNPs to **p = 0.000218**.
* **Unsupervised Learning:** Applied **UMAP** to visualize SNP trajectories, uncover latent structure in frequency shifts, and identify clustering patterns linked to drift vs. selection.

---

## 📂 Repository Structure

* `supervised/` – Random Forest classification pipeline for SNP evolution prediction
* `unsupervised/` – UMAP dimensionality reduction and visualization scripts
* `SNP_CSV.csv` – Example dataset (if permitted for sharing)
* `results/` – Feature importance plots, UMAP embeddings, evaluation metrics

---

## 🧠 Methods

### Supervised: Random Forest

Key steps:

* Balanced class weighting to handle imbalance between evolving vs. non-evolving SNPs.
* Feature importance analysis to evaluate the influence of genomic position, selection metrics, and frequency trajectories.

### Unsupervised: UMAP

* Reduced high-dimensional SNP frequency data (\~2.2M base pairs) into 2D embeddings.
* Explored how SNP clusters aligned with evolutionary pressures.
* Revealed emergent structure in SNP frequency dynamics across generations.

---

## 📊 Results

* Refined **p-value cutoff: 0.000218**, improving signal-to-noise ratio in detecting selective SNPs.
* Random Forest models identified **selection coefficients and positional information** as key drivers, while UMAP captured **population-level evolutionary trajectories**.
* Demonstrated the power of combining **supervised (predictive)** and **unsupervised (exploratory)** machine learning in evolutionary genomics.

---

## ⚙️ Requirements

* Python 3.9+
* Packages: `pandas`, `numpy`, `scikit-learn`, `umap`, `matplotlib`

---

## 📚 Citation

---

## 🔗 Links

* Lab Website: [Rose–Mueller Labs](https://rosemuellerlabs.bio.uci.edu/)
* Funding: [Calit2](https://calit2.org/)

---

## 👩🏽‍💻 Author

Developed by **Shreya Nakum** as part of undergraduate research in the Rose Lab.
For questions or collaboration, please reach out via [LinkedIn](https://www.linkedin.com/in/shreyanakum/) or e-mail.
