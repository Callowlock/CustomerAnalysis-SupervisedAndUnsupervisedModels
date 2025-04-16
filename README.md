# Mall Customer Segmentation

This project explores both supervised and unsupervised machine learning techniques to analyze and segment customers from a shopping mall dataset.

## Objectives

1. **Classification (Supervised Learning)**: Predict customer spending level (low, medium, high) based on age, income, and gender using:
   - K-Nearest Neighbors (KNN)
   - Naive Bayes (GaussianNB, BernoulliNB)

2. **Clustering (Unsupervised Learning)**: Segment customers into groups using:
   - K-Means Clustering
   - Agglomerative Clustering

---

## Dataset

- Source: `data/Mall_Customers.csv`
- 200 entries with 4 columns:
  - Gender
  - Age
  - Annual Income (k$)
  - Spending Score (1-100)

Preprocessing included:
- Dropping CustomerID
- Renaming columns for convenience
- Binning `spending_score` into quantiles: low, medium, high

---

## Exploratory Data Analysis (EDA)

- **Distributions** of income, age, and spending score were visualized.
- **Correlation analysis** revealed:
  - No correlation between income and spending
  - Mild negative correlation between age and spending
- **Gender distribution**: Slight female majority
- **Age**: Right-skewed

---

## Supervised Learning

### K-Nearest Neighbors (KNN)

- Applied one-hot encoding to gender
- Standardized age and income
- Mapped labels (low=0, medium=1, high=2)

Performance:
- Initial accuracy: 0.75
- With GridSearch (n_neighbors): 0.80
- With full parameter tuning (algorithm, weights, p): 0.825

### Naive Bayes

#### GaussianNB
- Accuracy: ~0.70 (improved slightly with GridSearch and feature selection)

#### BernoulliNB
- Converted age/income to binary
- Performance was poor (accuracy: ~0.5-0.57)
- GridSearch on binarization threshold and smoothing provided minor improvements

---

## Unsupervised Learning

### K-Means Clustering

- Standardized age, income, and spending score
- Tested optimal number of clusters (elbow method suggested 4)
- Observed clusters based on age, income, and gender behavior

### Agglomerative Clustering

- Tested various linkage methods and distance metrics
- Evaluated silhouette scores to select optimal configuration (ward + euclidean, k=10)
- Dendrograms provided insight into hierarchical groupings
- Compared results to KMeans with 10 clusters (86.5% similarity)

---

## Key Learnings

- KNN outperformed Naive Bayes in classifying spending levels
- Agglomerative and KMeans clustering produced similar and interpretable groupings
- Spending score does not correlate with income, but shows mild inverse correlation with age

---

## Next Steps

- Apply dimensionality reduction (PCA, t-SNE) for visualization
- Try ensemble classifiers or neural networks
- Further analyze cluster behavior for targeted marketing strategies

---

## Environment

- Libraries: pandas, numpy, seaborn, matplotlib, sklearn, scipy
- Tools: GridSearchCV, StratifiedKFold, StandardScaler, MinMaxScaler, Pipelines

---

## Run the Notebook

Ensure the following structure:
```
project-root/
├── data/
│   └── Mall_Customers.csv
├── your_notebook.ipynb
```

To run:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

---

## Author
Chris Nevares

