import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le CSV
df = pd.read_csv("models/features_analysis.csv")

# Visualiser la distribution de feature_mean par catégorie
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='category', y='feature_mean')
plt.title('Distribution de la moyenne des caractéristiques par catégorie')
plt.show()

# Visualiser d'autres statistiques également
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.boxplot(data=df, x='category', y='feature_std', ax=axes[0, 0])
sns.boxplot(data=df, x='category', y='feature_min', ax=axes[0, 1])
sns.boxplot(data=df, x='category', y='feature_max', ax=axes[1, 0])
sns.boxplot(data=df, x='category', y='feature_median', ax=axes[1, 1])
plt.tight_layout()
plt.show()