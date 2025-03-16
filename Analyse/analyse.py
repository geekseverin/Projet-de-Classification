# analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 1. Charger les données
print("Chargement des données...")
df = pd.read_csv("models/features_analysis.csv")
pca_info = pd.read_csv("models/features_analysis_pca_info.csv")

# 2. Exploration de base des données
print("\n--- APERÇU DES DONNÉES ---")
print(f"Nombre total d'images: {len(df)}")
print(df['category'].value_counts())

# 3. Analyse des composantes principales
print("\n--- ANALYSE DES COMPOSANTES PRINCIPALES ---")
print("Variance expliquée par les composantes principales:")
plt.figure(figsize=(10, 5))
plt.bar(pca_info['component'], pca_info['explained_variance'])
plt.plot(pca_info['component'], pca_info['cumulative_variance'], 'r-o', linewidth=2)
plt.xlabel('Composante Principale')
plt.ylabel('Variance Expliquée')
plt.title('Variance expliquée par chaque composante principale')
plt.xticks(rotation=45)
plt.axhline(y=0.7, color='g', linestyle='--', label='Seuil 70%')
plt.legend()
plt.tight_layout()
plt.savefig('D:/TP_Apeke/Analyse/results/pca_variance.png')
print(f"Variance expliquée par les 10 premières composantes: {pca_info['cumulative_variance'].iloc[-1]:.2%}")

# 4. Visualisation des caractéristiques
print("\n--- VISUALISATION DES CARACTÉRISTIQUES ---")

# 4.1 Distribution des statistiques par catégorie
print("Génération des graphiques de distribution...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for i, feature in enumerate(['feature_mean', 'feature_std', 'feature_min', 'feature_max']):
    row, col = i // 2, i % 2
    sns.boxplot(data=df, x='category', y=feature, ax=axes[row, col])
    axes[row, col].set_title(f'Distribution de {feature} par catégorie')
plt.tight_layout()
plt.savefig('D:/TP_Apeke/Analyse/results/feature_distributions.png')

# 4.2 Projection PCA
plt.figure(figsize=(10, 8))
scatter = sns.scatterplot(data=df, x='pca_component_1', y='pca_component_2', 
                         hue='category', palette='viridis', s=100, alpha=0.7)
plt.title('Projection des images selon les 2 premières composantes principales')
plt.xlabel('Composante Principale 1')
plt.ylabel('Composante Principale 2')
plt.legend(title='Catégorie')
plt.grid(True, alpha=0.3)
plt.savefig('D:/TP_Apeke/Analyse/results/pca_projection.png')

# 4.3 t-SNE pour une meilleure visualisation
print("Calcul de t-SNE pour une meilleure visualisation...")
# Sélectionner les 10 composantes PCA pour l'entrée de t-SNE
pca_components = df[[f'pca_component_{i+1}' for i in range(10)]].values

# Appliquer t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
tsne_results = tsne.fit_transform(pca_components)

# Ajouter les résultats au dataframe
df['tsne_1'] = tsne_results[:, 0]
df['tsne_2'] = tsne_results[:, 1]

# Visualiser t-SNE
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='tsne_1', y='tsne_2', hue='category', palette='viridis', s=100, alpha=0.7)
plt.title('Projection t-SNE des images (basée sur les 10 premières composantes PCA)')
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.legend(title='Catégorie')
plt.grid(True, alpha=0.3)
plt.savefig('D:/TP_Apeke/Analyse/results/tsne_projection.png')

# 5. Analyse statistique
print("\n--- ANALYSE STATISTIQUE ---")
stats_by_category = df.groupby('category')[['feature_mean', 'feature_std', 'feature_min', 'feature_max', 'feature_median']].agg(['mean', 'std'])
print(stats_by_category)

# Enregistrement des statistiques dans un fichier CSV
stats_by_category.to_csv('D:/TP_Apeke/Analyse/results/category_statistics.csv')
print("Statistiques par catégorie enregistrées dans 'D:/TP_Apeke/Analyse/results/category_statistics.csv'")

# 6. Construction d'un modèle de classification simple
print("\n--- MODÈLE DE CLASSIFICATION ---")
# Préparer les données
features = df[[f'pca_component_{i+1}' for i in range(10)]]
target = df['label_id']

# Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

# Entraîner un modèle Random Forest
print("Entraînement d'un modèle Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Évaluer le modèle
y_pred = rf.predict(X_test)
accuracy = np.mean(y_pred == y_test)

print(f"Précision du modèle: {accuracy:.2%}")
print("\nRapport de classification:")
print(classification_report(y_test, y_pred, target_names=df['category'].unique()))

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=df['category'].unique(),
            yticklabels=df['category'].unique())
plt.xlabel('Prédiction')
plt.ylabel('Valeur réelle')
plt.title('Matrice de confusion')
plt.tight_layout()
plt.savefig('D:/TP_Apeke/Analyse/results/confusion_matrix.png')

# 7. Importance des caractéristiques
print("\n--- IMPORTANCE DES CARACTÉRISTIQUES ---")
feature_importance = pd.DataFrame({
    'feature': [f'PCA_{i+1}' for i in range(10)],
    'importance': rf.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Importance des composantes PCA dans la classification')
plt.xlabel('Importance')
plt.ylabel('Composante')
plt.tight_layout()
plt.savefig('D:/TP_Apeke/Analyse/results/feature_importance.png')

# 8. Analyse des erreurs
print("\n--- ANALYSE DES ERREURS ---")
# Créer un DataFrame avec les résultats de test
test_results = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred,
    'correct': y_test == y_pred
})

# Ajouter les indices originaux pour pouvoir relier aux données d'origine
test_results.index = X_test.index
error_cases = test_results[~test_results['correct']]

# Joindre avec les données originales
error_analysis = df.loc[error_cases.index].copy()
error_analysis['actual'] = error_cases['actual'].map({i: cat for i, cat in enumerate(df['category'].unique())})
error_analysis['predicted'] = error_cases['predicted'].map({i: cat for i, cat in enumerate(df['category'].unique())})

# Sauvegarder les cas d'erreur pour analyse future
error_analysis.to_csv('D:/TP_Apeke/Analyse/results/error_analysis.csv', index=False)
print(f"Nombre de cas mal classés: {len(error_analysis)}")
print(f"Détails des erreurs enregistrés dans 'D:/TP_Apeke/Analyse/results/error_analysis.csv'")

print("\n Analyse terminée! Tous les résultats ont été enregistrés dans le dossier 'results/'")