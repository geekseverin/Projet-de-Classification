# # feature_extraction.py
# import os
# import numpy as np
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# import pickle
# from tqdm import tqdm

# def extract_features(data_dir, output_file):
#     """
#     Extrait les caractéristiques des images à l'aide d'un modèle préentraîné (MobileNetV2)
#     """
#     # Charger le modèle préentraîné sans la couche de classification
#     base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    
#     categories = ['bon', 'moyen', 'mauvais']
#     X = []  # Caractéristiques
#     y = []  # Étiquettes
#     image_paths = []  # Chemins des images pour référence
    
#     # Pour chaque catégorie
#     for i, category in enumerate(categories):
#         print(f"Extraction des caractéristiques pour la catégorie: {category}")
#         category_path = os.path.join(data_dir, category)
        
#         if not os.path.exists(category_path):
#             print(f"Le dossier {category_path} n'existe pas!")
#             continue
            
#         images = os.listdir(category_path)
        
#         # Pour chaque image dans la catégorie
#         for img_name in tqdm(images):
#             img_path = os.path.join(category_path, img_name)
            
#             # Charger et prétraiter l'image
#             img = load_img(img_path, target_size=(224, 224))
#             img_array = img_to_array(img)
#             img_array = preprocess_input(img_array)
#             img_array = np.expand_dims(img_array, axis=0)
            
#             # Extraire les caractéristiques
#             features = base_model.predict(img_array, verbose=0)
            
#             # Ajouter aux listes
#             X.append(features.flatten())
#             y.append(i)  # L'indice de la catégorie comme étiquette
#             image_paths.append(img_path)
    
#     # Convertir en tableaux numpy
#     X = np.array(X)
#     y = np.array(y)
    
#     # Sauvegarder les caractéristiques extraites
#     with open(output_file, 'wb') as f:
#         pickle.dump({
#             'features': X,
#             'labels': y,
#             'paths': image_paths,
#             'categories': categories
#         }, f)
    
#     print(f"Caractéristiques extraites: {X.shape}")
#     print(f"Étiquettes: {y.shape}")
#     print(f"Caractéristiques sauvegardées dans {output_file}")
    
#     return X, y, image_paths, categories

# def visualize_features(features_file):
#     """
#     Visualise les caractéristiques extraites par ACP (Analyse en Composantes Principales)
#     """
#     from sklearn.decomposition import PCA
#     import matplotlib.pyplot as plt
    
#     # Charger les caractéristiques
#     with open(features_file, 'rb') as f:
#         data = pickle.load(f)
    
#     X = data['features']
#     y = data['labels']
#     categories = data['categories']
    
#     # Réduire la dimensionnalité avec ACP
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X)
    
#     # Visualiser
#     plt.figure(figsize=(10, 8))
#     colors = ['blue', 'green', 'red']
    
#     for i, category in enumerate(categories):
#         mask = (y == i)
#         plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], label=category, alpha=0.7)
    
#     plt.title('Visualisation des caractéristiques (ACP)')
#     plt.xlabel('Composante principale 1')
#     plt.ylabel('Composante principale 2')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.show()
    
#     # Afficher la variance expliquée
#     print(f"Variance expliquée par les 2 premières composantes: {pca.explained_variance_ratio_.sum():.2%}")

# if __name__ == "__main__":
#     AUG_DIR = "data/augmented"
#     FEATURES_FILE = "models/features.pkl"
    
#     # Extraire les caractéristiques
#     extract_features(AUG_DIR, FEATURES_FILE)
    
#     # Visualiser les caractéristiques
#     visualize_features(FEATURES_FILE)


# feature_extraction.py


# feature_extraction.py
import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def extract_features(data_dir, output_file):
    """
    Extrait les caractéristiques des images à l'aide d'un modèle préentraîné (MobileNetV2)
    """
    # Charger le modèle préentraîné sans la couche de classification
    base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    
    categories = ['bon', 'moyen', 'mauvais']
    X = []  # Caractéristiques
    y = []  # Étiquettes
    image_paths = []  # Chemins des images pour référence
    
    # Pour chaque catégorie
    for i, category in enumerate(categories):
        print(f"Extraction des caractéristiques pour la catégorie: {category}")
        category_path = os.path.join(data_dir, category)
        
        if not os.path.exists(category_path):
            print(f"Le dossier {category_path} n'existe pas!")
            continue
            
        images = os.listdir(category_path)
        
        # Pour chaque image dans la catégorie
        for img_name in tqdm(images):
            img_path = os.path.join(category_path, img_name)
            
            # Charger et prétraiter l'image
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Extraire les caractéristiques
            features = base_model.predict(img_array, verbose=0)
            
            # Ajouter aux listes
            X.append(features.flatten())
            y.append(i)  # L'indice de la catégorie comme étiquette
            image_paths.append(img_path)
    
    # Convertir en tableaux numpy
    X = np.array(X)
    y = np.array(y)
    
    # Sauvegarder les caractéristiques extraites
    with open(output_file, 'wb') as f:
        pickle.dump({
            'features': X,
            'labels': y,
            'paths': image_paths,
            'categories': categories
        }, f)
    
    print(f"Caractéristiques extraites: {X.shape}")
    print(f"Étiquettes: {y.shape}")
    print(f"Caractéristiques sauvegardées dans {output_file}")
    
    return X, y, image_paths, categories

def visualize_features(features_file):
    """
    Visualise les caractéristiques extraites par ACP (Analyse en Composantes Principales)
    """
    # Charger les caractéristiques
    with open(features_file, 'rb') as f:
        data = pickle.load(f)
    
    X = data['features']
    y = data['labels']
    categories = data['categories']
    
    # Réduire la dimensionnalité avec ACP
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Visualiser
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'red']
    
    for i, category in enumerate(categories):
        mask = (y == i)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], label=category, alpha=0.7)
    
    plt.title('Visualisation des caractéristiques (ACP)')
    plt.xlabel('Composante principale 1')
    plt.ylabel('Composante principale 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Afficher la variance expliquée
    print(f"Variance expliquée par les 2 premières composantes: {pca.explained_variance_ratio_.sum():.2%}")

def save_features_to_csv(features_file, output_csv):
    """
    Convertit les caractéristiques extraites du format pickle en CSV
    """
    # Charger les caractéristiques
    with open(features_file, 'rb') as f:
        data = pickle.load(f)
    
    X = data['features']
    y = data['labels']
    paths = data['paths']
    categories = data['categories']
    
    # Créer un dataframe
    # D'abord, créer les colonnes pour les caractéristiques
    feature_columns = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Créer le dataframe
    df = pd.DataFrame(X, columns=feature_columns)
    
    # Ajouter les colonnes pour les étiquettes et les chemins
    df['label_id'] = y
    df['category'] = [categories[i] for i in y]
    df['image_path'] = paths
    
    # Sauvegarder en CSV
    df.to_csv(output_csv, index=False)
    print(f"Caractéristiques sauvegardées dans {output_csv}")
    
    return df

def create_analysis_csv(features_file, output_csv):
    """
    Crée un fichier CSV plus informatif pour l'analyse des caractéristiques
    """
    # Charger les caractéristiques
    with open(features_file, 'rb') as f:
        data = pickle.load(f)
    
    X = data['features']
    y = data['labels']
    paths = data['paths']
    categories = data['categories']
    
    # 1. Créer un dataframe de base
    df_base = pd.DataFrame({
        'category': [categories[i] for i in y],
        'label_id': y,
        'image_path': paths
    })
    
    # 2. Ajouter des statistiques de base sur les caractéristiques
    df_base['feature_mean'] = np.mean(X, axis=1)
    df_base['feature_std'] = np.std(X, axis=1)
    df_base['feature_min'] = np.min(X, axis=1)
    df_base['feature_max'] = np.max(X, axis=1)
    df_base['feature_median'] = np.median(X, axis=1)
    
    # 3. Ajouter les premiers éléments de l'ACP
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)
    
    for i in range(10):
        df_base[f'pca_component_{i+1}'] = X_pca[:, i]
    
    # 4. Ajouter la variance expliquée par chaque composante
    df_components = pd.DataFrame({
        'component': [f'PCA_{i+1}' for i in range(10)],
        'explained_variance': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
    })
    
    # 5. Sauvegarder les fichiers CSV
    df_base.to_csv(output_csv, index=False)
    df_components.to_csv(output_csv.replace('.csv', '_pca_info.csv'), index=False)
    
    print(f"Analyse des caractéristiques sauvegardée dans {output_csv}")
    print(f"Informations PCA sauvegardées dans {output_csv.replace('.csv', '_pca_info.csv')}")
    
    return df_base, df_components

if __name__ == "__main__":
    AUG_DIR = "data/augmented"
    FEATURES_FILE = "models/features.pkl"
    CSV_FILE = "models/features_analysis.csv"
    
    # Extraire les caractéristiques
    extract_features(AUG_DIR, FEATURES_FILE)
    
    # Créer un CSV pour l'analyse
    create_analysis_csv(FEATURES_FILE, CSV_FILE)
    
    # Visualiser les caractéristiques
    visualize_features(FEATURES_FILE)