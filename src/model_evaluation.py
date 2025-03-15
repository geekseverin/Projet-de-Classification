# model_evaluation.py
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
import seaborn as sns

def load_data(features_file):
    """Charge les caractéristiques extraites"""
    with open(features_file, 'rb') as f:
        data = pickle.load(f)
    
    X = data['features']
    y = data['labels']
    paths = data['paths']
    categories = data['categories']
    
    return X, y, paths, categories

def evaluate_model(model_dir, features_file):
    """Évalue le modèle entraîné"""
    # Charger le modèle
    model_path = os.path.join(model_dir, 'best_model.h5')
    model = load_model(model_path)
    
    # Charger les catégories
    with open(os.path.join(model_dir, 'categories.pkl'), 'rb') as f:
        categories = pickle.load(f)
    
    # Charger les données
    X, y_true, paths, _ = load_data(features_file)
    
    # Faire des prédictions
    y_pred_prob = model.predict(X)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Créer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    
    # Afficher la matrice de confusion
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Prédiction')
    plt.ylabel('Vérité')
    plt.title('Matrice de confusion')
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
    plt.show()
    
    # Afficher le rapport de classification
    report = classification_report(y_true, y_pred, target_names=categories)
    print("\nRapport de classification:")
    print(report)
    
    # Sauvegarder le rapport de classification
    with open(os.path.join(model_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Afficher quelques exemples mal classés
    misclassified = np.where(y_pred != y_true)[0]
    
    if len(misclassified) > 0:
        plt.figure(figsize=(15, min(len(misclassified), 10) * 3))
        
        for i, idx in enumerate(misclassified[:10]):  # Afficher au maximum 10 exemples
            # Charger l'image
            from PIL import Image
            img_path = paths[idx]
            img = Image.open(img_path)
            
            # Afficher l'image
            plt.subplot(min(len(misclassified), 10), 1, i + 1)
            plt.imshow(np.array(img))
            plt.title(f"Vrai: {categories[y_true[idx]]}, Prédit: {categories[y_pred[idx]]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'misclassified_examples.png'))
        plt.show()
    
    return report, cm

if __name__ == "__main__":
    FEATURES_FILE = "models/features.pkl"
    MODEL_DIR = "models"
    
    # Évaluer le modèle
    report, cm = evaluate_model(MODEL_DIR, FEATURES_FILE)