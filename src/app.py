# app.py
import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import io
import base64

# Définir les chemins
#MODEL_PATH = '../models/best_model.h5'
#CATEGORIES_PATH = '../models/categories.pkl'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Dossier où se trouve app.py
PARENT_DIR = os.path.dirname(BASE_DIR)  # Aller un niveau en arrière

MODEL_PATH = os.path.join(PARENT_DIR, "models", "best_model.h5")
CATEGORIES_PATH = os.path.join(PARENT_DIR, "models", "categories.pkl")

# Fonction pour charger l'image et extraire les caractéristiques
@st.cache_data
def load_model_and_categories(model_path, categories_path):
    """Charge le modèle et les catégories"""
    model = load_model(model_path)
    with open(categories_path, 'rb') as f:
        categories = pickle.load(f)
    return model, categories

def extract_features(img, base_model):
    """Extrait les caractéristiques de l'image"""
    # Redimensionner l'image
    img = img.resize((224, 224))
    # Convertir en tableau numpy
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # Prétraiter l'image
    img_array = preprocess_input(img_array)
    # Ajouter une dimension pour le lot
    img_array = np.expand_dims(img_array, axis=0)
    # Extraire les caractéristiques
    features = base_model.predict(img_array)
    return features.flatten()

def predict_image(img, model, categories, base_model):
    """Prédit la classe de l'image"""
    # Extraire les caractéristiques
    features = extract_features(img, base_model)
    # Faire la prédiction
    prediction = model.predict(np.expand_dims(features, axis=0))
    # Obtenir la classe prédite
    predicted_class = np.argmax(prediction)
    # Obtenir la confiance
    confidence = float(prediction[0][predicted_class])
    # Retourner la classe et la confiance
    return categories[predicted_class], confidence, prediction[0]

def main():
    st.title("Classification de Gboma")
    st.write("Cette application permet de classifier les images de gboma en trois catégories: bon, moyen, mauvais.")
    
    # Charger le modèle
    try:
        # Charger le modèle de classification
        model, categories = load_model_and_categories(MODEL_PATH, CATEGORIES_PATH)
        
        # Charger le modèle d'extraction de caractéristiques
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            pooling='avg'
        )
        
        st.success("Modèles chargés avec succès!")
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {e}")
        return
    
    # Interface utilisateur
    upload_option = st.radio(
        "Comment souhaitez-vous charger l'image?",
        ("Télécharger une image", "Prendre une photo")
    )
    
    if upload_option == "Télécharger une image":
        uploaded_file = st.file_uploader("Choisissez une image de gboma...", type=["jpg", "jpeg", "png"])
        image_source = uploaded_file
    else:
        image_source = st.camera_input("Prenez une photo de gboma")
    
    if image_source is not None:
        # Afficher l'image
        img = Image.open(image_source).convert('RGB')
        st.image(img, caption="Image chargée", use_column_width=True)
        
        # Bouton pour classifier
        if st.button("Classifier"):
            with st.spinner("Classification en cours..."):
                # Faire la prédiction
                category, confidence, all_probs = predict_image(img, model, categories, base_model)
                
                # Afficher le résultat
                st.success(f"Prédiction: **{category}** avec une confiance de {confidence:.2%}")
                
                # Afficher toutes les probabilités
                fig, ax = plt.subplots(figsize=(8, 3))
                bars = ax.bar(categories, all_probs * 100)
                
                # Colorer la barre de la prédiction
                for i, bar in enumerate(bars):
                    if categories[i] == category:
                        bar.set_color('green')
                
                ax.set_ylabel('Probabilité (%)')
                ax.set_title('Probabilités par catégorie')
                ax.set_ylim(0, 100)
                
                for i, v in enumerate(all_probs):
                    ax.text(i, v * 100 + 2, f"{v * 100:.1f}%", ha='center')
                
                st.pyplot(fig)
                
                # Conseils basés sur la prédiction
                if category == "bon":
                    st.info("Ce gboma est de bonne qualité! Parfait pour la consommation.")
                elif category == "moyen":
                    st.warning("Ce gboma est de qualité moyenne. Il peut être consommé mais surveillez sa qualité.")
                else:
                    st.error("Ce gboma est de mauvaise qualité. Il n'est pas recommandé pour la consommation.")

if __name__ == "__main__":
    main()