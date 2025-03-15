# data_augmentation.py
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def create_directories(base_dir):
    """Crée les répertoires nécessaires s'ils n'existent pas"""
    categories = ['bon', 'moyen', 'mauvais']
    for category in categories:
        os.makedirs(os.path.join(base_dir, category), exist_ok=True)

def augment_data(input_dir, output_dir, n_samples=5):
    """
    Augmente les données d'images et les enregistre dans output_dir
    """
    categories = ['bon', 'moyen', 'mauvais']
    
    # Définir les transformations pour l'augmentation des données
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Créer les répertoires de sortie
    create_directories(output_dir)
    
    # Pour chaque catégorie
    for category in categories:
        print(f"Traitement de la catégorie: {category}")
        input_path = os.path.join(input_dir, category)
        output_path = os.path.join(output_dir, category)
        
        if not os.path.exists(input_path):
            print(f"Le dossier {input_path} n'existe pas!")
            continue
        
        images = os.listdir(input_path)
        if not images:
            print(f"Pas d'images dans {input_path}")
            continue
        
        # Pour chaque image dans la catégorie
        for img_name in images:
            img_path = os.path.join(input_path, img_name)
            
            # Ouvrir et préparer l'image
            img = Image.open(img_path)
            img = img.convert('RGB')  # Convertir en RGB au cas où
            img = img.resize((224, 224))  # Redimensionner pour cohérence
            x = np.array(img)
            x = x.reshape((1,) + x.shape)  # Reshape pour keras (1, height, width, channels)
            
            # Générer n_samples nouvelles images
            i = 0
            for batch in datagen.flow(x, batch_size=1,
                                     save_to_dir=output_path,
                                     save_prefix=f'aug_{os.path.splitext(img_name)[0]}',
                                     save_format='jpg'):
                i += 1
                if i >= n_samples:
                    break
            
            # Copier également l'image originale
            img.save(os.path.join(output_path, img_name))

def show_augmented_comparison(original_dir, augmented_dir):
    """Affiche des comparaisons entre images originales et augmentées"""
    categories = ['bon', 'moyen', 'mauvais']
    
    for category in categories:
        # Obtenir une image originale
        orig_path = os.path.join(original_dir, category)
        if not os.path.exists(orig_path):
            continue
            
        orig_images = os.listdir(orig_path)
        if not orig_images:
            continue
            
        sample_img = orig_images[0]
        sample_path = os.path.join(orig_path, sample_img)
        
        # Trouver les images augmentées correspondantes
        aug_path = os.path.join(augmented_dir, category)
        aug_images = [img for img in os.listdir(aug_path) if img.startswith(f'aug_{os.path.splitext(sample_img)[0]}')]
        
        # Afficher la comparaison
        plt.figure(figsize=(15, 5))
        
        # Image originale
        plt.subplot(1, len(aug_images) + 1, 1)
        img_orig = Image.open(sample_path)
        plt.imshow(np.array(img_orig))
        plt.title("Original")
        plt.axis('off')
        
        # Images augmentées
        for i, aug_img in enumerate(aug_images):
            aug_img_path = os.path.join(aug_path, aug_img)
            img_aug = Image.open(aug_img_path)
            
            plt.subplot(1, len(aug_images) + 1, i + 2)
            plt.imshow(np.array(img_aug))
            plt.title(f"Augmenté {i+1}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        break  # Montrer seulement un exemple

if __name__ == "__main__":
    RAW_DIR = "data/raw"
    AUG_DIR = "data/augmented"
    
    # Augmenter les données (5 nouvelles images par image originale)
    augment_data(RAW_DIR, AUG_DIR, n_samples=5)
    
    # Afficher une comparaison
    show_augmented_comparison(RAW_DIR, AUG_DIR)
    
    # Afficher des statistiques
    categories = ['bon', 'moyen', 'mauvais']
    for category in categories:
        raw_path = os.path.join(RAW_DIR, category)
        aug_path = os.path.join(AUG_DIR, category)
        
        if os.path.exists(raw_path) and os.path.exists(aug_path):
            raw_count = len(os.listdir(raw_path))
            aug_count = len(os.listdir(aug_path))
            print(f"Catégorie {category}: {raw_count} images originales → {aug_count} après augmentation")