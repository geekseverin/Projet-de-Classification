# data_visualization.py
import os
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np

def show_images(directory, n=5):
    """
    Affiche n images aléatoires de chaque catégorie
    """
    categories = ['bon', 'moyen', 'mauvais']
    
    plt.figure(figsize=(15, 5*len(categories)))
    
    for i, category in enumerate(categories):
        path = os.path.join(directory, category)
        if not os.path.exists(path):
            print(f"Le dossier {path} n'existe pas!")
            continue
            
        images = os.listdir(path)
        if not images:
            print(f"Pas d'images dans {path}")
            continue
            
        sample = random.sample(images, min(n, len(images)))
        
        for j, img_name in enumerate(sample):
            img_path = os.path.join(path, img_name)
            img = Image.open(img_path)
            
            plt.subplot(len(categories), n, i*n + j + 1)
            plt.imshow(np.array(img))
            plt.title(f"{category} - {img_name}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Assurez-vous que ce chemin pointe vers votre dossier d'images
    DATA_DIR = "data/raw"
    show_images(DATA_DIR)
    
    # Afficher quelques statistiques sur les données
    categories = ['bon', 'moyen', 'mauvais']
    for category in categories:
        path = os.path.join(DATA_DIR, category)
        if os.path.exists(path):
            num_images = len(os.listdir(path))
            print(f"Catégorie {category}: {num_images} images")