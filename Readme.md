# Projet de Classification de Gboma

Ce projet implémente un système de classification d'images de gboma (un type de légume) en trois catégories: bon, moyen et mauvais. Le système utilise des techniques de deep learning et de traitement d'image pour analyser et classifier les gboma.

## Structure du Projet

```
projet_gboma/
│
├── data/
│   ├── raw/                   # Images originales
│   │   ├── bon/
│   │   ├── moyen/
│   │   └── mauvais/
│   │
│   └── augmented/             # Images après augmentation
│       ├── bon/
│       ├── moyen/
│       └── mauvais/
│
├── src/
│   ├── data_visualization.py  # Visualisation des données
│   ├── data_augmentation.py   # Code pour l'augmentation des données
│   ├── feature_extraction.py  # Extraction des caractéristiques
│   ├── model_training.py      # Entraînement du modèle
│   └── model_evaluation.py    # Évaluation du modèle
│
├── models/                    # Modèles entraînés sauvegardés
│
├── app/
│   └── app.py                 # Application Streamlit
│
├── requirements.txt           # Dépendances du projet
└── README.md                  # Ce fichier
```

## Installation

1. Clonez ce dépôt :
```bash
git clone <url-du-depot>
cd projet_gboma
```

2. Créez un environnement virtuel et activez-le :
```bash
python -m venv venv
# Sur Windows
venv\Scripts\activate
# Sur macOS/Linux
source venv/bin/activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Préparation des Données

1. Collectez des images de gboma et organisez-les dans les dossiers suivants :
   - `data/raw/bon/` : Images de gboma de bonne qualité
   - `data/raw/moyen/` : Images de gboma de qualité moyenne
   - `data/raw/mauvais/` : Images de gboma de mauvaise qualité

2. Visualisez les données :
```bash
cd src
python data_visualization.py
```

## Pipeline d'Exécution

1. **Augmentation des données** :
```bash
python data_augmentation.py
```
Cette étape génère des variations des images originales pour améliorer la robustesse du modèle.

2. **Extraction des caractéristiques** :
```bash
python feature_extraction.py
```
Utilise un modèle préentraîné (MobileNetV2) pour extraire des caractéristiques des images.

3. **Entraînement du modèle** :
```bash
python model_training.py
```
Entraîne un modèle de classification sur les caractéristiques extraites.

4. **Évaluation du modèle** :
```bash
python model_evaluation.py
```
Évalue les performances du modèle et génère des rapports.

## Application de Démonstration

Lancez l'application Streamlit pour démontrer et tester le modèle :
```bash
cd ../app
streamlit run app.py
```

L'application permet de télécharger une image ou de prendre une photo pour classifier un gboma.

## Technologies Utilisées

- Python 3.8+
- TensorFlow/Keras pour le deep learning
- Streamlit pour l'interface utilisateur
- Scikit-learn pour l'évaluation du modèle
- PIL/Pillow pour le traitement d'images
- Matplotlib et Seaborn pour la visualisation

## Méthode

1. **Prétraitement des images** : Redimensionnement et normalisation
2. **Augmentation des données** : Rotation, zoom, décalage, etc.
3. **Extraction des caractéristiques** : Utilisation d'un réseau de neurones préentraîné
4. **Classification** : Réseau de neurones dense sur les caractéristiques extraites
5. **Évaluation** : Analyse de la précision, rappel, matrice de confusion, etc.

## Auteur

[Votre Nom]

## Licence

[Spécifiez votre licence ici]
