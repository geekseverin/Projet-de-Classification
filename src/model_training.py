# model_training.py (suite)
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

def load_features(features_file):
    """Charge les caractéristiques extraites"""
    with open(features_file, 'rb') as f:
        data = pickle.load(f)
    
    X = data['features']
    y = data['labels']
    categories = data['categories']
    
    return X, y, categories

def build_model(input_shape, num_classes):
    """Construit un modèle d'apprentissage profond"""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(features_file, model_output_dir):
    """Entraîne le modèle sur les caractéristiques extraites"""
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Charger les données
    X, y, categories = load_features(features_file)
    num_classes = len(categories)
    
    # Convertir les étiquettes en format one-hot
    y_categorical = to_categorical(y, num_classes=num_classes)
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Dimensions des données d'entraînement: {X_train.shape}")
    print(f"Dimensions des données de test: {X_test.shape}")
    
    # Construire le modèle
    model = build_model(X_train.shape[1], num_classes)
    
    # Callbacks pour l'entraînement
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(
            filepath=os.path.join(model_output_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Entraîner le modèle
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Évaluer le modèle
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Précision sur l'ensemble de test: {test_accuracy:.4f}")
    
    # Sauvegarder le modèle final
    model.save(os.path.join(model_output_dir, 'final_model.h5'))
    
    # Sauvegarder les catégories
    with open(os.path.join(model_output_dir, 'categories.pkl'), 'wb') as f:
        pickle.dump(categories, f)
    
    # Tracer l'historique d'entraînement
    plt.figure(figsize=(12, 5))
    
    # Précision
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entraînement')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Précision du modèle')
    plt.xlabel('Époque')
    plt.ylabel('Précision')
    plt.legend()
    
    # Perte
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entraînement')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Perte du modèle')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_output_dir, 'training_history.png'))
    plt.show()
    
    return model, history, categories

if __name__ == "__main__":
    FEATURES_FILE = "models/features.pkl"
    MODEL_DIR = "models"
    
    # Entraîner le modèle
    model, history, categories = train_model(FEATURES_FILE, MODEL_DIR)