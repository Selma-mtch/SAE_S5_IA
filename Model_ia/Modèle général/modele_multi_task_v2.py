# -*- coding: utf-8 -*-
"""
Modèle Multi-Tâches Optimal 
Ce script regroupe nos meilleures techniques pour l'âge, le genre et l'ethnie.
Optimisé pour Kaggle avec export TFLite intégré.
"""

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, Input, backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- 0. Fixer les graines pour la reproductivité ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- 1. Paramètres et Chargement ---
IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 60

# Recherche automatique du dossier UTKFace sur Kaggle
def search_dataset(base="/kaggle/input"):
    for root, dirs, files in os.walk(base):
        if any(f.endswith(".jpg") for f in files) and len(files) > 100:
            return root
    return None

image_folder = search_dataset()
if not image_folder:
    # Fallback si on est en local ou autre
    import kagglehub
    path = kagglehub.dataset_download("jangedoo/utkface-new")
    image_folder = os.path.join(path, "UTKFace")

print(f"Chargement des images depuis : {image_folder}")

def load_data(folder):
    imgs, age, gen, eth = [], [], [], []
    files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    
    for f in files:
        try:
            parts = f.split('_')
            if len(parts) < 3: continue
            
            # On extrait tout d'abord (pour éviter les décalages de liste si ça plante)
            age_val = int(parts[0])
            gen_val = int(parts[1])
            eth_val = int(parts[2])
            
            img = Image.open(os.path.join(folder, f)).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
            
            # On n'ajoute que si tout est réussi
            imgs.append(np.array(img))
            age.append(age_val)
            gen.append(gen_val)
            eth.append(eth_val)
        except: continue
            
    return np.array(imgs), np.array(age), np.array(gen), np.array(eth)

X, y_age, y_gen, y_eth = load_data(image_folder)

# Normalisation : [-1, 1] pour aider la convergence du réseau profond
X = (X.astype('float32') - 127.5) / 127.5
y_age_norm = y_age / 100.0 # On ramène l'âge sur [0, 1] pour équilibrer la Loss
y_gen_cat = to_categorical(y_gen, 2)
y_eth_cat = to_categorical(y_eth, 5)

# Split 80/20
(X_train, X_test, age_tr, age_ts, gen_tr, gen_ts, eth_tr, eth_ts) = train_test_split(
    X, y_age_norm, y_gen_cat, y_eth_cat, test_size=0.2, random_state=SEED
)

# --- 2. Fonctions de Perte Avancées ---

def focal_loss(gamma=2.0, alpha=0.25):
    """Focal Loss pour aider le modèle sur les ethnies minoritaires."""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return K.mean(K.sum(loss, axis=1))
    return focal_loss_fixed

# --- 3. Architecture du Modèle ---

# Data Augmentation intégrée au graphe (active seulement en train)
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="augmentation")

def build_big_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_aug(inputs)
    
    # Backbone profond (5 blocs)
    for filters in [32, 64, 128, 256, 512]:
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    
    # Branche Âge (Regression)
    age_b = layers.Dense(256, activation='relu')(x)
    age_b = layers.Dense(128, activation='relu')(age_b)
    age_out = layers.Dense(1, activation='sigmoid', name='age')(age_b) # Sigmoid car age borné [0,1]

    # Branche Genre (Classification)
    gen_b = layers.Dense(128, activation='relu')(x)
    gen_b = layers.Dropout(0.3)(gen_b)
    gen_out = layers.Dense(2, activation='softmax', name='gender')(gen_b)

    # Branche Ethnie (Classification complexe)
    eth_b = layers.Dense(256, activation='relu')(x)
    eth_b = layers.Dense(128, activation='relu')(eth_b)
    eth_b = layers.Dropout(0.4)(eth_b)
    eth_out = layers.Dense(5, activation='softmax', name='ethnicity')(eth_b)

    return models.Model(inputs, [age_out, gen_out, eth_out], name="MultiTask_Optimal")

model = build_big_model()

# Compilation : on pondère l'ethnie car c'est le plus dur
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        'age': tf.keras.losses.Huber(delta=1.0),
        'gender': 'categorical_crossentropy',
        'ethnicity': focal_loss(gamma=2.0)
    },
    loss_weights={'age': 1.0, 'gender': 1.0, 'ethnicity': 2.0},
    metrics={'age': 'mae', 'gender': 'accuracy', 'ethnicity': 'accuracy'}
)

# --- 4. Entraînement ---
callbacks = [
    EarlyStopping(patience=8, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6)
]

print("Démarrage de l'entraînement...")
history = model.fit(
    X_train, {'age': age_tr, 'gender': gen_tr, 'ethnicity': eth_tr},
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# --- 5. Évaluation et Rapports ---
print("\nÉvaluation sur le set de test...")
results = model.evaluate(X_test, {'age': age_ts, 'gender': gen_ts, 'ethnicity': eth_ts})

# Affichage des courbes
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(history.history['age_mae'], label='Train')
plt.plot(history.history['val_age_mae'], label='Val')
plt.title('MAE Age (Normalisé)')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['gender_accuracy'], label='Train')
plt.plot(history.history['val_gender_accuracy'], label='Val')
plt.title('Accuracy Genre')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history['ethnicity_accuracy'], label='Train')
plt.plot(history.history['val_ethnicity_accuracy'], label='Val')
plt.title('Accuracy Ethnie')
plt.legend()
plt.show()

# --- 6. Export TFLite ---
print("\nConversion en TFLite...")
inference_model = models.Model(inputs=model.input, outputs=model.outputs)
converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("model_multitask_optimal.tflite", "wb") as f:
    f.write(tflite_model)

print("Export terminé : model_multitask_optimal.tflite")
print(f"MAE Age réel estimé : {results[4] * 100:.2f} ans")
