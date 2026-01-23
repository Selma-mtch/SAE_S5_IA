# -*- coding: utf-8 -*-
"""
Modèle général amélioré - Version SAE 1
- Stabilisation (Seeds + Full Dataset)
- Pondération des pertes (Loss Weights)
- Métriques avancées (MAE, MSE, Accuracy, AUC)
- Calcul du R² après entraînement
"""

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import kagglehub
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- 0. Reproductibilité (Seeds) ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- 1. Téléchargement et Chargement des Données ---
print("Téléchargement du dataset...")
base_path = kagglehub.dataset_download("jangedoo/utkface-new")
image_folder = os.path.join(base_path, "UTKFace")

print(f"Dataset localisé à : {image_folder}")

image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
print(f"Nombre total de fichiers trouvés : {len(image_files)}")

images = []
labels_age = []
labels_gender = []
labels_race = []

# Paramètres
IMG_SIZE = 128

print("Chargement de l'intégralité du dataset (Full Loading)...")
for file in image_files:
    try:
        parts = file.split("_")
        if len(parts) < 4: continue
        
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])

        img_path = os.path.join(image_folder, file)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
        images.append(np.array(img))
        labels_age.append(age)
        labels_gender.append(gender)
        labels_race.append(race)

    except (ValueError, IndexError):
        continue

images = np.array(images, dtype="float32") / 255.0
labels_age = np.array(labels_age, dtype="float32")
labels_gender = np.array(labels_gender)
labels_race = np.array(labels_race)

# Normalisation de l'âge (Regression)
age_min, age_max = 0, 116
labels_age_norm = (labels_age - age_min) / (age_max - age_min)

# One-hot encoding pour classification
labels_gender_cat = to_categorical(labels_gender, num_classes=2)
labels_race_cat = to_categorical(labels_race, num_classes=5)

# --- 2. Split des données ---
(train_images, test_images, 
 train_age, test_age, 
 train_gender, test_gender, 
 train_race, test_race) = train_test_split(
    images, labels_age_norm, labels_gender_cat, labels_race_cat, 
    test_size=0.2, random_state=SEED
)

# --- 3. Architecture du Modèle ---
def build_model():
    input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.GlobalAveragePooling2D()(x)

    dense = layers.Dense(128, activation='relu')(x)
    dense = layers.Dropout(0.3)(dense)

    age_output = layers.Dense(1, activation='linear', name='age_output')(dense)
    gender_output = layers.Dense(2, activation='softmax', name='gender_output')(dense)
    race_output = layers.Dense(5, activation='softmax', name='race_output')(dense)

    model = models.Model(inputs=input_layer, outputs=[age_output, gender_output, race_output])
    return model

model = build_model()

# Compilation avec PONDÉRATION des pertes (Loss Weights) et nouvelles métriques (AUC)
model.compile(
    optimizer='adam',
    loss={
        'age_output': 'mse',
        'gender_output': 'categorical_crossentropy',
        'race_output': 'categorical_crossentropy'
    },
    loss_weights={
        'age_output': 0.5,      # Réduit l'impact de la régression car les valeurs MSE sont souvent élevées
        'gender_output': 1.0,
        'race_output': 1.0
    },
    metrics={
        'age_output': ['mae', 'mse'],
        'gender_output': ['accuracy', tf.keras.metrics.AUC(name='auc')],
        'race_output': ['accuracy', tf.keras.metrics.AUC(name='auc')]
    }
)

# --- 4. Entraînement ---
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=3)
]

print("Début de l'entraînement sur toutes les données...")
history = model.fit(
    train_images,
    {
        'age_output': train_age,
        'gender_output': train_gender,
        'race_output': train_race
    },
    validation_split=0.1,
    epochs=25,
    batch_size=32,
    callbacks=callbacks
)

# --- 5. Évaluation et métriques avancées (R²) ---
print("\nÉvaluation finale sur le set de test...")
results = model.evaluate(test_images, {
    'age_output': test_age,
    'gender_output': test_gender,
    'race_output': test_race
})

# Calcul du R² pour l'âge
predictions = model.predict(test_images)
age_preds = predictions[0]
r2 = r2_score(test_age, age_preds)
print(f"Coefficient de détermination (R²) pour l'âge : {r2:.4f}")

# --- 6. Visualisations ---
# Courbes d'apprentissage
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['age_output_mae'], label='Age MAE')
plt.plot(history.history['val_age_output_mae'], label='Val Age MAE')
plt.title('MAE de l\'âge')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['gender_output_accuracy'], label='Gender Acc')
plt.plot(history.history['race_output_accuracy'], label='Race Acc')
plt.title('Accuracy des tâches de classification')
plt.legend()
plt.show()

# Exemples de prédictions avec étiquettes claires
gender_map = {0: 'Homme', 1: 'Femme'}
race_map = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Other'}

plt.figure(figsize=(12, 10))
for i in range(4):
    idx = np.random.randint(0, len(test_images))
    img = test_images[idx].reshape(1, IMG_SIZE, IMG_SIZE, 3)
    p = model.predict(img)
    
    p_age = int(p[0][0][0] * (age_max - age_min) + age_min)
    p_gender = gender_map[np.argmax(p[1][0])]
    p_race = race_map[np.argmax(p[2][0])]
    
    t_age = int(test_age[idx] * (age_max - age_min) + age_min)
    t_gender = gender_map[np.argmax(test_gender[idx])]
    t_race = race_map[np.argmax(test_race[idx])]
    
    plt.subplot(2, 2, i+1)
    plt.imshow(test_images[idx])
    plt.title(f"Pred: {p_age}y, {p_gender}, {p_race}\nTrue: {t_age}y, {t_gender}, {t_race}")
    plt.axis('off')
plt.tight_layout()
plt.show()

print("Script terminé. Les métriques exigées (MAE, MSE, Accuracy, AUC, R2) sont calculées.")
