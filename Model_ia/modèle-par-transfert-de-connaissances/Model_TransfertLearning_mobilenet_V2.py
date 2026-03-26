# -*- coding: utf-8 -*-
"""
Modèle MobileNetV2 Multi-tâches optimisé.
Corrections : Data Augmentation, Normalisation Age, et Pondération des pertes.
"""

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# --- 0. Configuration ---
SEED = 42
IMG_SIZE = 128
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- 1. Chargement et Préparation ---
def load_utkface_data(image_folder, limit=None):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
    if limit: image_files = image_files[:limit]
    
    images, ages, genders, races = [], [], [], []

    print(f"Chargement de {len(image_files)} images...")
    for file in image_files:
        try:
            parts = file.split("_")
            if len(parts) < 4: continue
            
            # Normalisation de l'âge (0 à 1) pour équilibrer la Loss
            age = int(parts[0]) / 100.0 
            gender = int(parts[1])
            race = int(parts[2])

            img_path = os.path.join(image_folder, file)
            img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            
            images.append(np.array(img))
            ages.append(age)
            genders.append(gender)
            races.append(race)
        except: continue
            
    return np.array(images), np.array(ages, dtype="float32"), np.array(genders), np.array(races)

import kagglehub
base_path = kagglehub.dataset_download("jangedoo/utkface-new")
image_folder = os.path.join(base_path, "UTKFace")

X, y_age, y_gender, y_race = load_utkface_data(image_folder)
X = preprocess_input(X.astype('float32'))

y_gender_cat = to_categorical(y_gender, num_classes=2)
y_race_cat = to_categorical(y_race, num_classes=5)

(X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test, y_race_train, y_race_test) = train_test_split(
    X, y_age, y_gender_cat, y_race_cat, test_size=0.15, random_state=SEED
)

# --- 2. Architecture du Modèle ---

# Data Augmentation intégrée (active uniquement au .fit)
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])

def build_improved_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False 

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_aug(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    
    # Tronc commun plus puissant
    shared = layers.Dense(512, activation='relu')(x)
    shared = layers.Dropout(0.5)(shared)

    # Branche Âge (Régression normalisée)
    age_d = layers.Dense(256, activation='relu')(shared)
    age_d = layers.BatchNormalization()(age_d)
    age_output = layers.Dense(1, activation='sigmoid', name='age_output')(age_d)

    # Branche Genre (Poids augmenté plus tard)
    gen_d = layers.Dense(256, activation='relu')(shared)
    gen_d = layers.Dropout(0.3)(gen_d)
    gender_output = layers.Dense(2, activation='softmax', name='gender_output')(gen_d)

    # Branche Ethnie
    race_d = layers.Dense(256, activation='relu')(shared)
    race_output = layers.Dense(5, activation='softmax', name='race_output')(race_d)

    model = models.Model(inputs=inputs, outputs=[age_output, gender_output, race_output])
    return model, base_model

model, base_model = build_improved_model()

# --- 3. Entraînement ---

# On force le modèle à se concentrer sur le Genre et l'Age
loss_weights = {'age_output': 4.0, 'gender_output': 5.0, 'race_output': 1.5}

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss={'age_output': 'mse', 'gender_output': 'categorical_crossentropy', 'race_output': 'categorical_crossentropy'},
    loss_weights=loss_weights,
    metrics={'age_output': 'mae', 'gender_output': 'accuracy', 'race_output': 'accuracy'}
)

early_stop = callbacks.EarlyStopping(patience=6, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(factor=0.2, patience=3)

print("\n--- Phase 1 : Entraînement des têtes ---")
model.fit(X_train, {'age_output': y_age_train, 'gender_output': y_gender_train, 'race_output': y_race_train},
          validation_split=0.15, epochs=20, batch_size=64, callbacks=[early_stop, reduce_lr])

print("\n--- Phase 2 : Fine-tuning (Dégel partiel) ---")
base_model.trainable = True
for layer in base_model.layers[:-30]: layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss={'age_output': 'mse', 'gender_output': 'categorical_crossentropy', 'race_output': 'categorical_crossentropy'},
    loss_weights=loss_weights,
    metrics={'age_output': 'mae', 'gender_output': 'accuracy', 'race_output': 'accuracy'}
)

model.fit(X_train, {'age_output': y_age_train, 'gender_output': y_gender_train, 'race_output': y_race_train},
          validation_split=0.15, epochs=10, batch_size=32, callbacks=[early_stop])

# --- 4. Sauvegarde et Export ---
model.save("model_final_optimise.keras")

# Note : Pour lire l'âge réel après prédiction : age_reel = prediction * 100
print("\nModèle optimisé sauvegardé. L'âge est maintenant prédit sur une échelle de 0 à 1.")
