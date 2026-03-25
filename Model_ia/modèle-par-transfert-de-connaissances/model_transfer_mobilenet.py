# -*- coding: utf-8 -*-
"""
Notre modèle final en Transfer Learning avec MobileNetV2.
On essaie de prédire l'âge, le genre et l'ethnie en même temps 
pour que ce soit plus rapide et efficace.
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

# --- 0. On prépare nos outils et on fixe les seeds pour avoir les mêmes résultats ---
SEED = 42
IMG_SIZE = 128
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- 1. Ici on charge nos photos de UTKFace ---
def load_utkface_data(image_folder, limit=None):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
    if limit:
        image_files = image_files[:limit]
    
    images = []
    labels_age = []
    labels_gender = []
    labels_race = []

    print(f"Chargement de {len(image_files)} images...")
    for file in image_files:
        try:
            # On découpe le nom du fichier pour récupérer les labels (age_genre_ethnie)
            parts = file.split("_")
            if len(parts) < 4: continue
            
            age = int(parts[0])
            gender = int(parts[1])
            race = int(parts[2])

            img_path = os.path.join(image_folder, file)
            # On redimensionne direct en 128x128 pour que ce soit moins lourd
            img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            
            images.append(np.array(img))
            labels_age.append(age)
            labels_gender.append(gender)
            labels_race.append(race)
        except:
            continue
            
    return (np.array(images), 
            np.array(labels_age, dtype="float32"), 
            np.array(labels_gender), 
            np.array(labels_race))

# On cherche où est le dataset (Kaggle ou local)
import kagglehub
print("Recherche du dataset...")
base_path = kagglehub.dataset_download("jangedoo/utkface-new")
image_folder = os.path.join(base_path, "UTKFace")

X, y_age, y_gender, y_race = load_utkface_data(image_folder)

# On normalise les images pour MobileNet (entre -1 et 1)
X = preprocess_input(X.astype('float32'))

# On prépare nos étiquettes pour la classification
y_gender_cat = to_categorical(y_gender, num_classes=2)
y_race_cat = to_categorical(y_race, num_classes=5)

# On sépare nos données (80% pour l'entraînement, 20% pour le test)
(X_train, X_test, 
 y_age_train, y_age_test, 
 y_gender_train, y_gender_test, 
 y_race_train, y_race_test) = train_test_split(
    X, y_age, y_gender_cat, y_race_cat, 
    test_size=0.2, random_state=SEED
)

# --- 2. Construction de notre modèle branché ---
def build_transfer_model():
    # On prend MobileNetV2 déjà entraîné sur ImageNet comme base
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False # Pour l'instant on ne touche pas au modèle de base

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    
    # On crée un petit tronc commun avant de se séparer en 3
    shared = layers.Dense(256, activation='relu')(x)
    shared = layers.Dropout(0.4)(shared)

    # Branche pour l'Âge (c'est une régression linéaire)
    age_dense = layers.Dense(128, activation='relu')(shared)
    age_output = layers.Dense(1, activation='linear', name='age_output')(age_dense)

    # Branche pour le Genre (Homme ou Femme)
    gender_dense = layers.Dense(64, activation='relu')(shared)
    gender_output = layers.Dense(2, activation='softmax', name='gender_output')(gender_dense)

    # Branche pour l'Ethnie (5 catégories)
    race_dense = layers.Dense(128, activation='relu')(shared)
    race_output = layers.Dense(5, activation='softmax', name='race_output')(race_dense)

    model = models.Model(inputs=inputs, outputs=[age_output, gender_output, race_output])
    return model, base_model

model, base_model = build_transfer_model()

# --- 3. Phase 1 : On entraîne seulement nos nouvelles couches ---
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss={
        'age_output': 'huber',
        'gender_output': 'categorical_crossentropy',
        'race_output': 'categorical_crossentropy'
    },
    loss_weights={
        'age_output': 1.0,
        'gender_output': 1.0,
        'race_output': 1.5 # On booste l'ethnie car c'est le plus dur à prédire
    },
    metrics={
        'age_output': 'mae',
        'gender_output': 'accuracy',
        'race_output': 'accuracy'
    }
)

early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True)

print("\n--- Étape 1 : On entraîne nos têtes de prédiction ---")
history1 = model.fit(
    X_train, 
    {'age_output': y_age_train, 'gender_output': y_gender_train, 'race_output': y_race_train},
    validation_split=0.2,
    epochs=15,
    batch_size=32,
    callbacks=[early_stop]
)

# --- 4. Phase 2 : On fait du Fine-tuning pour plus de précision ---
print("\n--- Étape 2 : On ajuste la base du modèle ---")
base_model.trainable = True
# On ne débloque que le haut du modèle (les 20 dernières couches)
for layer in base_model.layers[:-20]:
    layer.trainable = False

# On baisse le learning rate pour ne pas tout casser
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss=model.loss,
    loss_weights=model.loss_weights,
    metrics=model.metrics
)

history2 = model.fit(
    X_train, 
    {'age_output': y_age_train, 'gender_output': y_gender_train, 'race_output': y_race_train},
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[early_stop]
)

# --- 5. On teste nos performances finales ---
print("\nÉvaluation finale...")
results = model.evaluate(X_test, {
    'age_output': y_age_test,
    'gender_output': y_gender_test,
    'race_output': y_race_test
})

# --- 6. On affiche nos courbes pour voir si on a bien appris ---
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history1.history['loss'] + history2.history['loss'], label='Total Loss')
plt.title('Notre courbe de perte')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history1.history['race_output_accuracy'] + history2.history['race_output_accuracy'], label='Race Acc')
plt.plot(history1.history['gender_output_accuracy'] + history2.history['gender_output_accuracy'], label='Gender Acc')
plt.title('Nos scores de précision')
plt.legend()
plt.show()

# On sauvegarde notre bébé
model.save("model_transfer_mobilenet_v1.keras")
print("C'est fini ! Modèle sauvegardé sous model_transfer_mobilenet_v1.keras")
