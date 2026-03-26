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
    loss={
        'age_output': 'huber',
        'gender_output': 'categorical_crossentropy',
        'race_output': 'categorical_crossentropy'
    },
    loss_weights={
        'age_output': 1.0,
        'gender_output': 1.0,
        'race_output': 1.5
    },
    metrics={
        'age_output': 'mae',
        'gender_output': 'accuracy',
        'race_output': 'accuracy'
    }
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

# On fusionne les historiques des 2 phases
def merge_histories(h1, h2, key):
    return h1.history.get(key, []) + h2.history.get(key, [])

epochs_total = range(1, len(merge_histories(history1, history2, 'loss')) + 1)
fine_tune_start = len(history1.history['loss'])

# --- Graphique 1 : Loss totale (train vs val) ---
plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
plt.plot(epochs_total, merge_histories(history1, history2, 'loss'), label='Train')
plt.plot(epochs_total, merge_histories(history1, history2, 'val_loss'), label='Validation')
plt.axvline(x=fine_tune_start, color='gray', linestyle='--', label='Début Fine-tuning')
plt.title('Loss totale')
plt.xlabel('Époque')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# --- Graphique 2 : MAE Âge (train vs val) ---
plt.subplot(2, 3, 2)
plt.plot(epochs_total, merge_histories(history1, history2, 'age_output_mae'), label='Train')
plt.plot(epochs_total, merge_histories(history1, history2, 'val_age_output_mae'), label='Validation')
plt.axvline(x=fine_tune_start, color='gray', linestyle='--', label='Début Fine-tuning')
plt.title('MAE - Âge')
plt.xlabel('Époque')
plt.ylabel('MAE (années)')
plt.legend()
plt.grid(True, alpha=0.3)

# --- Graphique 3 : Accuracy Genre (train vs val) ---
plt.subplot(2, 3, 3)
plt.plot(epochs_total, merge_histories(history1, history2, 'gender_output_accuracy'), label='Train')
plt.plot(epochs_total, merge_histories(history1, history2, 'val_gender_output_accuracy'), label='Validation')
plt.axvline(x=fine_tune_start, color='gray', linestyle='--', label='Début Fine-tuning')
plt.title('Accuracy - Genre')
plt.xlabel('Époque')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# --- Graphique 4 : Accuracy Ethnie (train vs val) ---
plt.subplot(2, 3, 4)
plt.plot(epochs_total, merge_histories(history1, history2, 'race_output_accuracy'), label='Train')
plt.plot(epochs_total, merge_histories(history1, history2, 'val_race_output_accuracy'), label='Validation')
plt.axvline(x=fine_tune_start, color='gray', linestyle='--', label='Début Fine-tuning')
plt.title('Accuracy - Ethnie')
plt.xlabel('Époque')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# --- Graphique 5 : Loss par tâche ---
plt.subplot(2, 3, 5)
plt.plot(epochs_total, merge_histories(history1, history2, 'age_output_loss'), label='Âge')
plt.plot(epochs_total, merge_histories(history1, history2, 'gender_output_loss'), label='Genre')
plt.plot(epochs_total, merge_histories(history1, history2, 'race_output_loss'), label='Ethnie')
plt.axvline(x=fine_tune_start, color='gray', linestyle='--', label='Début Fine-tuning')
plt.title('Loss par tâche (Train)')
plt.xlabel('Époque')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# --- Graphique 6 : Résumé des performances finales ---
plt.subplot(2, 3, 6)
metric_names = ['MAE Âge', 'Acc. Genre', 'Acc. Ethnie']
final_train = [
    merge_histories(history1, history2, 'age_output_mae')[-1],
    merge_histories(history1, history2, 'gender_output_accuracy')[-1],
    merge_histories(history1, history2, 'race_output_accuracy')[-1],
]
final_val = [
    merge_histories(history1, history2, 'val_age_output_mae')[-1],
    merge_histories(history1, history2, 'val_gender_output_accuracy')[-1],
    merge_histories(history1, history2, 'val_race_output_accuracy')[-1],
]
x_pos = np.arange(len(metric_names))
plt.bar(x_pos - 0.15, final_train, 0.3, label='Train')
plt.bar(x_pos + 0.15, final_val, 0.3, label='Validation')
plt.xticks(x_pos, metric_names)
plt.title('Performances finales')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('courbes_entrainement_mobilenet.png', dpi=150, bbox_inches='tight')
plt.show()
print("Graphiques sauvegardés dans courbes_entrainement_mobilenet.png")

# --- 7. On sauvegarde notre modèle ---
model.save("model_transfer_mobilenet_v1.keras")
print("Modèle Keras sauvegardé sous model_transfer_mobilenet_v1.keras")

# --- 8. Export TensorFlow Lite ---
print("\nConversion en TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

tflite_path = "model_transfer_mobilenet_v1.tflite"
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

tflite_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
keras_size_mb = os.path.getsize("model_transfer_mobilenet_v1.keras") / (1024 * 1024)
print(f"Modèle TFLite sauvegardé : {tflite_path} ({tflite_size_mb:.2f} Mo)")
print(f"Réduction de taille : {keras_size_mb:.2f} Mo → {tflite_size_mb:.2f} Mo ({(1 - tflite_size_mb/keras_size_mb)*100:.0f}% plus léger)")
print("\nC'est fini !")
