# %% [code]
# -*- coding: utf-8 -*-
"""
# Modèle d'ethnicité UTKFace - Version de base

**Entraînement sur Kaggle**

**Preprocessing :**
- Images en niveaux de gris (1 canal)
- Redimensionnement 128x128
- Normalisation /255
- Class weights pour équilibrage

**Pas de Data Augmentation**

**Dataset :** jangedoo/utkface-new

## 1. Chargement des données (Kaggle)
"""

import os
import numpy as np
from PIL import Image

# Chemins Kaggle
KAGGLE_INPUT_PATH = "/kaggle/input/utkface-new"  # Dataset d'entrée
OUTPUT_PATH = "/kaggle/working"  # Dossier de sortie pour les fichiers générés

# Trouver automatiquement le dossier contenant les images
possible_folders = ["UTKFace", "utkface_aligned_cropped", "crop_part1", ""]
image_folder = None

for folder in possible_folders:
    test_path = os.path.join(KAGGLE_INPUT_PATH, folder) if folder else KAGGLE_INPUT_PATH
    if os.path.exists(test_path):
        files = os.listdir(test_path)
        jpg_files = [f for f in files if f.endswith(".jpg")]
        if jpg_files:
            image_folder = test_path
            print(f"Dossier d'images trouvé : {image_folder}")
            break

if image_folder is None:
    raise FileNotFoundError("Impossible de trouver le dossier contenant les images UTKFace. Vérifiez que le dataset 'jangedoo/utkface-new' est ajouté au notebook.")

image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
print(f"Nombre de fichiers .jpg trouvés : {len(image_files)}")

images = []
labels = []

for file in image_files:
    try:
        parts = file.split("_")
        age = int(parts[0])
        gender = int(parts[1])
        try:
            race = int(parts[2])
        except:
            race = 4

        # Conversion en niveaux de gris et redimensionnement 128x128
        img = Image.open(os.path.join(image_folder, file)).convert("L").resize((128, 128))
        images.append(np.array(img))
        labels.append([age, gender, race])
    except:
        continue

images = np.array(images)
labels = np.array(labels)
print(f"Images chargées : {len(images)}")
print(f"Shape des images : {images.shape}")

"""## 2. Imports et préparation des données"""

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

print(f"TensorFlow version : {tf.__version__}")
print(f"GPU disponible : {tf.config.list_physical_devices('GPU')}")

# Extraire X et y_ethnicity
X = images
y_ethnicity = labels[:, 2]  # La 3ème colonne = ethnie

# Ajouter une dimension pour le canal (niveaux de gris = 1 canal)
X = X.reshape(X.shape[0], 128, 128, 1)
print(f"Shape après reshape : {X.shape}")

# Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_ethnicity,
    test_size=0.2,
    random_state=42
)

# Normalisation simple /255
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(f"X_train min/max : {X_train.min():.2f} / {X_train.max():.2f}")

# Calcul des class weights pour équilibrer le dataset
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights : {class_weight_dict}")

# One-hot encoding (5 classes d'ethnicité)
y_train_cat = to_categorical(y_train, num_classes=5)
y_test_cat = to_categorical(y_test, num_classes=5)

print(f"X_train : {X_train.shape}")
print(f"X_test : {X_test.shape}")

"""## 3. Création du modèle CNN"""

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

"""## 4. Entraînement"""

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train_cat,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    class_weight=class_weight_dict
)

"""## 5. Visualisation de l'entraînement"""

# Graphiques de base : Loss et Accuracy
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train', linewidth=2, marker='o', markersize=4)
axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2, marker='s', markersize=4)
axes[0].set_title('Loss durant l\'entraînement', fontsize=12)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Train', linewidth=2, marker='o', markersize=4)
axes[1].plot(history.history['val_accuracy'], label='Validation', linewidth=2, marker='s', markersize=4)
axes[1].set_title('Accuracy durant l\'entraînement', fontsize=12)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'training_curves_base.png'), dpi=150)
plt.show()

# Résumé de l'entraînement
print("=" * 50)
print("RÉSUMÉ DE L'ENTRAÎNEMENT")
print("=" * 50)
print(f"Nombre d'epochs effectuées : {len(history.history['loss'])}")
print(f"\nMeilleure accuracy validation : {max(history.history['val_accuracy'])*100:.2f}%")
print(f"Meilleure loss validation : {min(history.history['val_loss']):.4f}")
print(f"\nAccuracy finale (train) : {history.history['accuracy'][-1]*100:.2f}%")
print(f"Accuracy finale (val) : {history.history['val_accuracy'][-1]*100:.2f}%")

"""## 6. Évaluation du modèle"""

y_pred_proba = model.predict(X_test)
y_pred = y_pred_proba.argmax(axis=1)

loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"\nAccuracy sur le test set : {accuracy*100:.2f}%")

# AUC et AP (métriques multi-classes)
auc_score = roc_auc_score(y_test_cat, y_pred_proba, multi_class='ovr', average='macro')
ap_score = average_precision_score(y_test_cat, y_pred_proba, average='macro')
print(f"AUC (macro) : {auc_score:.4f}")
print(f"AP (macro) : {ap_score:.4f}")

eth_labels = ['Blanc', 'Noir', 'Asiatique', 'Indien', 'Autre']
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=eth_labels))

"""## 7. Matrice de confusion"""

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=eth_labels,
    yticklabels=eth_labels
)
plt.title('Matrice de confusion - Modèle de base (Niveaux de gris)')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix_base.png'), dpi=150)
plt.show()

"""## 8. Sauvegarde du modèle"""

model.save(os.path.join(OUTPUT_PATH, 'ethnicity_model_base.keras'))
print(f"Modèle sauvegardé : {OUTPUT_PATH}/ethnicity_model_base.keras")

print("\n→ Les fichiers sont disponibles dans l'onglet 'Output' de Kaggle pour téléchargement.")

"""## 9. Analyse des performances par classe"""

from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

# Graphique des métriques par classe
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(eth_labels))
width = 0.25

bars1 = ax.bar(x - width, precision * 100, width, label='Precision', color='steelblue')
bars2 = ax.bar(x, recall * 100, width, label='Recall', color='teal')
bars3 = ax.bar(x + width, f1 * 100, width, label='F1-Score', color='coral')

ax.set_ylabel('Score (%)')
ax.set_title('Performances par classe - Modèle de base')
ax.set_xticks(x)
ax.set_xticklabels(eth_labels)
ax.legend()
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'metrics_per_class_base.png'), dpi=150)
plt.show()

# Résumé final
print("=" * 60)
print("RÉSUMÉ FINAL - MODÈLE DE BASE")
print("=" * 60)
print(f"\nAccuracy globale : {accuracy*100:.2f}%")
print(f"\nPerformances par classe :")
for i, label in enumerate(eth_labels):
    print(f"  {label:10s} - Precision: {precision[i]*100:5.1f}% | Recall: {recall[i]*100:5.1f}% | F1: {f1[i]*100:5.1f}% | Support: {support[i]}")

print(f"\n" + "=" * 60)
print("FICHIERS SAUVEGARDÉS")
print("=" * 60)
print("  - ethnicity_model_base.keras")
print("  - training_curves_base.png")
print("  - confusion_matrix_base.png")
print("  - metrics_per_class_base.png")
