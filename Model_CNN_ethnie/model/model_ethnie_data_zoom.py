# %% [code]
# -*- coding: utf-8 -*-
"""
# Modèle d'ethnicité UTKFace - Version avec Data Augmentation (Zoom uniquement)

**Entraînement sur Kaggle**

**Preprocessing :**
- Images en niveaux de gris (1 canal)
- Redimensionnement 128x128
- Normalisation /255
- Class weights pour équilibrage

**Data Augmentation :**
- Zoom ±10% uniquement

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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

"""## 3. Data Augmentation avec ImageDataGenerator (Zoom uniquement)"""

# Configuration de l'augmentation pour l'entraînement
# Uniquement le zoom ±10%
train_datagen = ImageDataGenerator(
    zoom_range=0.1,         # Zoom ±10%
    fill_mode='nearest',    # Remplissage des pixels manquants
    validation_split=0.2    # 20% pour la validation
)

# Pas d'augmentation pour la validation (images originales)
val_datagen = ImageDataGenerator(
    validation_split=0.2
)

# Créer les générateurs
BATCH_SIZE = 64

train_generator = train_datagen.flow(
    X_train, y_train_cat,
    batch_size=BATCH_SIZE,
    subset='training',
    shuffle=True
)

# Validation sans augmentation (images originales)
val_generator = val_datagen.flow(
    X_train, y_train_cat,
    batch_size=BATCH_SIZE,
    subset='validation',
    shuffle=False
)

print(f"Échantillons d'entraînement : {train_generator.n}")
print(f"Échantillons de validation : {val_generator.n}")

"""## 4. Création du modèle CNN"""

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

"""## 5. Entraînement avec Data Augmentation à la volée"""

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Calculer les steps par epoch
steps_per_epoch = train_generator.n // BATCH_SIZE
validation_steps = val_generator.n // BATCH_SIZE

history = model.fit(
    train_generator,
    epochs=30,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=[early_stop],
    class_weight=class_weight_dict
)

"""## 6. Visualisation de l'entraînement"""

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
plt.savefig(os.path.join(OUTPUT_PATH, 'training_curves_augmented.png'), dpi=150)
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

"""## 7. Évaluation du modèle"""

y_pred = model.predict(X_test).argmax(axis=1)

loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"\nAccuracy sur le test set : {accuracy*100:.2f}%")

eth_labels = ['Blanc', 'Noir', 'Asiatique', 'Indien', 'Autre']
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=eth_labels))

"""## 8. Matrice de confusion"""

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
plt.title('Matrice de confusion - Modèle avec Data Augmentation (Zoom)')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix_augmented.png'), dpi=150)
plt.show()

"""## 9. Sauvegarde du modèle"""

model.save(os.path.join(OUTPUT_PATH, 'ethnicity_model_augmented.keras'))
print(f"Modèle sauvegardé : {OUTPUT_PATH}/ethnicity_model_augmented.keras")

print("\n→ Les fichiers sont disponibles dans l'onglet 'Output' de Kaggle pour téléchargement.")

"""## 10. Analyse des performances par classe"""

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
ax.set_title('Performances par classe - Modèle avec Data Augmentation (Zoom)')
ax.set_xticks(x)
ax.set_xticklabels(eth_labels)
ax.legend()
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'metrics_per_class_augmented.png'), dpi=150)
plt.show()

# Résumé final
print("=" * 60)
print("RÉSUMÉ FINAL - MODÈLE AVEC DATA AUGMENTATION (ZOOM)")
print("=" * 60)
print(f"\nAccuracy globale : {accuracy*100:.2f}%")
print(f"\nPerformances par classe :")
for i, label in enumerate(eth_labels):
    print(f"  {label:10s} - Precision: {precision[i]*100:5.1f}% | Recall: {recall[i]*100:5.1f}% | F1: {f1[i]*100:5.1f}% | Support: {support[i]}")

print(f"\n" + "=" * 60)
print("FICHIERS SAUVEGARDÉS")
print("=" * 60)
print("  - ethnicity_model_augmented.keras")
print("  - training_curves_augmented.png")
print("  - confusion_matrix_augmented.png")
print("  - metrics_per_class_augmented.png")

"""## 11. Visualisation des augmentations (Optionnel)"""

def visualize_augmentations(image, datagen, n_examples=6):
    """Visualise les différentes augmentations appliquées à une image."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.flatten()

    # Image originale
    axes[0].imshow(image.squeeze(), cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Générer des versions augmentées
    img_array = image.reshape((1,) + image.shape)
    aug_iter = datagen.flow(img_array, batch_size=1)

    for i in range(1, n_examples):
        aug_img = next(aug_iter)[0]
        aug_img = np.clip(aug_img, 0, 1)  # Évite les valeurs hors [0,1]
        axes[i].imshow(aug_img.squeeze(), cmap='gray')
        axes[i].set_title(f'Zoom {i}')
        axes[i].axis('off')

    plt.suptitle('Exemples de Data Augmentation (Zoom ±10%)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'augmentation_examples.png'), dpi=150)
    plt.show()

# Visualiser les augmentations sur une image exemple
print("\nVisualisation des augmentations :")
visualize_augmentations(X_train[0], train_datagen)
