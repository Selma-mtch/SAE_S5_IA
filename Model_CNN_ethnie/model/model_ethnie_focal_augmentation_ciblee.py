# %% [code]
# -*- coding: utf-8 -*-
"""
# Modèle d'ethnicité UTKFace - Focal Loss + Augmentation Ciblée

**Entraînement sur Kaggle**

**Améliorations :**
- Focal Loss au lieu de Cross-Entropy (focus sur les cas difficiles)
- Augmentation ciblée : plus d'augmentations pour les classes minoritaires
  - Rotation ±15°, Zoom ±10%, Flip horizontal, Shift H/V ±10%
  - Sans modification de brightness
- Learning rate réduit (0.0001) pour convergence stable
- Batch size 128 pour gradients moins bruités
- ReduceLROnPlateau : divise LR par 2 après 3 epochs sans amélioration
- EarlyStopping patience=7, max 50 epochs

**Preprocessing :**
- Images en niveaux de gris (1 canal)
- Redimensionnement 128x128
- Normalisation /255

**Dataset :** jangedoo/utkface-new

## 1. Chargement des données (Kaggle)
"""

import os
import numpy as np
from PIL import Image
from collections import Counter

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
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
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

"""## 3. Focal Loss - Fonction personnalisée"""

def focal_loss(gamma=3.0, alpha=None):
    """
    Focal Loss pour gérer le déséquilibre de classes.

    Formule: FL(p) = -alpha * (1-p)^gamma * log(p)

    Args:
        gamma: Facteur de focalisation (défaut=3.0)
               - gamma=0 équivaut à cross-entropy standard
               - gamma>0 réduit la contribution des exemples bien classifiés
        alpha: Poids par classe (optionnel, liste de 5 valeurs pour 5 classes)

    Returns:
        Fonction de loss compatible avec Keras
    """
    def focal_loss_fixed(y_true, y_pred):
        # Éviter log(0) en clippant les prédictions
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Calcul de la cross-entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Facteur de modulation focal: (1 - p)^gamma
        # p = probabilité prédite pour la vraie classe
        p_t = K.sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = K.pow(1.0 - p_t, gamma)

        # Application du facteur focal
        focal_cross_entropy = focal_weight * cross_entropy

        # Application des poids alpha par classe si fournis
        if alpha is not None:
            alpha_tensor = K.constant(alpha, dtype=K.floatx())
            alpha_weight = K.sum(y_true * alpha_tensor, axis=-1, keepdims=True)
            focal_cross_entropy = alpha_weight * focal_cross_entropy

        # Somme sur les classes, moyenne sur le batch
        loss = K.sum(focal_cross_entropy, axis=-1)
        return K.mean(loss)

    return focal_loss_fixed


"""## 4. Augmentation Ciblée - Plus d'augmentations pour classes minoritaires"""

def apply_augmentation(image, aug_type):
    """
    Applique une augmentation spécifique à une image.

    Args:
        image: Image numpy array (128, 128, 1)
        aug_type: Type d'augmentation ('rotation', 'zoom', 'flip', 'shift')

    Returns:
        Image augmentée
    """
    img = tf.convert_to_tensor(image, dtype=tf.float32)

    if aug_type == 'rotation_pos':
        # Rotation +15 degrés
        img = tf.keras.preprocessing.image.random_rotation(
            image, 15, row_axis=0, col_axis=1, channel_axis=2
        )
    elif aug_type == 'rotation_neg':
        # Rotation -15 degrés
        img = tf.keras.preprocessing.image.random_rotation(
            image, 15, row_axis=0, col_axis=1, channel_axis=2
        )
    elif aug_type == 'zoom_in':
        # Zoom in 10%
        img = tf.keras.preprocessing.image.random_zoom(
            image, (0.9, 0.9), row_axis=0, col_axis=1, channel_axis=2
        )
    elif aug_type == 'zoom_out':
        # Zoom out 10%
        img = tf.keras.preprocessing.image.random_zoom(
            image, (1.1, 1.1), row_axis=0, col_axis=1, channel_axis=2
        )
    elif aug_type == 'flip':
        # Flip horizontal
        img = np.fliplr(image)
    elif aug_type == 'shift_h':
        # Shift horizontal 10%
        img = tf.keras.preprocessing.image.random_shift(
            image, 0.1, 0, row_axis=0, col_axis=1, channel_axis=2
        )
    elif aug_type == 'shift_v':
        # Shift vertical 10%
        img = tf.keras.preprocessing.image.random_shift(
            image, 0, 0.1, row_axis=0, col_axis=1, channel_axis=2
        )
    else:
        img = image

    return np.array(img)


def targeted_augmentation(X_train, y_train, target_count=None):
    """
    Applique une augmentation ciblée pour équilibrer les classes.

    Les classes minoritaires reçoivent plus d'augmentations que les classes majoritaires.

    Args:
        X_train: Images d'entraînement
        y_train: Labels d'entraînement
        target_count: Nombre cible d'images par classe (défaut: nombre de la classe majoritaire)

    Returns:
        X_augmented, y_augmented: Dataset augmenté et équilibré
    """
    # Compter les images par classe
    class_counts = Counter(y_train)
    print(f"\nDistribution originale des classes:")
    eth_labels = ['Blanc', 'Noir', 'Asiatique', 'Indien', 'Autre']
    for cls, count in sorted(class_counts.items()):
        print(f"  {eth_labels[cls]}: {count} images")

    # Définir le nombre cible (classe majoritaire par défaut)
    if target_count is None:
        target_count = max(class_counts.values())
    print(f"\nObjectif: {target_count} images par classe")

    # Liste des augmentations disponibles (sans brightness)
    augmentation_types = [
        'rotation_pos', 'rotation_neg',
        'zoom_in', 'zoom_out',
        'flip',
        'shift_h', 'shift_v'
    ]

    X_augmented = list(X_train)
    y_augmented = list(y_train)

    for cls in range(5):
        current_count = class_counts[cls]
        needed = target_count - current_count

        if needed <= 0:
            print(f"  {eth_labels[cls]}: Pas d'augmentation nécessaire")
            continue

        # Récupérer les indices des images de cette classe
        class_indices = np.where(y_train == cls)[0]

        # Calculer combien d'augmentations par image
        aug_per_image = needed // current_count + 1
        print(f"  {eth_labels[cls]}: +{needed} images nécessaires ({aug_per_image} augmentations/image)")

        augmented_count = 0
        aug_idx = 0

        while augmented_count < needed:
            for idx in class_indices:
                if augmented_count >= needed:
                    break

                # Appliquer l'augmentation
                aug_type = augmentation_types[aug_idx % len(augmentation_types)]
                aug_image = apply_augmentation(X_train[idx], aug_type)

                X_augmented.append(aug_image)
                y_augmented.append(cls)
                augmented_count += 1

            aug_idx += 1

    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)

    # Afficher la nouvelle distribution
    new_counts = Counter(y_augmented)
    print(f"\nDistribution après augmentation ciblée:")
    for cls, count in sorted(new_counts.items()):
        print(f"  {eth_labels[cls]}: {count} images")

    print(f"\nDataset total: {len(X_augmented)} images (avant: {len(X_train)})")

    return X_augmented, y_augmented


# Appliquer l'augmentation ciblée
print("=" * 60)
print("AUGMENTATION CIBLÉE")
print("=" * 60)
X_train_aug, y_train_aug = targeted_augmentation(X_train, y_train)

# Mélanger le dataset augmenté
shuffle_idx = np.random.permutation(len(X_train_aug))
X_train_aug = X_train_aug[shuffle_idx]
y_train_aug = y_train_aug[shuffle_idx]

# One-hot encoding (5 classes d'ethnicité)
y_train_cat = to_categorical(y_train_aug, num_classes=5)
y_test_cat = to_categorical(y_test, num_classes=5)

print(f"\nX_train augmenté : {X_train_aug.shape}")
print(f"X_test : {X_test.shape}")

"""## 5. Calcul des poids alpha pour Focal Loss"""

# Calculer les poids inversement proportionnels à la fréquence
# Même après augmentation, on garde des poids pour aider les classes difficiles
class_counts_aug = Counter(y_train_aug)
total_samples = len(y_train_aug)
n_classes = 5

# Poids alpha : classes rares ont un poids plus élevé
alpha_weights = []
for cls in range(n_classes):
    # Formule: n_samples / (n_classes * n_samples_class)
    weight = total_samples / (n_classes * class_counts_aug[cls])
    alpha_weights.append(weight)

# Normaliser pour que la somme = n_classes
alpha_sum = sum(alpha_weights)
alpha_weights = [w * n_classes / alpha_sum for w in alpha_weights]

print(f"\nPoids alpha pour Focal Loss:")
eth_labels = ['Blanc', 'Noir', 'Asiatique', 'Indien', 'Autre']
for i, label in enumerate(eth_labels):
    print(f"  {label}: {alpha_weights[i]:.3f}")

"""## 6. Création du modèle CNN"""

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

# Compiler avec Focal Loss (gamma=3.0 pour focus accru sur les cas difficiles)
# Learning rate réduit à 0.0001 pour une convergence plus stable
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=focal_loss(gamma=3.0, alpha=alpha_weights),
    metrics=['accuracy']
)

model.summary()

"""## 7. Entraînement"""

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,        # Divise le LR par 2
    patience=3,        # Après 3 epochs sans amélioration
    min_lr=1e-6,       # Learning rate minimum
    verbose=1
)

print("\n" + "=" * 60)
print("ENTRAÎNEMENT AVEC FOCAL LOSS + AUGMENTATION CIBLÉE")
print("=" * 60)
print("Configuration :")
print("  - Learning rate initial : 0.0001")
print("  - Batch size : 128")
print("  - ReduceLROnPlateau : factor=0.5, patience=3")
print("  - EarlyStopping : patience=7")

history = model.fit(
    X_train_aug, y_train_cat,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr]
)

"""## 8. Visualisation de l'entraînement"""

# Graphiques de base : Loss et Accuracy
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train', linewidth=2, marker='o', markersize=4)
axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2, marker='s', markersize=4)
axes[0].set_title('Focal Loss durant l\'entraînement', fontsize=12)
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
plt.savefig(os.path.join(OUTPUT_PATH, 'training_curves_focal_aug.png'), dpi=150)
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

"""## 9. Évaluation du modèle"""

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

"""## 10. Matrice de confusion"""

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
plt.title('Matrice de confusion - Focal Loss + Augmentation Ciblée')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix_focal_aug.png'), dpi=150)
plt.show()

"""## 11. Sauvegarde du modèle"""

model.save(os.path.join(OUTPUT_PATH, 'ethnicity_model_focal_aug.keras'))
print(f"Modèle sauvegardé : {OUTPUT_PATH}/ethnicity_model_focal_aug.keras")

print("\n→ Les fichiers sont disponibles dans l'onglet 'Output' de Kaggle pour téléchargement.")

"""## 12. Analyse des performances par classe"""

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
ax.set_title('Performances par classe - Focal Loss + Augmentation Ciblée')
ax.set_xticks(x)
ax.set_xticklabels(eth_labels)
ax.legend()
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'metrics_per_class_focal_aug.png'), dpi=150)
plt.show()

# Résumé final
print("=" * 60)
print("RÉSUMÉ FINAL - FOCAL LOSS + AUGMENTATION CIBLÉE")
print("=" * 60)
print(f"\nAccuracy globale : {accuracy*100:.2f}%")
print(f"\nPerformances par classe :")
for i, label in enumerate(eth_labels):
    print(f"  {label:10s} - Precision: {precision[i]*100:5.1f}% | Recall: {recall[i]*100:5.1f}% | F1: {f1[i]*100:5.1f}% | Support: {support[i]}")

print(f"\n" + "=" * 60)
print("AMÉLIORATIONS APPORTÉES")
print("=" * 60)
print("  1. Focal Loss (gamma=2.0) : Focus sur les exemples difficiles")
print("  2. Augmentation ciblée : Équilibrage des classes par augmentation")
print("     - Rotation ±15°")
print("     - Zoom ±10%")
print("     - Flip horizontal")
print("     - Shift horizontal/vertical ±10%")
print("  3. Poids alpha par classe dans Focal Loss")

print(f"\n" + "=" * 60)
print("FICHIERS SAUVEGARDÉS")
print("=" * 60)
print("  - ethnicity_model_focal_aug.keras")
print("  - training_curves_focal_aug.png")
print("  - confusion_matrix_focal_aug.png")
print("  - metrics_per_class_focal_aug.png")
