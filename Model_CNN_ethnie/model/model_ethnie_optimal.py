# %% [code]
# -*- coding: utf-8 -*-
"""
# Modèle d'ethnicité UTKFace - Architecture Optimale

**Entraînement sur Kaggle**

**Combinaison des meilleures techniques identifiées :**

1. **Focal Loss** (du meilleur modèle 78.38%)
   - Gère le déséquilibre des classes
   - γ=2.0 pour pénaliser les erreurs faciles

2. **ResNet (Skip Connections)**
   - Meilleur flux de gradient
   - Évite la dégradation du réseau

3. **SE-Net (Squeeze-and-Excitation)**
   - Attention sur les channels
   - Apprend l'importance des filtres

4. **Separable Convolutions**
   - Réduit les paramètres
   - Meilleure généralisation

5. **Data Augmentation légère**
   - Rotation ±10°
   - Zoom ±5%

6. **Régularisation renforcée**
   - Dropout progressif : 0.3 → 0.4 → 0.5
   - L2 regularization (1e-4) sur Conv2D, SeparableConv2D et Dense

**Architecture intermédiaire (évite overfitting) :**
- Bloc 1 : 48 filtres
- Bloc 2 : 96 filtres
- Bloc 3 : 192 filtres
- Dense : 256 → 5

**Preprocessing :**
- Images en niveaux de gris (1 canal)
- Redimensionnement 128x128
- Normalisation /255
- Class weights pour équilibrage

**Dataset :** jangedoo/utkface-new

## 1. Chargement des données (Kaggle)
"""

import os
import numpy as np
from PIL import Image

# Chemins Kaggle
KAGGLE_INPUT_PATH = "/kaggle/input/utkface-new"
OUTPUT_PATH = "/kaggle/working"

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
    raise FileNotFoundError("Impossible de trouver le dossier contenant les images UTKFace.")

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
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, MaxPooling2D, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D, Flatten, Add, Multiply,
    Reshape, Activation, RandomRotation, RandomZoom
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

print(f"TensorFlow version : {tf.__version__}")
print(f"GPU disponible : {tf.config.list_physical_devices('GPU')}")

X = images
y_ethnicity = labels[:, 2]

X = X.reshape(X.shape[0], 128, 128, 1)
print(f"Shape après reshape : {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_ethnicity,
    test_size=0.2,
    random_state=42,
    stratify=y_ethnicity
)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(f"X_train min/max : {X_train.min():.2f} / {X_train.max():.2f}")

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights : {class_weight_dict}")

y_train_cat = to_categorical(y_train, num_classes=5)
y_test_cat = to_categorical(y_test, num_classes=5)

print(f"X_train : {X_train.shape}")
print(f"X_test : {X_test.shape}")

"""## 3. Définition de la Focal Loss"""


def focal_loss(gamma=2.0):
    """
    Focal Loss pour gérer le déséquilibre des classes.

    La Focal Loss réduit le poids des exemples bien classifiés
    et se concentre sur les exemples difficiles.

    Args:
        gamma: Facteur de focalisation (2.0 recommandé)

    Returns:
        Fonction de loss
    """
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = y_true * tf.pow(1 - y_pred, gamma)
        focal = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))
    return loss_fn


"""## 4. Définition du SE-Block"""


def se_block(x, ratio=8):
    """
    Squeeze-and-Excitation Block

    Apprend l'importance de chaque channel/filtre.

    Args:
        x: Input tensor
        ratio: Ratio de réduction pour le bottleneck

    Returns:
        Tensor recalibré
    """
    filters = x.shape[-1]

    # Squeeze : moyenne globale par channel
    se = GlobalAveragePooling2D()(x)

    # Excitation : apprendre les poids
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)

    # Reshape pour multiplication
    se = Reshape((1, 1, filters))(se)

    # Recalibration
    return Multiply()([x, se])


"""## 5. Construction du modèle optimal"""


def build_optimal_model(input_shape=(128, 128, 1), num_classes=5):
    """
    Architecture Optimale combinant les meilleures techniques.

    Structure intermédiaire (entre base et complexe) :
    - Bloc 1 : 48 filtres
    - Bloc 2 : 96 filtres
    - Bloc 3 : 192 filtres
    - Dense : 256 → 5

    Techniques :
    - ResNet : Skip connections
    - SE-Net : Channel attention
    - Separable Conv : Réduction paramètres
    - Dropout progressif : 0.3 → 0.4 → 0.5
    - L2 regularization
    """
    inputs = Input(shape=input_shape)

    # ================================================================
    # DATA AUGMENTATION (légère, intégrée au modèle)
    # Appliquée UNE SEULE FOIS sur l'input
    # ================================================================
    augmented = RandomRotation(0.028)(inputs)  # ±10°
    augmented = RandomZoom(0.05)(augmented)    # ±5%

    # ================================================================
    # BLOC 1 : 48 filtres
    # ================================================================
    # Première conv en Conv2D classique (SeparableConv pas efficace sur 1 channel)
    x = Conv2D(48, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(augmented)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = se_block(x, ratio=8)

    # Skip connection : utilise le MÊME input augmenté
    shortcut = Conv2D(48, (1, 1), padding='same', kernel_regularizer=l2(1e-4))(augmented)
    x = Add()([x, shortcut])

    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)  # Dropout progressif - niveau 1

    # ================================================================
    # BLOC 2 : 96 filtres
    # ================================================================
    shortcut = x

    x = SeparableConv2D(96, (3, 3), padding='same', depthwise_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = se_block(x, ratio=8)

    # Skip connection : adapter 48 → 96 channels
    shortcut = Conv2D(96, (1, 1), padding='same', kernel_regularizer=l2(1e-4))(shortcut)
    x = Add()([x, shortcut])

    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4)(x)  # Dropout progressif - niveau 2

    # ================================================================
    # BLOC 3 : 192 filtres
    # ================================================================
    shortcut = x

    x = SeparableConv2D(192, (3, 3), padding='same', depthwise_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = se_block(x, ratio=8)

    # Skip connection : adapter 96 → 192 channels
    shortcut = Conv2D(192, (1, 1), padding='same', kernel_regularizer=l2(1e-4))(shortcut)
    x = Add()([x, shortcut])

    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)  # Dropout progressif - niveau 3

    # ================================================================
    # CLASSIFICATION
    # ================================================================
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='Optimal_ResNet_SE_Focal')
    return model


# Créer le modèle
model = build_optimal_model(input_shape=(128, 128, 1), num_classes=5)

# Compilation avec Focal Loss
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=focal_loss(gamma=2.0),
    metrics=['accuracy']
)

model.summary()

# Informations sur le modèle
print(f"\n{'='*60}")
print("MODÈLE OPTIMAL - COMBINAISON DES MEILLEURES TECHNIQUES")
print(f"{'='*60}")
print(f"""
Architecture intermédiaire :
  - Bloc 1 : 48 filtres
  - Bloc 2 : 96 filtres
  - Bloc 3 : 192 filtres
  - Dense  : 256 → 5

Techniques combinées :
  - Focal Loss (γ=2.0) : Gestion déséquilibre classes
  - ResNet : Skip connections
  - SE-Net : Channel attention
  - Separable Conv : Réduction paramètres
  - Data Augmentation : Rotation ±10°, Zoom ±5%
  - Dropout progressif : 0.3 → 0.4 → 0.5
  - L2 regularization : 1e-4

Paramètres : {model.count_params():,}
""")

"""## 6. Entraînement"""

# Split stratifié
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train_cat,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print(f"X_train_final : {X_train_final.shape}")
print(f"X_val : {X_val.shape}")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    X_train_final, y_train_final,
    epochs=50,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight_dict
)

"""## 7. Visualisation de l'entraînement"""

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
plt.savefig(os.path.join(OUTPUT_PATH, 'training_curves_optimal.png'), dpi=150)
plt.show()

print("=" * 50)
print("RÉSUMÉ DE L'ENTRAÎNEMENT")
print("=" * 50)
print(f"Nombre d'epochs effectuées : {len(history.history['loss'])}")
print(f"\nMeilleure accuracy validation : {max(history.history['val_accuracy'])*100:.2f}%")
print(f"Meilleure loss validation : {min(history.history['val_loss']):.4f}")
print(f"\nAccuracy finale (train) : {history.history['accuracy'][-1]*100:.2f}%")
print(f"Accuracy finale (val) : {history.history['val_accuracy'][-1]*100:.2f}%")

"""## 8. Évaluation du modèle"""

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

"""## 9. Matrice de confusion"""

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
plt.title('Matrice de confusion - Modèle Optimal')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix_optimal.png'), dpi=150)
plt.show()

"""## 10. Sauvegarde du modèle"""

model.save(os.path.join(OUTPUT_PATH, 'ethnicity_model_optimal.keras'))
print(f"Modèle sauvegardé : {OUTPUT_PATH}/ethnicity_model_optimal.keras")

print("\n→ Les fichiers sont disponibles dans l'onglet 'Output' de Kaggle pour téléchargement.")

"""## 11. Analyse des performances par classe"""

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.arange(len(eth_labels))
width = 0.25

bars1 = ax.bar(x_pos - width, precision * 100, width, label='Precision', color='steelblue')
bars2 = ax.bar(x_pos, recall * 100, width, label='Recall', color='teal')
bars3 = ax.bar(x_pos + width, f1 * 100, width, label='F1-Score', color='coral')

ax.set_ylabel('Score (%)')
ax.set_title('Performances par classe - Modèle Optimal')
ax.set_xticks(x_pos)
ax.set_xticklabels(eth_labels)
ax.legend()
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'metrics_per_class_optimal.png'), dpi=150)
plt.show()

# Résumé final
print("=" * 60)
print("RÉSUMÉ FINAL - MODÈLE OPTIMAL")
print("=" * 60)
print(f"\nArchitecture :")
print(f"  - 3 blocs convolutionnels")
print(f"  - Filtres : 48 → 96 → 192")
print(f"  - Dense : 256 → 5")
print(f"  - Paramètres : {model.count_params():,}")
print(f"\nTechniques combinées :")
print(f"  - Focal Loss (γ=2.0)")
print(f"  - ResNet : Skip connections")
print(f"  - SE-Net : Channel attention")
print(f"  - Separable Conv")
print(f"  - Data Augmentation : Rotation ±10°, Zoom ±5%")
print(f"  - Dropout progressif : 0.3 → 0.4 → 0.5")
print(f"  - L2 regularization")
print(f"\nAccuracy globale : {accuracy*100:.2f}%")
print(f"AUC (macro) : {auc_score:.4f}")
print(f"AP (macro) : {ap_score:.4f}")
print(f"\nPerformances par classe :")
for i, label in enumerate(eth_labels):
    print(f"  {label:10s} - Precision: {precision[i]*100:5.1f}% | Recall: {recall[i]*100:5.1f}% | F1: {f1[i]*100:5.1f}% | Support: {support[i]}")

print(f"\n" + "=" * 60)
print("FICHIERS SAUVEGARDÉS")
print("=" * 60)
print("  - ethnicity_model_optimal.keras")
print("  - training_curves_optimal.png")
print("  - confusion_matrix_optimal.png")
print("  - metrics_per_class_optimal.png")
