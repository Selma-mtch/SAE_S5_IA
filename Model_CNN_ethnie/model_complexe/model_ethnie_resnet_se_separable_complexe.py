# %% [code]
# -*- coding: utf-8 -*-
"""
# Modèle d'ethnicité UTKFace - Architecture Complexe + ResNet + SE-Net + Separable Conv

**Entraînement sur Kaggle**

**Architecture complexe (VGG-style) améliorée avec :**

1. **ResNet (Skip Connections)** :
   - Connexions résiduelles autour de chaque bloc conv
   - Meilleur flux de gradient

2. **SE-Net (Squeeze-and-Excitation)** :
   - Attention sur les channels après chaque convolution
   - Le modèle apprend quels filtres sont importants

3. **Separable Convolutions** :
   - Remplace Conv2D par SeparableConv2D
   - Réduit le nombre de paramètres

**Structure complexe (VGG-style) :**
- Bloc 1 : 64 filtres (double conv)
- Bloc 2 : 128 filtres (double conv)
- Bloc 3 : 256 filtres (double conv)
- Bloc 4 : 512 filtres (double conv)
- Dense : 512 → 256 → 5

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
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, MaxPooling2D, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D, Flatten, Add, Multiply,
    Reshape, Activation
)
from tensorflow.keras.models import Model
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
    random_state=42
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

"""## 3. Définition des blocs (SE-Block)"""


def se_block(x, ratio=8):
    """
    Squeeze-and-Excitation Block

    Apprend l'importance de chaque channel/filtre.
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


def conv_block_with_se(x, filters, use_separable=True):
    """
    Bloc de double convolution style VGG avec SE-Net

    Args:
        x: Input tensor
        filters: Nombre de filtres
        use_separable: Utiliser SeparableConv2D (True) ou Conv2D (False)

    Returns:
        Output tensor après double conv + SE
    """
    ConvLayer = SeparableConv2D if use_separable else Conv2D

    # Première convolution
    x = ConvLayer(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Deuxième convolution
    x = ConvLayer(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # SE-Block : attention sur les channels
    x = se_block(x)

    return x


"""## 4. Construction du modèle (Architecture Complexe + 3 techniques)"""


def build_model(input_shape=(128, 128, 1), num_classes=5):
    """
    Architecture Complexe (VGG-style) avec ResNet + SE-Net + Separable Conv

    Même structure que model_ethnie_base_complexe.py :
    - Bloc 1 : 64 filtres (double conv)
    - Bloc 2 : 128 filtres (double conv)
    - Bloc 3 : 256 filtres (double conv)
    - Bloc 4 : 512 filtres (double conv)
    - Dense : 512 → 256 → 5

    Modifications :
    - Conv2D → SeparableConv2D (sauf premier bloc)
    - Ajout SE-Block après chaque double conv
    - Ajout Skip Connection autour de chaque bloc
    """
    inputs = Input(shape=input_shape)

    # ================================================================
    # BLOC 1 : 64 filtres (double conv comme VGG)
    # ================================================================
    # Premier bloc en Conv2D classique (SeparableConv pas efficace sur 1 channel)
    x = conv_block_with_se(inputs, 64, use_separable=False)

    # Skip connection : adapter l'input à 64 channels
    shortcut = Conv2D(64, (1, 1), padding='same')(inputs)
    x = Add()([x, shortcut])  # +ResNet : Skip connection

    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # ================================================================
    # BLOC 2 : 128 filtres (double conv)
    # ================================================================
    shortcut = x  # Sauvegarder pour skip connection

    x = conv_block_with_se(x, 128, use_separable=True)  # +Separable

    # Skip connection : adapter 64 → 128 channels
    shortcut = Conv2D(128, (1, 1), padding='same')(shortcut)
    x = Add()([x, shortcut])  # +ResNet

    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # ================================================================
    # BLOC 3 : 256 filtres (double conv)
    # ================================================================
    shortcut = x

    x = conv_block_with_se(x, 256, use_separable=True)  # +Separable

    # Skip connection : adapter 128 → 256 channels
    shortcut = Conv2D(256, (1, 1), padding='same')(shortcut)
    x = Add()([x, shortcut])  # +ResNet

    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # ================================================================
    # BLOC 4 : 512 filtres (double conv) - NOUVEAU vs modèle de base
    # ================================================================
    shortcut = x

    x = conv_block_with_se(x, 512, use_separable=True)  # +Separable

    # Skip connection : adapter 256 → 512 channels
    shortcut = Conv2D(512, (1, 1), padding='same')(shortcut)
    x = Add()([x, shortcut])  # +ResNet

    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # ================================================================
    # CLASSIFICATION (comme le modèle complexe : 512 → 256 → 5)
    # ================================================================
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='Complexe_ResNet_SE_Separable')
    return model


# Créer le modèle
model = build_model(input_shape=(128, 128, 1), num_classes=5)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Comparaison avec le modèle complexe de base
print(f"\n{'='*60}")
print("COMPARAISON AVEC LE MODÈLE COMPLEXE DE BASE")
print(f"{'='*60}")
print(f"""
Structure identique au modèle complexe (VGG-style) :
  - Bloc 1 : 64 filtres (double conv)
  - Bloc 2 : 128 filtres (double conv)
  - Bloc 3 : 256 filtres (double conv)
  - Bloc 4 : 512 filtres (double conv)
  - Dense  : 512 → 256 → 5

Modifications apportées :
  +ResNet    : Skip connections autour de chaque bloc
  +SE-Net    : Attention sur les channels (après chaque double conv)
  +Separable : SeparableConv2D au lieu de Conv2D (blocs 2, 3 et 4)

Paramètres : {model.count_params():,}
""")

"""## 5. Entraînement"""

# Split stratifié pour garder les mêmes proportions de classes
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
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
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

"""## 6. Visualisation de l'entraînement"""

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
plt.savefig(os.path.join(OUTPUT_PATH, 'training_curves_complexe_resnet_se_separable.png'), dpi=150)
plt.show()

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
plt.title('Matrice de confusion - Complexe + ResNet + SE + Separable')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix_complexe_resnet_se_separable.png'), dpi=150)
plt.show()

"""## 9. Sauvegarde du modèle"""

model.save(os.path.join(OUTPUT_PATH, 'ethnicity_model_complexe_resnet_se_separable.keras'))
print(f"Modèle sauvegardé : {OUTPUT_PATH}/ethnicity_model_complexe_resnet_se_separable.keras")

print("\n→ Les fichiers sont disponibles dans l'onglet 'Output' de Kaggle pour téléchargement.")

"""## 10. Analyse des performances par classe"""

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(eth_labels))
width = 0.25

bars1 = ax.bar(x - width, precision * 100, width, label='Precision', color='steelblue')
bars2 = ax.bar(x, recall * 100, width, label='Recall', color='teal')
bars3 = ax.bar(x + width, f1 * 100, width, label='F1-Score', color='coral')

ax.set_ylabel('Score (%)')
ax.set_title('Performances par classe - Complexe + ResNet + SE + Separable')
ax.set_xticks(x)
ax.set_xticklabels(eth_labels)
ax.legend()
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'metrics_per_class_complexe_resnet_se_separable.png'), dpi=150)
plt.show()

# Résumé final
print("=" * 60)
print("RÉSUMÉ FINAL - COMPLEXE + ResNet + SE-Net + Separable Conv")
print("=" * 60)
print(f"\nArchitecture :")
print(f"  - 4 blocs convolutionnels (double conv style VGG)")
print(f"  - Filtres : 64 → 128 → 256 → 512")
print(f"  - Dense : 512 → 256 → 5")
print(f"  - Paramètres : {model.count_params():,}")
print(f"\nTechniques ajoutées :")
print(f"  - ResNet : Skip connections")
print(f"  - SE-Net : Channel attention")
print(f"  - Separable Conv : Réduction paramètres")
print(f"\nAccuracy globale : {accuracy*100:.2f}%")
print(f"\nPerformances par classe :")
for i, label in enumerate(eth_labels):
    print(f"  {label:10s} - Precision: {precision[i]*100:5.1f}% | Recall: {recall[i]*100:5.1f}% | F1: {f1[i]*100:5.1f}% | Support: {support[i]}")

print(f"\n" + "=" * 60)
print("FICHIERS SAUVEGARDÉS")
print("=" * 60)
print("  - ethnicity_model_complexe_resnet_se_separable.keras")
print("  - training_curves_complexe_resnet_se_separable.png")
print("  - confusion_matrix_complexe_resnet_se_separable.png")
print("  - metrics_per_class_complexe_resnet_se_separable.png")
