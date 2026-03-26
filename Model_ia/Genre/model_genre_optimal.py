# %% [code]
# -*- coding: utf-8 -*-
"""
# Modèle 5 : Architecture Optimale - Classification de Genre

**Entraînement sur Kaggle**

**Approche :** CNN custom from scratch combinant toutes les techniques optimales
- 4 blocs convolutionnels élargis (48 → 96 → 192 → 384 filtres)
- ResNet (Skip Connections) + SE-Net (Channel Attention) + Separable Conv
- Binary Crossentropy + Class Weights
- Dropout progressif (0.25 → 0.25 → 0.3 → 0.3)
- Entraînement en 2 phases (lr=1e-3 puis lr=1e-4)
- Data Augmentation renforcée intégrée au modèle

**Architecture :**
- Bloc 1 : Conv2D(48) → BN → ReLU → SE(4) + Skip → MaxPool → Dropout(0.25)
- Bloc 2 : SeparableConv2D(96) → BN → ReLU → SE(8) + Skip → MaxPool → Dropout(0.25)
- Bloc 3 : SeparableConv2D(192) → BN → ReLU → SE(8) + Skip → MaxPool → Dropout(0.3)
- Bloc 4 : SeparableConv2D(384) → BN → ReLU → SE(16) + Skip → MaxPool → Dropout(0.3)
- GAP → Dense(512) → Dropout(0.5) → Dense(128) → Dropout(0.3) → Dense(1, sigmoid)

**Preprocessing :**
- Images RGB (3 canaux)
- Redimensionnement 128x128
- Normalisation /255

**Dataset :** jangedoo/utkface-new

## 0. Imports et configuration
"""

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score, average_precision_score
)
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Reproductibilité
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Chemins Kaggle
OUTPUT_PATH = "/kaggle/working"

print(f"TensorFlow version : {tf.__version__}")
print(f"GPU disponible : {tf.config.list_physical_devices('GPU')}")

# %% [code]
"""## 1. Chargement des données (RGB)"""


def find_utkface_folder():
    """Recherche automatique du dossier UTKFace dans tous les emplacements possibles."""
    kaggle_input = "/kaggle/input"
    if os.path.exists(kaggle_input):
        for root, dirs, files in os.walk(kaggle_input):
            jpg_files = [f for f in files if f.endswith(".jpg")]
            if len(jpg_files) > 100:
                print(f"Dossier d'images trouvé : {root} ({len(jpg_files)} fichiers)")
                return root

    try:
        import kagglehub
        print("Dataset non trouvé dans /kaggle/input, téléchargement via kagglehub...")
        path = kagglehub.dataset_download("jangedoo/utkface-new")
        print(f"Dataset téléchargé : {path}")
        for root, dirs, files in os.walk(path):
            jpg_files = [f for f in files if f.endswith(".jpg")]
            if len(jpg_files) > 100:
                print(f"Dossier d'images trouvé : {root} ({len(jpg_files)} fichiers)")
                return root
    except Exception as e:
        print(f"Erreur kagglehub : {e}")

    raise FileNotFoundError(
        "Impossible de trouver le dataset UTKFace.\n"
        "Vérifiez que le dataset 'jangedoo/utkface-new' est ajouté comme Input dans le notebook Kaggle."
    )


image_folder = find_utkface_folder()

image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
print(f"Nombre de fichiers .jpg trouvés : {len(image_files)}")

IMG_SIZE = 128
images = []
labels = []

for file in image_files:
    try:
        parts = file.split("_")
        age = int(parts[0])
        gender = int(parts[1])

        img = Image.open(os.path.join(image_folder, file)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        images.append(np.array(img))
        labels.append(gender)
    except:
        continue

images = np.array(images)
labels = np.array(labels)
print(f"Images chargées : {len(images)}")
print(f"Shape des images : {images.shape}")

# %% [code]
"""## 2. Préparation des données (Genre uniquement)"""

X = images
y_gender = labels.astype('float32')

gender_labels = ['Homme', 'Femme']
print(f"\nDistribution des classes :")
print(f"  Homme (0) : {np.sum(y_gender == 0)}")
print(f"  Femme (1) : {np.sum(y_gender == 1)}")

# Split train/test (80/20) stratifié sur le genre
X_train, X_test, y_train, y_test = train_test_split(
    X, y_gender,
    test_size=0.2,
    random_state=SEED,
    stratify=y_gender
)

# Normalisation /255
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Split train/val depuis le train set
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=SEED,
    stratify=y_train
)

print(f"\nX_train : {X_train.shape}")
print(f"X_val   : {X_val.shape}")
print(f"X_test  : {X_test.shape}")

# Calcul de alpha à partir de la distribution des données
n_pos = np.sum(y_train == 1)
n_neg = np.sum(y_train == 0)
alpha_focal = n_neg / (n_pos + n_neg)
print(f"\nAlpha pour Focal Loss (calculé depuis la distribution) : {alpha_focal:.4f}")

# %% [code]
"""## 3. Binary Focal Loss avec Label Smoothing

La Focal Loss modifie la binary cross-entropy :
- FL(p) = -alpha * (1-p)^gamma * log(p)
- gamma > 0 : réduit la contribution des exemples bien classifiés
- alpha : pondère selon le déséquilibre des classes
- Label smoothing : adoucit les labels pour meilleure généralisation
"""


def binary_focal_loss_smooth(gamma=2.0, alpha=0.5, label_smoothing=0.05):
    """
    Binary Focal Loss avec Label Smoothing.

    Args:
        gamma: Facteur de focalisation (2.0 = standard)
        alpha: Poids pour la classe positive
        label_smoothing: Facteur de lissage des labels

    Returns:
        Fonction de loss compatible Keras
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Label smoothing
        y_true_smooth = y_true * (1.0 - label_smoothing) + label_smoothing * 0.5

        # Focal weight
        p_t = y_true_smooth * y_pred + (1 - y_true_smooth) * (1 - y_pred)
        alpha_t = y_true_smooth * alpha + (1 - y_true_smooth) * (1 - alpha)
        focal_weight = K.pow(1.0 - p_t, gamma)

        loss = -alpha_t * focal_weight * K.log(p_t + epsilon)
        return K.mean(loss)

    return focal_loss_fixed


# %% [code]
"""## 4. Data Augmentation renforcée"""

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.06),
    layers.RandomZoom((-0.15, 0.15)),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.15),
], name='data_augmentation')

print("Couches d'augmentation :")
for layer in data_augmentation.layers:
    print(f"  - {layer.name}")

# %% [code]
"""## 5. Construction du modèle - Architecture Optimale

SE-Block (Squeeze-and-Excitation) + ResNet Skip Connections + Separable Conv
+ L2 Regularization + Dropout progressif + Architecture élargie (4 blocs)
"""


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
    se = layers.GlobalAveragePooling2D()(x)

    # Excitation : apprendre les poids
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)

    # Reshape pour multiplication
    se = layers.Reshape((1, 1, filters))(se)

    # Recalibration
    return layers.Multiply()([x, se])


def build_optimal_model(input_shape=(128, 128, 3)):
    """
    Architecture Optimale pour classification binaire de genre.

    Structure élargie avec 4 blocs :
    - Bloc 1 : 48 filtres (Conv2D + SE + Skip)
    - Bloc 2 : 96 filtres (SeparableConv2D + SE + Skip)
    - Bloc 3 : 192 filtres (SeparableConv2D + SE + Skip)
    - Bloc 4 : 384 filtres (SeparableConv2D + SE + Skip)
    - Head : GAP → Dense(512) → Dense(128) → Dense(1, sigmoid)

    Techniques :
    - ResNet : Skip connections
    - SE-Net : Channel attention
    - Separable Conv : Réduction paramètres
    - Dropout progressif : 0.25 → 0.25 → 0.3 → 0.3
    - Data augmentation intégrée
    """
    inputs = layers.Input(shape=input_shape)

    # Data augmentation (active uniquement pendant l'entraînement)
    augmented = data_augmentation(inputs)

    # ================================================================
    # BLOC 1 : 48 filtres
    # ================================================================
    x = layers.Conv2D(48, (3, 3), padding='same')(augmented)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = se_block(x, ratio=4)

    # Skip connection : adapter l'input à 48 channels
    shortcut = layers.Conv2D(48, (1, 1), padding='same')(augmented)
    x = layers.Add()([x, shortcut])

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # ================================================================
    # BLOC 2 : 96 filtres
    # ================================================================
    shortcut = x

    x = layers.SeparableConv2D(96, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = se_block(x, ratio=8)

    # Skip connection : adapter 48 → 96 channels
    shortcut = layers.Conv2D(96, (1, 1), padding='same')(shortcut)
    x = layers.Add()([x, shortcut])

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # ================================================================
    # BLOC 3 : 192 filtres
    # ================================================================
    shortcut = x

    x = layers.SeparableConv2D(192, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = se_block(x, ratio=8)

    # Skip connection : adapter 96 → 192 channels
    shortcut = layers.Conv2D(192, (1, 1), padding='same')(shortcut)
    x = layers.Add()([x, shortcut])

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # ================================================================
    # BLOC 4 : 384 filtres
    # ================================================================
    shortcut = x

    x = layers.SeparableConv2D(384, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = se_block(x, ratio=16)

    # Skip connection : adapter 192 → 384 channels
    shortcut = layers.Conv2D(384, (1, 1), padding='same')(shortcut)
    x = layers.Add()([x, shortcut])

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # ================================================================
    # HEAD : Classification binaire
    # ================================================================
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name='CNN_Genre_Optimal')
    return model


# Créer le modèle
model = build_optimal_model(input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Class weights pour gérer le déséquilibre (plus stable que focal loss pour un CNN from scratch)
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class weights : {class_weight_dict}")

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

print(f"\n{'='*60}")
print("MODÈLE 5 : ARCHITECTURE OPTIMALE - GENRE")
print(f"{'='*60}")
print(f"""
Architecture élargie (4 blocs) :
  - Bloc 1 : Conv2D(48) + SE(4) + Skip + Dropout(0.25)
  - Bloc 2 : SeparableConv2D(96) + SE(8) + Skip + Dropout(0.25)
  - Bloc 3 : SeparableConv2D(192) + SE(8) + Skip + Dropout(0.3)
  - Bloc 4 : SeparableConv2D(384) + SE(16) + Skip + Dropout(0.3)
  - Head : GAP → Dense(512) → Dropout(0.5) → Dense(128) → Dropout(0.3) → Dense(1, sigmoid)
  - Input : RGB {IMG_SIZE}x{IMG_SIZE}
  - Paramètres totaux : {model.count_params():,}

Techniques combinées :
  - ResNet : Skip connections
  - SE-Net : Channel attention
  - Separable Conv : Réduction paramètres
  - Dropout progressif : 0.25 → 0.25 → 0.3 → 0.3
  - Binary Crossentropy + Class Weights
  - Data augmentation renforcée
""")

# %% [code]
"""## 6. Phase 1 : Entraînement principal (lr=1e-3)"""

early_stop1 = EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    patience=7,
    restore_best_weights=True,
    start_from_epoch=15,
    verbose=1
)

reduce_lr1 = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

print("\n" + "=" * 60)
print("PHASE 1 : ENTRAÎNEMENT PRINCIPAL (lr=1e-3)")
print("=" * 60)
print("Configuration :")
print("  - Architecture : 4 blocs (48→96→192→384) + SE + Skip + L2")
print("  - Loss : Binary Crossentropy + Class Weights")
print("  - Optimizer : Adam (lr=0.001)")
print("  - Epochs : 40 (min 20)")

history1 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop1, reduce_lr1]
)

print(f"\nPhase 1 terminée - Meilleure accuracy val : {max(history1.history['val_accuracy'])*100:.2f}%")

# %% [code]
"""## 7. Phase 2 : Fine-tuning (lr=1e-4)"""

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop2 = EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    patience=7,
    restore_best_weights=True,
    start_from_epoch=10,
    verbose=1
)

reduce_lr2 = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

print("\n" + "=" * 60)
print("PHASE 2 : FINE-TUNING (lr=1e-4)")
print("=" * 60)
print("  - Optimizer : Adam (lr=0.0001)")
print("  - Epochs : 30 (min 10)")

history2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop2, reduce_lr2]
)

print(f"\nPhase 2 terminée - Meilleure accuracy val : {max(history2.history['val_accuracy'])*100:.2f}%")

# %% [code]
"""## 8. Évaluation du modèle"""

y_pred_proba = model.predict(X_test).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

accuracy = np.mean(y_pred == y_test)
print(f"GENRE - Accuracy : {accuracy*100:.2f}%")

auc_score = roc_auc_score(y_test, y_pred_proba)
ap_score = average_precision_score(y_test, y_pred_proba)
print(f"AUC : {auc_score:.4f}")
print(f"AP : {ap_score:.4f}")

print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=gender_labels))

# %% [code]
"""## 9. Graphiques d'entraînement (2 phases combinées)"""

all_loss = history1.history['loss'] + history2.history['loss']
all_val_loss = history1.history['val_loss'] + history2.history['val_loss']
all_acc = history1.history['accuracy'] + history2.history['accuracy']
all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
phase1_epochs = len(history1.history['loss'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(all_loss, label='Train', linewidth=2, marker='o', markersize=3)
axes[0].plot(all_val_loss, label='Validation', linewidth=2, marker='s', markersize=3)
axes[0].axvline(x=phase1_epochs - 0.5, color='red', linestyle='--', alpha=0.7, label='Début phase 2')
axes[0].set_title('Loss durant l\'entraînement', fontsize=12)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(all_acc, label='Train', linewidth=2, marker='o', markersize=3)
axes[1].plot(all_val_acc, label='Validation', linewidth=2, marker='s', markersize=3)
axes[1].axvline(x=phase1_epochs - 0.5, color='red', linestyle='--', alpha=0.7, label='Début phase 2')
axes[1].set_title('Accuracy durant l\'entraînement', fontsize=12)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Modèle 5 : Architecture Optimale', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'training_curves_genre_optimal.png'), dpi=150, bbox_inches='tight')
plt.show()

print("=" * 50)
print("RÉSUMÉ DE L'ENTRAÎNEMENT")
print("=" * 50)
print(f"Phase 1 : {len(history1.history['loss'])} epochs")
print(f"Phase 2 : {len(history2.history['loss'])} epochs")
print(f"Total   : {len(all_loss)} epochs")
print(f"\nMeilleure accuracy validation : {max(all_val_acc)*100:.2f}%")
print(f"Meilleure loss validation : {min(all_val_loss):.4f}")

# %% [code]
"""## 10. Matrice de confusion"""

cm_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=gender_labels,
    yticklabels=gender_labels
)
plt.title('Matrice de confusion - Modèle 5 : Architecture Optimale')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix_genre_optimal.png'), dpi=150)
plt.show()

# %% [code]
"""## 11. Performances par classe"""

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))

x_pos = np.arange(len(gender_labels))
width = 0.25

bars1 = ax.bar(x_pos - width, precision * 100, width, label='Precision', color='steelblue')
bars2 = ax.bar(x_pos, recall * 100, width, label='Recall', color='teal')
bars3 = ax.bar(x_pos + width, f1 * 100, width, label='F1-Score', color='coral')

ax.set_ylabel('Score (%)')
ax.set_title('Performances par classe - Modèle 5 : Architecture Optimale')
ax.set_xticks(x_pos)
ax.set_xticklabels(gender_labels)
ax.legend()
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'metrics_per_class_genre_optimal.png'), dpi=150)
plt.show()

# %% [code]
"""## 12. Grad-CAM : Visualiser comment le modèle réfléchit"""


def make_gradcam_heatmap(img_array, model, pred_index=None):
    """
    Génère une heatmap Grad-CAM pour un CNN custom (architecture plate).
    """
    # Trouver la dernière couche convolutionnelle
    last_conv_layer_name = None
    for layer in model.layers[::-1]:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name is None:
        return np.ones((4, 4), dtype=np.float32) * 0.5

    # Sous-modèle : entrée → [dernière conv, sortie finale]
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        tape.watch(conv_outputs)
        # Pour binaire : gradient de la sortie sigmoid
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        return np.ones((4, 4), dtype=np.float32) * 0.5

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def display_gradcam(img, heatmap, alpha=0.4):
    """Superpose la heatmap Grad-CAM sur l'image originale."""
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], (IMG_SIZE, IMG_SIZE)
    ).numpy().squeeze()

    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]

    superimposed = heatmap_colored * alpha + img * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 1)
    return superimposed


# Générer les Grad-CAM sur 8 images du test set
print("Génération des Grad-CAM heatmaps...")
fig, axes = plt.subplots(4, 4, figsize=(16, 16))

np.random.seed(SEED)
indices = np.random.choice(len(X_test), 8, replace=False)

for i, idx in enumerate(indices):
    img = X_test[idx]
    img_array = np.expand_dims(img, axis=0)

    pred_proba = model.predict(img_array, verbose=0).flatten()[0]
    pred_class = int(pred_proba > 0.5)
    true_class = int(y_test[idx])
    confidence = pred_proba if pred_class == 1 else (1 - pred_proba)
    confidence *= 100

    heatmap = make_gradcam_heatmap(img_array, model)
    superimposed = display_gradcam(img, heatmap)

    color = 'green' if pred_class == true_class else 'red'

    row = i // 2
    col = (i % 2) * 2
    axes[row, col].imshow(img)
    axes[row, col].set_title(
        f"Réel: {gender_labels[true_class]}\nPrédit: {gender_labels[pred_class]} ({confidence:.1f}%)",
        fontsize=10, color=color
    )
    axes[row, col].axis('off')

    axes[row, col + 1].imshow(superimposed)
    axes[row, col + 1].set_title('Grad-CAM', fontsize=10)
    axes[row, col + 1].axis('off')

plt.suptitle('Grad-CAM - Modèle 5 : Architecture Optimale\n(Zones chaudes = où le modèle regarde)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'gradcam_genre_optimal.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [code]
"""## 13. Distribution de confiance des prédictions"""

# Pour binaire : confiance = distance à 0.5
y_pred_confidence = np.where(y_pred == 1, y_pred_proba, 1 - y_pred_proba)
correct_mask = (y_pred == y_test)
incorrect_mask = ~correct_mask

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(y_pred_confidence[correct_mask], bins=30, alpha=0.7,
        label=f'Correct ({correct_mask.sum()})', color='green', edgecolor='darkgreen')
ax.hist(y_pred_confidence[incorrect_mask], bins=30, alpha=0.7,
        label=f'Incorrect ({incorrect_mask.sum()})', color='red', edgecolor='darkred')

ax.set_xlabel('Confiance de la prédiction')
ax.set_ylabel('Nombre de prédictions')
ax.set_title('Distribution de confiance - Modèle 5 : Architecture Optimale')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confidence_distribution_genre_optimal.png'), dpi=150)
plt.show()

print(f"Confiance moyenne (prédictions correctes) : {y_pred_confidence[correct_mask].mean()*100:.1f}%")
print(f"Confiance moyenne (prédictions incorrectes) : {y_pred_confidence[incorrect_mask].mean()*100:.1f}%")

# %% [code]
"""## 14. Sauvegarde du modèle"""

model.save(os.path.join(OUTPUT_PATH, 'gender_model_optimal.keras'))
print(f"Modèle sauvegardé : {OUTPUT_PATH}/gender_model_optimal.keras")

# %% [code]
"""## 15. Export TensorFlow Lite

Le modèle contient des couches d'augmentation (RandomFlip, RandomRotation, etc.)
qui ne sont pas supportées par TFLite. On crée un modèle d'inférence propre
sans augmentation, on réutilise les poids, puis on exporte.
"""

print("\n" + "=" * 60)
print("EXPORT TENSORFLOW LITE")
print("=" * 60)

# Les couches d'augmentation (RandomFlip, RandomRotation, etc.) sont
# automatiquement désactivées en mode inférence (training=False).
# On peut donc exporter le modèle directement sans reconstruire le graphe.

# Export TFLite
tflite_path = os.path.join(OUTPUT_PATH, 'gender_optimal.tflite')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
print(f"Modèle TFLite sauvegardé : {tflite_path} ({size_mb:.1f} MB)")

# Vérification du modèle TFLite
print("\nVérification TFLite :")
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"  Input  : shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
print(f"  Output : shape={output_details[0]['shape']}, dtype={output_details[0]['dtype']}")

test_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
test_output = interpreter.get_tensor(output_details[0]['index'])
print(f"  Test prediction : {test_output.flatten()}")
print("  TFLite OK !")

# %% [code]
"""## 16. Résumé final"""

print("=" * 60)
print("RÉSUMÉ FINAL - MODÈLE 5 : ARCHITECTURE OPTIMALE GENRE")
print("=" * 60)
print(f"""
Architecture :
  - Type : CNN custom from scratch (pas de transfer learning)
  - Bloc 1 : Conv2D(48) → BN → ReLU → SE(4) + Skip → MaxPool → Dropout(0.25)
  - Bloc 2 : SeparableConv2D(96) → BN → ReLU → SE(8) + Skip → MaxPool → Dropout(0.25)
  - Bloc 3 : SeparableConv2D(192) → BN → ReLU → SE(8) + Skip → MaxPool → Dropout(0.3)
  - Bloc 4 : SeparableConv2D(384) → BN → ReLU → SE(16) + Skip → MaxPool → Dropout(0.3)
  - Head : GAP → Dense(512) → Dropout(0.5) → Dense(128) → Dropout(0.3) → Dense(1, sigmoid)
  - Input : RGB {IMG_SIZE}x{IMG_SIZE}
  - Paramètres totaux : {model.count_params():,}

Techniques combinées :
  - ResNet : Skip connections autour de chaque bloc
  - SE-Net : Channel attention (ratios 4, 8, 8, 16)
  - Separable Conv : Blocs 2, 3, 4
  - Dropout progressif : 0.25 → 0.25 → 0.3 → 0.3
  - Binary Crossentropy + Class Weights
  - Data augmentation renforcée : Flip, Rotation, Zoom, Translation, Brightness, Contrast
  - Architecture élargie : 4 blocs (48 → 96 → 192 → 384 filtres)

Entraînement en 2 phases :
  - Phase 1 : lr=0.001, {phase1_epochs} epochs
  - Phase 2 : lr=0.0001, {len(history2.history['loss'])} epochs
  - Total : {len(all_loss)} epochs

Résultats :
  - Accuracy globale : {accuracy*100:.2f}%
  - AUC : {auc_score:.4f}
  - AP : {ap_score:.4f}
""")

print("Performances par classe :")
for i, label in enumerate(gender_labels):
    print(f"  {label:10s} - Precision: {precision[i]*100:5.1f}% | Recall: {recall[i]*100:5.1f}% | F1: {f1[i]*100:5.1f}% | Support: {support[i]}")

print(f"\n{'=' * 60}")
print("FICHIERS SAUVEGARDÉS")
print("=" * 60)
print("  - gender_model_optimal.keras")
print("  - gender_optimal.tflite")
print("  - training_curves_genre_optimal.png")
print("  - confusion_matrix_genre_optimal.png")
print("  - metrics_per_class_genre_optimal.png")
print("  - gradcam_genre_optimal.png")
print("  - confidence_distribution_genre_optimal.png")
