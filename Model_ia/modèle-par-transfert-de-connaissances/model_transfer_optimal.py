# %% [code]
# -*- coding: utf-8 -*-
"""
# Modèle 5 : Architecture Optimale - EfficientNetB0

**Entraînement sur Kaggle**

**Changements vs Modèle 4 :** Tout est optimisé pour la performance maximale

1. **EfficientNetB0** au lieu de MobileNetV2
   - Meilleur ratio accuracy/paramètres
   - Intègre déjà des SE blocks (Squeeze-and-Excitation)
   - Compound scaling optimisé

2. **Preprocessing EfficientNet** au lieu de /255
   - Normalisation [-1, 1] via preprocess_input
   - Correspond aux poids ImageNet pré-entraînés

3. **Head renforcé**
   - GAP → BN → Dense(512) → Dropout(0.4) → Dense(128) → Dropout(0.3) → Dense(5)
   - Plus de capacité pour les 5 classes

4. **3 phases de fine-tuning** au lieu de 2
   - Phase 1 : head seul (lr=1e-3)
   - Phase 2 : top 20 couches (lr=1e-4)
   - Phase 3 : top 50 couches (lr=1e-5)

5. **Mixed precision** (float16) pour accélérer le training

6. **Label smoothing** (0.1) pour éviter la surconfiance

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
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Reproductibilité
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Mixed precision pour accélérer sur GPU
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision activée (float16)")
except Exception as e:
    print(f"Mixed precision non disponible : {e}")
    print("Utilisation de float32")

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
        try:
            race = int(parts[2])
        except:
            race = 4

        img = Image.open(os.path.join(image_folder, file)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        images.append(np.array(img))
        labels.append([age, gender, race])
    except:
        continue

images = np.array(images)
labels = np.array(labels)
print(f"Images chargées : {len(images)}")
print(f"Shape des images : {images.shape}")

# %% [code]
"""## 2. Préparation des données

**CHANGEMENT :** Utilisation de preprocess_input d'EfficientNet (normalise en [-1, 1])
au lieu de /255.
"""

from tensorflow.keras.applications.efficientnet import preprocess_input

X = images
y_ethnicity = labels[:, 2]

eth_labels = ['Blanc', 'Noir', 'Asiatique', 'Indien', 'Autre']
print("\nDistribution des classes :")
for i in range(5):
    count = np.sum(y_ethnicity == i)
    print(f"  {eth_labels[i]} : {count}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_ethnicity,
    test_size=0.2,
    random_state=SEED,
    stratify=y_ethnicity
)

# Preprocessing EfficientNet (normalise en [-1, 1])
X_train = preprocess_input(X_train.astype('float32'))
X_test = preprocess_input(X_test.astype('float32'))

print(f"\nX_train : {X_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"X_train min/max : {X_train.min():.2f} / {X_train.max():.2f}")

y_train_cat = to_categorical(y_train, num_classes=5)
y_test_cat = to_categorical(y_test, num_classes=5)

# %% [code]
"""## 3. Focal Loss avec label smoothing"""


def focal_loss(gamma=2.0, alpha=None, label_smoothing=0.1):
    """
    Focal Loss avec label smoothing.

    Args:
        gamma: Facteur de focalisation
        alpha: Poids par classe
        label_smoothing: Lissage des labels (0.1 = 10%)
    """
    def focal_loss_fixed(y_true, y_pred):
        num_classes = tf.shape(y_true)[-1]
        num_classes_f = tf.cast(num_classes, dtype=y_pred.dtype)

        # Label smoothing
        if label_smoothing > 0:
            y_true = y_true * (1.0 - label_smoothing) + label_smoothing / num_classes_f

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        cross_entropy = -y_true * K.log(y_pred)

        p_t = K.sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = K.pow(1.0 - p_t, gamma)

        focal_cross_entropy = focal_weight * cross_entropy

        if alpha is not None:
            alpha_tensor = K.constant(alpha, dtype=K.floatx())
            alpha_weight = K.sum(y_true * alpha_tensor, axis=-1, keepdims=True)
            focal_cross_entropy = alpha_weight * focal_cross_entropy

        loss = K.sum(focal_cross_entropy, axis=-1)
        return K.mean(loss)

    return focal_loss_fixed


# Calcul des poids alpha
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
alpha_weights = list(class_weights)
alpha_sum = sum(alpha_weights)
alpha_weights = [w * 5 / alpha_sum for w in alpha_weights]

print("Poids alpha pour Focal Loss :")
for i, label in enumerate(eth_labels):
    print(f"  {label} : {alpha_weights[i]:.3f}")

# %% [code]
"""## 4. Data Augmentation"""

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom((-0.1, 0.1)),
    layers.RandomBrightness(0.1),
    layers.RandomContrast(0.1),
], name='data_augmentation')

# %% [code]
"""## 5. Construction du modèle - EfficientNetB0

**CHANGEMENTS vs Modèle 4 :**
- EfficientNetB0 au lieu de MobileNetV2
- Head renforcé (512 → 128 → 5)
- BatchNormalization après GAP
"""

# CHANGEMENT : EfficientNetB0 au lieu de MobileNetV2
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)

# CHANGEMENT : Head renforcé
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
# dtype='float32' obligatoire avec mixed precision pour la couche de sortie
outputs = layers.Dense(5, activation='softmax', dtype='float32')(x)

model = models.Model(inputs, outputs, name='Transfer_EfficientNetB0_Optimal')

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=focal_loss(gamma=2.0, alpha=alpha_weights, label_smoothing=0.1),
    metrics=['accuracy']
)

model.summary()

print(f"\nParamètres totaux : {model.count_params():,}")
print(f"Paramètres entraînables : {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")

# %% [code]
"""## 6. Phase 1 : Entraînement du head seul"""

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

print("\n" + "=" * 60)
print("PHASE 1 : HEAD SEUL (EfficientNetB0 frozen)")
print("=" * 60)
print("  - Learning rate : 0.001")
print("  - Loss : Focal Loss (gamma=2.0, label_smoothing=0.1)")

history1 = model.fit(
    X_train, y_train_cat,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr]
)

phase1_acc = max(history1.history['val_accuracy'])
print(f"\nPhase 1 terminée - Accuracy val : {phase1_acc*100:.2f}%")

# %% [code]
"""## 7. Phase 2 : Fine-tuning des couches hautes (top 20)"""

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=focal_loss(gamma=2.0, alpha=alpha_weights, label_smoothing=0.1),
    metrics=['accuracy']
)

trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
print(f"Phase 2 - Paramètres entraînables : {trainable_params:,}")
print(f"Couches débloquées : 20 dernières couches d'EfficientNetB0")

early_stop2 = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
reduce_lr2 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

print("\n" + "=" * 60)
print("PHASE 2 : FINE-TUNING TOP 20 COUCHES")
print("=" * 60)
print("  - Learning rate : 0.0001")

history2 = model.fit(
    X_train, y_train_cat,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop2, reduce_lr2]
)

phase2_acc = max(history2.history['val_accuracy'])
print(f"\nPhase 2 terminée - Accuracy val : {phase2_acc*100:.2f}%")

# %% [code]
"""## 8. Phase 3 : Fine-tuning profond (top 50 couches)

**CHANGEMENT CLÉ vs Modèle 4 :** 3ème phase de fine-tuning avec LR très faible.
Permet d'adapter les features mid-level (textures, formes) aux visages.
"""

# Débloquer les 50 dernières couches
for layer in base_model.layers:
    layer.trainable = False
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=focal_loss(gamma=2.0, alpha=alpha_weights, label_smoothing=0.1),
    metrics=['accuracy']
)

trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
print(f"Phase 3 - Paramètres entraînables : {trainable_params:,}")

early_stop3 = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
reduce_lr3 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)

print("\n" + "=" * 60)
print("PHASE 3 : FINE-TUNING PROFOND (50 dernières couches)")
print("=" * 60)
print("  - Learning rate : 0.00001")

history3 = model.fit(
    X_train, y_train_cat,
    epochs=25,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop3, reduce_lr3]
)

phase3_acc = max(history3.history['val_accuracy'])
print(f"\nPhase 3 terminée - Accuracy val : {phase3_acc*100:.2f}%")

# %% [code]
"""## 9. Évaluation du modèle"""

y_pred_proba = model.predict(X_test)
y_pred = y_pred_proba.argmax(axis=1)

loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"\nAccuracy sur le test set : {accuracy*100:.2f}%")

auc_score = roc_auc_score(y_test_cat, y_pred_proba, multi_class='ovr', average='macro')
ap_score = average_precision_score(y_test_cat, y_pred_proba, average='macro')
print(f"AUC (macro) : {auc_score:.4f}")
print(f"AP (macro) : {ap_score:.4f}")

print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=eth_labels))

# %% [code]
"""## 10. Graphiques d'entraînement (3 phases combinées)"""

all_loss = history1.history['loss'] + history2.history['loss'] + history3.history['loss']
all_val_loss = history1.history['val_loss'] + history2.history['val_loss'] + history3.history['val_loss']
all_acc = history1.history['accuracy'] + history2.history['accuracy'] + history3.history['accuracy']
all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy'] + history3.history['val_accuracy']

phase1_end = len(history1.history['loss'])
phase2_end = phase1_end + len(history2.history['loss'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(all_loss, label='Train', linewidth=2, marker='o', markersize=2)
axes[0].plot(all_val_loss, label='Validation', linewidth=2, marker='s', markersize=2)
axes[0].axvline(x=phase1_end - 0.5, color='red', linestyle='--', alpha=0.7, label='Phase 2 (top 20)')
axes[0].axvline(x=phase2_end - 0.5, color='orange', linestyle='--', alpha=0.7, label='Phase 3 (top 50)')
axes[0].set_title('Focal Loss durant l\'entraînement', fontsize=12)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(all_acc, label='Train', linewidth=2, marker='o', markersize=2)
axes[1].plot(all_val_acc, label='Validation', linewidth=2, marker='s', markersize=2)
axes[1].axvline(x=phase1_end - 0.5, color='red', linestyle='--', alpha=0.7, label='Phase 2 (top 20)')
axes[1].axvline(x=phase2_end - 0.5, color='orange', linestyle='--', alpha=0.7, label='Phase 3 (top 50)')
axes[1].set_title('Accuracy durant l\'entraînement', fontsize=12)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Modèle 5 : EfficientNetB0 Optimal (3 phases)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'training_curves_transfer_optimal.png'), dpi=150, bbox_inches='tight')
plt.show()

# Résumé des phases
print("=" * 50)
print("RÉSUMÉ DES 3 PHASES")
print("=" * 50)
print(f"Phase 1 (head seul) : {len(history1.history['loss'])} epochs - Val acc: {phase1_acc*100:.2f}%")
print(f"Phase 2 (top 20)    : {len(history2.history['loss'])} epochs - Val acc: {phase2_acc*100:.2f}%")
print(f"Phase 3 (top 50)    : {len(history3.history['loss'])} epochs - Val acc: {phase3_acc*100:.2f}%")

# %% [code]
"""## 11. Matrice de confusion"""

cm_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=eth_labels,
    yticklabels=eth_labels
)
plt.title('Matrice de confusion - Modèle 5 : EfficientNetB0 Optimal')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix_transfer_optimal.png'), dpi=150)
plt.show()

# %% [code]
"""## 12. Performances par classe"""

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.arange(len(eth_labels))
width = 0.25

bars1 = ax.bar(x_pos - width, precision * 100, width, label='Precision', color='steelblue')
bars2 = ax.bar(x_pos, recall * 100, width, label='Recall', color='teal')
bars3 = ax.bar(x_pos + width, f1 * 100, width, label='F1-Score', color='coral')

ax.set_ylabel('Score (%)')
ax.set_title('Performances par classe - Modèle 5 : EfficientNetB0 Optimal')
ax.set_xticks(x_pos)
ax.set_xticklabels(eth_labels)
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
plt.savefig(os.path.join(OUTPUT_PATH, 'metrics_per_class_transfer_optimal.png'), dpi=150)
plt.show()

# %% [code]
"""## 13. Grad-CAM : Visualiser comment le modèle réfléchit"""


def make_gradcam_heatmap(img_array, model, base_model, pred_index=None):
    """Génère une heatmap Grad-CAM compatible avec les modèles Keras imbriqués."""
    last_conv_layer_name = None
    for layer in base_model.layers[::-1]:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name is None:
        return np.ones((4, 4), dtype=np.float32) * 0.5

    sub_base = tf.keras.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

    head_layers = []
    found_base = False
    for layer in model.layers:
        if layer.name == base_model.name:
            found_base = True
            continue
        if found_base:
            head_layers.append(layer)

    with tf.GradientTape() as tape:
        conv_outputs, base_features = sub_base(img_array, training=False)
        tape.watch(conv_outputs)

        x = base_features
        for layer in head_layers:
            x = layer(x, training=False)
        predictions = x

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

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
    # Dénormaliser l'image EfficientNet pour l'affichage
    img_display = (img + 1.0) / 2.0  # [-1, 1] → [0, 1]
    img_display = np.clip(img_display, 0, 1)

    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], (IMG_SIZE, IMG_SIZE)
    ).numpy().squeeze()
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    superimposed = heatmap_colored * alpha + img_display * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 1)
    return superimposed, img_display


print("Génération des Grad-CAM heatmaps...")
fig, axes = plt.subplots(4, 4, figsize=(16, 16))

np.random.seed(SEED)
indices = np.random.choice(len(X_test), 8, replace=False)

for i, idx in enumerate(indices):
    img = X_test[idx]
    img_array = np.expand_dims(img, axis=0)

    pred_proba = model.predict(img_array, verbose=0)
    pred_class = np.argmax(pred_proba)
    true_class = y_test[idx]
    confidence = pred_proba[0][pred_class] * 100

    heatmap = make_gradcam_heatmap(img_array, model, base_model, pred_class)
    superimposed, img_display = display_gradcam(img, heatmap)

    color = 'green' if pred_class == true_class else 'red'

    row = i // 2
    col = (i % 2) * 2
    axes[row, col].imshow(img_display)
    axes[row, col].set_title(
        f"Réel: {eth_labels[true_class]}\nPrédit: {eth_labels[pred_class]} ({confidence:.1f}%)",
        fontsize=10, color=color
    )
    axes[row, col].axis('off')

    axes[row, col + 1].imshow(superimposed)
    axes[row, col + 1].set_title('Grad-CAM', fontsize=10)
    axes[row, col + 1].axis('off')

plt.suptitle('Grad-CAM - Modèle 5 : EfficientNetB0 Optimal\n(Zones chaudes = où le modèle regarde)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'gradcam_transfer_optimal.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [code]
"""## 14. Distribution de confiance des prédictions"""

y_pred_max_proba = np.max(y_pred_proba, axis=1)
correct_mask = (y_pred == y_test)
incorrect_mask = ~correct_mask

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(y_pred_max_proba[correct_mask], bins=30, alpha=0.7,
        label=f'Correct ({correct_mask.sum()})', color='green', edgecolor='darkgreen')
ax.hist(y_pred_max_proba[incorrect_mask], bins=30, alpha=0.7,
        label=f'Incorrect ({incorrect_mask.sum()})', color='red', edgecolor='darkred')

ax.set_xlabel('Probabilité de la classe prédite')
ax.set_ylabel('Nombre de prédictions')
ax.set_title('Distribution de confiance - Modèle 5 : EfficientNetB0 Optimal')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confidence_distribution_transfer_optimal.png'), dpi=150)
plt.show()

print(f"Confiance moyenne (correctes) : {y_pred_max_proba[correct_mask].mean()*100:.1f}%")
print(f"Confiance moyenne (incorrectes) : {y_pred_max_proba[incorrect_mask].mean()*100:.1f}%")

# %% [code]
"""## 15. Sauvegarde du modèle"""

model.save(os.path.join(OUTPUT_PATH, 'ethnicity_model_transfer_optimal.keras'))
print(f"Modèle sauvegardé : {OUTPUT_PATH}/ethnicity_model_transfer_optimal.keras")

# %% [code]
"""## 16. Résumé final"""

print("=" * 60)
print("RÉSUMÉ FINAL - MODÈLE 5 : EFFICIENTNETB0 OPTIMAL")
print("=" * 60)
print(f"""
Architecture :
  - Base model : EfficientNetB0 (ImageNet)
  - Head : GAP → BN → Dense(512) → Dropout(0.4) → Dense(128) → Dropout(0.3) → Dense(5)
  - Input : RGB 128x128 (preprocessing EfficientNet [-1, 1])
  - Mixed precision : float16
  - Paramètres totaux : {model.count_params():,}

CHANGEMENTS vs Modèle 4 :
  → EfficientNetB0 au lieu de MobileNetV2
  → Preprocessing EfficientNet ([-1, 1]) au lieu de /255
  → Head renforcé (512 → 128 → 5) avec BatchNorm
  → 3 phases de fine-tuning (au lieu de 2)
  → Label smoothing (0.1)
  → Mixed precision (float16)

Entraînement :
  - Phase 1 : Head seul (lr=1e-3) - {len(history1.history['loss'])} epochs → Val acc: {phase1_acc*100:.2f}%
  - Phase 2 : Top 20 couches (lr=1e-4) - {len(history2.history['loss'])} epochs → Val acc: {phase2_acc*100:.2f}%
  - Phase 3 : Top 50 couches (lr=1e-5) - {len(history3.history['loss'])} epochs → Val acc: {phase3_acc*100:.2f}%
  - Loss : Focal Loss (gamma=2.0, label_smoothing=0.1)
  - Augmentation : RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast

Résultats :
  - Accuracy globale : {accuracy*100:.2f}%
  - AUC (macro) : {auc_score:.4f}
  - AP (macro) : {ap_score:.4f}
""")

print("Performances par classe :")
for i, label in enumerate(eth_labels):
    print(f"  {label:10s} - Precision: {precision[i]*100:5.1f}% | Recall: {recall[i]*100:5.1f}% | F1: {f1[i]*100:5.1f}% | Support: {support[i]}")

print(f"\n{'=' * 60}")
print("COMPARAISON AVEC LE MEILLEUR MODÈLE CNN CUSTOM")
print("=" * 60)
print(f"  Meilleur CNN custom (Focal+Augmentation) : 78.38%")
print(f"  Modèle 5 (EfficientNetB0 Optimal)       : {accuracy*100:.2f}%")
print(f"  Amélioration                             : {(accuracy*100 - 78.38):+.2f} points")

print(f"\n{'=' * 60}")
print("FICHIERS SAUVEGARDÉS")
print("=" * 60)
print("  - ethnicity_model_transfer_optimal.keras")
print("  - training_curves_transfer_optimal.png")
print("  - confusion_matrix_transfer_optimal.png")
print("  - metrics_per_class_transfer_optimal.png")
print("  - gradcam_transfer_optimal.png")
print("  - confidence_distribution_transfer_optimal.png")
