# %% [code]
# -*- coding: utf-8 -*-
"""
# Modèle 4 : Focal Loss - MobileNetV2

**Entraînement sur Kaggle**

**Changement unique vs Modèle 3 :** Remplacer cross-entropy par Focal Loss
- Focal Loss (gamma=2.0) : focus sur les exemples difficiles
- Poids alpha par classe : inversement proportionnels à la fréquence
- Remplace class_weight par les poids alpha intégrés dans la loss

**Pourquoi :** La classe "Autre" (325 images) a seulement 31% de F1 dans les modèles
précédents. La Focal Loss réduit la contribution des exemples faciles (Blanc, Noir)
et force le modèle à apprendre les cas difficiles (Autre, Indien).

**Architecture :** Identique au Modèle 3
- MobileNetV2 + fine-tuning 2 phases + data augmentation

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
"""## 2. Préparation des données"""

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

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(f"\nX_train : {X_train.shape}")
print(f"X_test : {X_test.shape}")

y_train_cat = to_categorical(y_train, num_classes=5)
y_test_cat = to_categorical(y_test, num_classes=5)

# %% [code]
"""## 3. Focal Loss (CHANGEMENT CLÉ vs Modèle 3)

La Focal Loss modifie la cross-entropy standard :
- FL(p) = -alpha * (1-p)^gamma * log(p)
- gamma > 0 : réduit la contribution des exemples bien classifiés
- alpha : poids par classe (classes rares → poids plus élevé)
"""


def focal_loss(gamma=2.0, alpha=None):
    """
    Focal Loss pour gérer le déséquilibre de classes.

    Args:
        gamma: Facteur de focalisation (2.0 = standard)
        alpha: Poids par classe (liste de 5 valeurs)

    Returns:
        Fonction de loss compatible Keras
    """
    def focal_loss_fixed(y_true, y_pred):
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


# Calcul des poids alpha à partir des class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

# Normaliser pour que la somme = nombre de classes
alpha_weights = list(class_weights)
alpha_sum = sum(alpha_weights)
alpha_weights = [w * 5 / alpha_sum for w in alpha_weights]

print("Poids alpha pour Focal Loss :")
for i, label in enumerate(eth_labels):
    print(f"  {label} : {alpha_weights[i]:.3f}")

# %% [code]
"""## 4. Data Augmentation (identique au Modèle 3)"""

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom((-0.1, 0.1)),
    layers.RandomBrightness(0.1),
    layers.RandomContrast(0.1),
], name='data_augmentation')

# %% [code]
"""## 5. Construction du modèle"""

base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(5, activation='softmax')(x)

model = models.Model(inputs, outputs, name='Transfer_MobileNetV2_Focal')

# CHANGEMENT : Focal Loss au lieu de categorical_crossentropy
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=focal_loss(gamma=2.0, alpha=alpha_weights),
    metrics=['accuracy']
)

model.summary()

# %% [code]
"""## 6. Phase 1 : Entraînement du head seul"""

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

print("\n" + "=" * 60)
print("PHASE 1 : HEAD SEUL + AUGMENTATION + FOCAL LOSS")
print("=" * 60)
print("  - Loss : Focal Loss (gamma=2.0, alpha=class weights)")
print("  - Pas de class_weight (intégré dans la loss)")

history1 = model.fit(
    X_train, y_train_cat,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr]
)

print(f"\nPhase 1 terminée - Accuracy val : {max(history1.history['val_accuracy'])*100:.2f}%")

# %% [code]
"""## 7. Phase 2 : Fine-tuning"""

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompiler avec Focal Loss et LR réduit
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=focal_loss(gamma=2.0, alpha=alpha_weights),
    metrics=['accuracy']
)

print(f"Phase 2 - Paramètres entraînables : {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")

early_stop2 = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
reduce_lr2 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

print("\n" + "=" * 60)
print("PHASE 2 : FINE-TUNING + AUGMENTATION + FOCAL LOSS")
print("=" * 60)

history2 = model.fit(
    X_train, y_train_cat,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop2, reduce_lr2]
)

print(f"\nPhase 2 terminée - Accuracy val : {max(history2.history['val_accuracy'])*100:.2f}%")

# %% [code]
"""## 8. Évaluation du modèle"""

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
"""## 9. Graphiques d'entraînement (2 phases combinées)"""

all_loss = history1.history['loss'] + history2.history['loss']
all_val_loss = history1.history['val_loss'] + history2.history['val_loss']
all_acc = history1.history['accuracy'] + history2.history['accuracy']
all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
phase1_epochs = len(history1.history['loss'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(all_loss, label='Train', linewidth=2, marker='o', markersize=3)
axes[0].plot(all_val_loss, label='Validation', linewidth=2, marker='s', markersize=3)
axes[0].axvline(x=phase1_epochs - 0.5, color='red', linestyle='--', alpha=0.7, label='Début fine-tuning')
axes[0].set_title('Focal Loss durant l\'entraînement', fontsize=12)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(all_acc, label='Train', linewidth=2, marker='o', markersize=3)
axes[1].plot(all_val_acc, label='Validation', linewidth=2, marker='s', markersize=3)
axes[1].axvline(x=phase1_epochs - 0.5, color='red', linestyle='--', alpha=0.7, label='Début fine-tuning')
axes[1].set_title('Accuracy durant l\'entraînement', fontsize=12)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Modèle 4 : Focal Loss (gamma=2.0)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'training_curves_transfer_focal.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [code]
"""## 10. Matrice de confusion"""

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
plt.title('Matrice de confusion - Modèle 4 : Focal Loss')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix_transfer_focal.png'), dpi=150)
plt.show()

# %% [code]
"""## 11. Performances par classe"""

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.arange(len(eth_labels))
width = 0.25

bars1 = ax.bar(x_pos - width, precision * 100, width, label='Precision', color='steelblue')
bars2 = ax.bar(x_pos, recall * 100, width, label='Recall', color='teal')
bars3 = ax.bar(x_pos + width, f1 * 100, width, label='F1-Score', color='coral')

ax.set_ylabel('Score (%)')
ax.set_title('Performances par classe - Modèle 4 : Focal Loss')
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
plt.savefig(os.path.join(OUTPUT_PATH, 'metrics_per_class_transfer_focal.png'), dpi=150)
plt.show()

# %% [code]
"""## 12. Grad-CAM : Visualiser comment le modèle réfléchit"""


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
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], (IMG_SIZE, IMG_SIZE)
    ).numpy().squeeze()
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    superimposed = heatmap_colored * alpha + img * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 1)
    return superimposed


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
    superimposed = display_gradcam(img, heatmap)

    color = 'green' if pred_class == true_class else 'red'

    row = i // 2
    col = (i % 2) * 2
    axes[row, col].imshow(img)
    axes[row, col].set_title(
        f"Réel: {eth_labels[true_class]}\nPrédit: {eth_labels[pred_class]} ({confidence:.1f}%)",
        fontsize=10, color=color
    )
    axes[row, col].axis('off')

    axes[row, col + 1].imshow(superimposed)
    axes[row, col + 1].set_title('Grad-CAM', fontsize=10)
    axes[row, col + 1].axis('off')

plt.suptitle('Grad-CAM - Modèle 4 : Focal Loss\n(Zones chaudes = où le modèle regarde)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'gradcam_transfer_focal.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [code]
"""## 13. Distribution de confiance des prédictions"""

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
ax.set_title('Distribution de confiance - Modèle 4 : Focal Loss')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confidence_distribution_transfer_focal.png'), dpi=150)
plt.show()

print(f"Confiance moyenne (correctes) : {y_pred_max_proba[correct_mask].mean()*100:.1f}%")
print(f"Confiance moyenne (incorrectes) : {y_pred_max_proba[incorrect_mask].mean()*100:.1f}%")

# %% [code]
"""## 14. Sauvegarde du modèle"""

model.save(os.path.join(OUTPUT_PATH, 'ethnicity_model_transfer_focal.keras'))
print(f"Modèle sauvegardé : {OUTPUT_PATH}/ethnicity_model_transfer_focal.keras")

# %% [code]
"""## 15. Résumé final"""

print("=" * 60)
print("RÉSUMÉ FINAL - MODÈLE 4 : FOCAL LOSS")
print("=" * 60)
print(f"""
Architecture :
  - Base model : MobileNetV2 (ImageNet)
  - Head : GAP → Dense(256) → Dropout(0.4) → Dense(5, softmax)
  - Input : RGB 128x128
  - Paramètres totaux : {model.count_params():,}

CHANGEMENT vs Modèle 3 :
  → Focal Loss (gamma=2.0) au lieu de categorical_crossentropy
  → Poids alpha par classe intégrés dans la loss
  → Plus de class_weight externe (évite double pénalisation)

  Poids alpha :""")
for i, label in enumerate(eth_labels):
    print(f"    {label} : {alpha_weights[i]:.3f}")

print(f"""
Entraînement :
  - Phase 1 : Head seul (lr=0.001) - {phase1_epochs} epochs
  - Phase 2 : Fine-tuning 30 dernières couches (lr=0.0001) - {len(history2.history['loss'])} epochs
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
print("FICHIERS SAUVEGARDÉS")
print("=" * 60)
print("  - ethnicity_model_transfer_focal.keras")
print("  - training_curves_transfer_focal.png")
print("  - confusion_matrix_transfer_focal.png")
print("  - metrics_per_class_transfer_focal.png")
print("  - gradcam_transfer_focal.png")
print("  - confidence_distribution_transfer_focal.png")
