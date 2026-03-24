# %% [code]
# -*- coding: utf-8 -*-
"""
# Modèle 1 : Baseline Transfer Learning - MobileNetV2 (Freeze complet)

**Entraînement sur Kaggle**

**Approche :** Transfer Learning basique
- MobileNetV2 pré-entraîné sur ImageNet (freeze complet)
- Entraîne uniquement le head (classification)
- Pas d'augmentation de données
- Cross-entropy + class weights pour le déséquilibre

**Changement vs modèles CNN custom :**
- Passage de grayscale à RGB (la couleur de peau est un signal fort)
- Utilisation de features pré-entraînées sur 1.2M images ImageNet

**Architecture :**
- MobileNetV2 (frozen) → GAP → Dense(256) → Dropout(0.4) → Dense(5, softmax)

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
    # 1. Chercher dans /kaggle/input/ (tous les sous-dossiers)
    kaggle_input = "/kaggle/input"
    if os.path.exists(kaggle_input):
        for root, dirs, files in os.walk(kaggle_input):
            jpg_files = [f for f in files if f.endswith(".jpg")]
            if len(jpg_files) > 100:  # UTKFace a 20k+ images
                print(f"Dossier d'images trouvé : {root} ({len(jpg_files)} fichiers)")
                return root

    # 2. Fallback : télécharger via kagglehub
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
        try:
            race = int(parts[2])
        except:
            race = 4

        # CHANGEMENT CLÉ : RGB au lieu de grayscale
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

print(f"Shape X : {X.shape}")

# Distribution des classes
eth_labels = ['Blanc', 'Noir', 'Asiatique', 'Indien', 'Autre']
print("\nDistribution des classes :")
for i in range(5):
    count = np.sum(y_ethnicity == i)
    print(f"  {eth_labels[i]} : {count}")

# Split train/test (80/20) stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X, y_ethnicity,
    test_size=0.2,
    random_state=SEED,
    stratify=y_ethnicity
)

# Normalisation /255
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(f"\nX_train : {X_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"X_train min/max : {X_train.min():.2f} / {X_train.max():.2f}")

# One-hot encoding
y_train_cat = to_categorical(y_train, num_classes=5)
y_test_cat = to_categorical(y_test, num_classes=5)

# Class weights pour gérer le déséquilibre
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

print(f"\nClass weights :")
for i, label in enumerate(eth_labels):
    print(f"  {label} : {class_weight_dict[i]:.3f}")

# %% [code]
"""## 3. Construction du modèle - MobileNetV2 Transfer Learning (Baseline)"""

# Charger MobileNetV2 pré-entraîné (sans le top)
base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# FREEZE COMPLET : on n'entraîne que le head
base_model.trainable = False

# Construction du modèle
# Baseline : head modéré avec dropout. L'overfit léger est ATTENDU
# car le backbone est frozen et il n'y a pas d'augmentation.
# C'est justement ce que les modèles suivants viennent corriger.
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(5, activation='softmax')(x)

model = models.Model(inputs, outputs, name='Transfer_MobileNetV2_Base')

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print(f"\nParamètres totaux : {model.count_params():,}")
print(f"Paramètres entraînables : {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")
print(f"Paramètres non-entraînables : {sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights):,}")

# %% [code]
"""## 4. Entraînement"""

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
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

print("\n" + "=" * 60)
print("ENTRAÎNEMENT - MODÈLE 1 : BASELINE TRANSFER LEARNING")
print("=" * 60)
print("Configuration :")
print("  - Base model : MobileNetV2 (frozen)")
print("  - Head : GAP → BN → Dense(128) → Dropout(0.4) → Dense(5)")
print("  - Learning rate : 0.001")
print("  - EarlyStopping patience=3 (coupe avant que l'overfit s'installe)")
print("  - Loss : categorical_crossentropy + class_weight")
print("  - Batch size : 32")
print("  - Max epochs : 25")

history = model.fit(
    X_train, y_train_cat,
    epochs=25,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr]
)

# %% [code]
"""## 5. Évaluation du modèle"""

y_pred_proba = model.predict(X_test)
y_pred = y_pred_proba.argmax(axis=1)

loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"\nAccuracy sur le test set : {accuracy*100:.2f}%")

# AUC et AP
auc_score = roc_auc_score(y_test_cat, y_pred_proba, multi_class='ovr', average='macro')
ap_score = average_precision_score(y_test_cat, y_pred_proba, average='macro')
print(f"AUC (macro) : {auc_score:.4f}")
print(f"AP (macro) : {ap_score:.4f}")

print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=eth_labels))

# %% [code]
"""## 6. Graphiques d'entraînement"""

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

plt.suptitle('Modèle 1 : Baseline Transfer Learning (MobileNetV2 Freeze)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'training_curves_transfer_base.png'), dpi=150, bbox_inches='tight')
plt.show()

# Résumé
print("=" * 50)
print("RÉSUMÉ DE L'ENTRAÎNEMENT")
print("=" * 50)
print(f"Nombre d'epochs effectuées : {len(history.history['loss'])}")
print(f"Meilleure accuracy validation : {max(history.history['val_accuracy'])*100:.2f}%")
print(f"Meilleure loss validation : {min(history.history['val_loss']):.4f}")

# %% [code]
"""## 7. Matrice de confusion"""

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
plt.title('Matrice de confusion - Modèle 1 : Baseline Transfer Learning')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix_transfer_base.png'), dpi=150)
plt.show()

# %% [code]
"""## 8. Performances par classe"""

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.arange(len(eth_labels))
width = 0.25

bars1 = ax.bar(x_pos - width, precision * 100, width, label='Precision', color='steelblue')
bars2 = ax.bar(x_pos, recall * 100, width, label='Recall', color='teal')
bars3 = ax.bar(x_pos + width, f1 * 100, width, label='F1-Score', color='coral')

ax.set_ylabel('Score (%)')
ax.set_title('Performances par classe - Modèle 1 : Baseline Transfer Learning')
ax.set_xticks(x_pos)
ax.set_xticklabels(eth_labels)
ax.legend()
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3, axis='y')

# Ajouter les valeurs sur les barres
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'metrics_per_class_transfer_base.png'), dpi=150)
plt.show()

# %% [code]
"""## 9. Grad-CAM : Visualiser comment le modèle réfléchit"""


def make_gradcam_heatmap(img_array, model, base_model, pred_index=None):
    """
    Génère une heatmap Grad-CAM compatible avec les modèles Keras imbriqués.

    Approche : on construit un sous-modèle du base_model qui sort les features
    de la dernière couche conv, puis on applique manuellement les couches du head.
    Tout est fait dans le même contexte GradientTape pour que les gradients
    puissent circuler de la prédiction jusqu'aux feature maps conv.
    """
    # Trouver la dernière couche convolutionnelle
    last_conv_layer_name = None
    for layer in base_model.layers[::-1]:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name is None:
        return np.ones((4, 4), dtype=np.float32) * 0.5

    # Sous-modèle : entrée du base_model → [dernière conv, sortie finale]
    sub_base = tf.keras.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

    # Identifier les couches du head (après le base_model dans le modèle complet)
    head_layers = []
    found_base = False
    for layer in model.layers:
        if layer.name == base_model.name:
            found_base = True
            continue
        if found_base:
            head_layers.append(layer)

    with tf.GradientTape() as tape:
        # Forward pass à travers le sous-modèle base
        conv_outputs, base_features = sub_base(img_array, training=False)
        tape.watch(conv_outputs)

        # Appliquer les couches du head manuellement
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
    # Redimensionner la heatmap à la taille de l'image
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], (IMG_SIZE, IMG_SIZE)
    ).numpy().squeeze()

    # Appliquer la colormap
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]

    # Superposer
    superimposed = heatmap_colored * alpha + img * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 1)
    return superimposed


# Générer les Grad-CAM sur 8 images du test set
print("Génération des Grad-CAM heatmaps...")
fig, axes = plt.subplots(4, 4, figsize=(16, 16))

# Sélectionner 8 images aléatoires
np.random.seed(SEED)
indices = np.random.choice(len(X_test), 8, replace=False)

for i, idx in enumerate(indices):
    img = X_test[idx]
    img_array = np.expand_dims(img, axis=0)

    # Prédiction
    pred_proba = model.predict(img_array, verbose=0)
    pred_class = np.argmax(pred_proba)
    true_class = y_test[idx]
    confidence = pred_proba[0][pred_class] * 100

    # Grad-CAM
    heatmap = make_gradcam_heatmap(img_array, model, base_model, pred_class)
    superimposed = display_gradcam(img, heatmap)

    # Couleur du titre : vert si correct, rouge si faux
    color = 'green' if pred_class == true_class else 'red'

    # Image originale (colonne gauche)
    row = i // 2
    col = (i % 2) * 2
    axes[row, col].imshow(img)
    axes[row, col].set_title(
        f"Réel: {eth_labels[true_class]}\nPrédit: {eth_labels[pred_class]} ({confidence:.1f}%)",
        fontsize=10, color=color
    )
    axes[row, col].axis('off')

    # Grad-CAM (colonne droite)
    axes[row, col + 1].imshow(superimposed)
    axes[row, col + 1].set_title('Grad-CAM', fontsize=10)
    axes[row, col + 1].axis('off')

plt.suptitle('Grad-CAM - Modèle 1 : Baseline Transfer Learning\n(Zones chaudes = où le modèle regarde)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'gradcam_transfer_base.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [code]
"""## 10. Distribution de confiance des prédictions"""

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
ax.set_title('Distribution de confiance - Modèle 1 : Baseline Transfer Learning')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confidence_distribution_transfer_base.png'), dpi=150)
plt.show()

# Statistiques de confiance
print(f"Confiance moyenne (prédictions correctes) : {y_pred_max_proba[correct_mask].mean()*100:.1f}%")
print(f"Confiance moyenne (prédictions incorrectes) : {y_pred_max_proba[incorrect_mask].mean()*100:.1f}%")

# %% [code]
"""## 11. Sauvegarde du modèle"""

model.save(os.path.join(OUTPUT_PATH, 'ethnicity_model_transfer_base.keras'))
print(f"Modèle sauvegardé : {OUTPUT_PATH}/ethnicity_model_transfer_base.keras")

# %% [code]
"""## 12. Résumé final"""

print("=" * 60)
print("RÉSUMÉ FINAL - MODÈLE 1 : BASELINE TRANSFER LEARNING")
print("=" * 60)
print(f"""
Architecture :
  - Base model : MobileNetV2 (ImageNet, frozen)
  - Head : GAP → Dense(256) → Dropout(0.4) → Dense(5, softmax)
  - Input : RGB 128x128
  - Paramètres totaux : {model.count_params():,}

Entraînement :
  - Loss : categorical_crossentropy + class_weight
  - Optimizer : Adam (lr=0.001)
  - Epochs : {len(history.history['loss'])}
  - Pas d'augmentation de données

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
print("  - ethnicity_model_transfer_base.keras")
print("  - training_curves_transfer_base.png")
print("  - confusion_matrix_transfer_base.png")
print("  - metrics_per_class_transfer_base.png")
print("  - gradcam_transfer_base.png")
print("  - confidence_distribution_transfer_base.png")
