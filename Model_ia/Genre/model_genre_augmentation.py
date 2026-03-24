# %% [code]
# -*- coding: utf-8 -*-
"""
# Modèle 3 : Data Augmentation - Classification de Genre

**Entraînement sur Kaggle**

**Changement unique vs Modèle 2 :** Ajout de data augmentation via couches Keras
- RandomFlip horizontal (les visages sont symétriques)
- RandomRotation ±18° (légère inclinaison)
- RandomZoom ±10%
- RandomBrightness ±15%
- RandomContrast ±15%

**Pourquoi :** L'augmentation aide à régulariser le modèle et à mieux généraliser,
surtout avec un dataset de seulement ~20k images. Les augmentations sont choisies
spécifiquement pour les visages (pas de flip vertical, rotations modérées).

**Architecture (identique au Modèle 2 + augmentation) :**
- Input(128,128,3) → data_augmentation
- Bloc 1 : Conv2D(32) → BN → ReLU → SE(4) + Skip → MaxPool → Dropout(0.25)
- Bloc 2 : SeparableConv2D(64) → BN → ReLU → SE(8) + Skip → MaxPool → Dropout(0.25)
- Bloc 3 : SeparableConv2D(128) → BN → ReLU → SE(8) + Skip → MaxPool → Dropout(0.3)
- Bloc 4 : SeparableConv2D(256) → BN → ReLU → SE(16) + Skip → MaxPool → Dropout(0.3)
- GAP → Dense(256, relu) → Dropout(0.5) → Dense(1, sigmoid)

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

# Class weights
class_weights_values = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weight_dict = {0: class_weights_values[0], 1: class_weights_values[1]}
print(f"\nClass weights : {class_weight_dict}")

# %% [code]
"""## 3. Data Augmentation (CHANGEMENT CLÉ vs Modèle 2)

Couches Keras intégrées au modèle, actives uniquement pendant l'entraînement.
Augmentations choisies spécifiquement pour les visages :
- Pas de flip vertical (les visages ne sont jamais à l'envers)
- Rotations modérées (les visages UTKFace sont déjà alignés)
- Brightness/contraste un peu plus agressifs que pour l'ethnicité (le genre ne dépend pas de la couleur)
"""

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),          # ±18°
    layers.RandomZoom((-0.1, 0.1)),       # ±10%
    layers.RandomBrightness(0.15),        # ±15%
    layers.RandomContrast(0.15),          # ±15%
], name='data_augmentation')

print("Couches d'augmentation :")
for layer in data_augmentation.layers:
    print(f"  - {layer.name}")

# %% [code]
"""## 4. Construction du modèle - ResNet + SE + SeparableConv + Augmentation

Architecture identique au Modèle 2 (4 blocs) avec data augmentation en entrée.
"""


def se_block(x, ratio=8):
    """
    Squeeze-and-Excitation Block

    Apprend l'importance de chaque channel/filtre.
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


# --- Construction avec Functional API ---
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Data Augmentation (actif uniquement pendant l'entraînement)
x = data_augmentation(inputs)

# ================================================================
# BLOC 1 : 32 filtres
# Conv2D classique (SeparableConv pas efficace sur 3 channels d'entrée)
# ================================================================
shortcut1 = x  # Sauvegarder pour skip connection
x = layers.Conv2D(32, (3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = se_block(x, ratio=4)

shortcut1 = layers.Conv2D(32, (1, 1), padding='same')(shortcut1)
x = layers.Add()([x, shortcut1])

x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.25)(x)

# ================================================================
# BLOC 2 : 64 filtres
# ================================================================
shortcut2 = x
x = layers.SeparableConv2D(64, (3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = se_block(x, ratio=8)

shortcut2 = layers.Conv2D(64, (1, 1), padding='same')(shortcut2)
x = layers.Add()([x, shortcut2])

x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.25)(x)

# ================================================================
# BLOC 3 : 128 filtres
# ================================================================
shortcut3 = x
x = layers.SeparableConv2D(128, (3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = se_block(x, ratio=8)

shortcut3 = layers.Conv2D(128, (1, 1), padding='same')(shortcut3)
x = layers.Add()([x, shortcut3])

x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.3)(x)

# ================================================================
# BLOC 4 : 256 filtres
# ================================================================
shortcut4 = x
x = layers.SeparableConv2D(256, (3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = se_block(x, ratio=16)

shortcut4 = layers.Conv2D(256, (1, 1), padding='same')(shortcut4)
x = layers.Add()([x, shortcut4])

x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.3)(x)

# ================================================================
# HEAD : Classification binaire
# ================================================================
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs, name='CNN_Genre_Augmentation')

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

print(f"\nParamètres totaux : {model.count_params():,}")

# %% [code]
"""## 5. Entraînement"""

early_stop = EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    patience=12,
    restore_best_weights=True,
    start_from_epoch=20,
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
print("ENTRAÎNEMENT - MODÈLE 3 : DATA AUGMENTATION GENRE")
print("=" * 60)
print("Configuration :")
print("  - Architecture : 4 blocs ResNet+SE+SeparableConv (32→64→128→256)")
print("  - Data Augmentation : RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast")
print("  - Loss : binary_crossentropy")
print("  - Optimizer : Adam (lr=0.001)")
print("  - Epochs : 60 (plus car augmentation ralentit la convergence)")
print("  - EarlyStopping : patience=12, start_from_epoch=20")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr]
)

# %% [code]
"""## 6. Évaluation du modèle"""

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
"""## 7. Graphiques d'entraînement"""

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

plt.suptitle('Modèle 3 : Data Augmentation Genre', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'training_curves_genre_augmentation.png'), dpi=150, bbox_inches='tight')
plt.show()

print("=" * 50)
print("RÉSUMÉ DE L'ENTRAÎNEMENT")
print("=" * 50)
print(f"Nombre d'epochs effectuées : {len(history.history['loss'])}")
print(f"Meilleure accuracy validation : {max(history.history['val_accuracy'])*100:.2f}%")
print(f"Meilleure loss validation : {min(history.history['val_loss']):.4f}")

# %% [code]
"""## 8. Matrice de confusion"""

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
plt.title('Matrice de confusion - Modèle 3 : Data Augmentation Genre')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix_genre_augmentation.png'), dpi=150)
plt.show()

# %% [code]
"""## 9. Performances par classe"""

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))

x_pos = np.arange(len(gender_labels))
width = 0.25

bars1 = ax.bar(x_pos - width, precision * 100, width, label='Precision', color='steelblue')
bars2 = ax.bar(x_pos, recall * 100, width, label='Recall', color='teal')
bars3 = ax.bar(x_pos + width, f1 * 100, width, label='F1-Score', color='coral')

ax.set_ylabel('Score (%)')
ax.set_title('Performances par classe - Modèle 3 : Data Augmentation Genre')
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
plt.savefig(os.path.join(OUTPUT_PATH, 'metrics_per_class_genre_augmentation.png'), dpi=150)
plt.show()

# %% [code]
"""## 10. Grad-CAM : Visualiser comment le modèle réfléchit"""


def make_gradcam_heatmap(img_array, model, pred_index=None):
    """
    Génère une heatmap Grad-CAM pour un CNN custom (architecture plate).
    Les couches d'augmentation sont no-op quand training=False, donc le même
    modèle fonctionne directement pour Grad-CAM.
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

plt.suptitle('Grad-CAM - Modèle 3 : Data Augmentation Genre\n(Zones chaudes = où le modèle regarde)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'gradcam_genre_augmentation.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [code]
"""## 11. Distribution de confiance des prédictions"""

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
ax.set_title('Distribution de confiance - Modèle 3 : Data Augmentation Genre')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confidence_distribution_genre_augmentation.png'), dpi=150)
plt.show()

print(f"Confiance moyenne (prédictions correctes) : {y_pred_confidence[correct_mask].mean()*100:.1f}%")
print(f"Confiance moyenne (prédictions incorrectes) : {y_pred_confidence[incorrect_mask].mean()*100:.1f}%")

# %% [code]
"""## 12. Sauvegarde du modèle"""

model.save(os.path.join(OUTPUT_PATH, 'gender_model_augmentation.keras'))
print(f"Modèle sauvegardé : {OUTPUT_PATH}/gender_model_augmentation.keras")

# %% [code]
"""## 13. Export TensorFlow Lite

Le modèle contient des couches d'augmentation (RandomFlip, RandomRotation, etc.)
qui ne sont pas supportées par TFLite. On crée un modèle d'inférence propre
sans augmentation, on réutilise les poids entraînés, puis on exporte.
"""

print("\n" + "=" * 60)
print("EXPORT TENSORFLOW LITE")
print("=" * 60)

# ================================================================
# Construire un modèle d'inférence SANS augmentation
# On reconstruit le même graphe en sautant la couche data_augmentation
# ================================================================
inference_inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# On parcourt les couches du modèle entraîné en sautant l'augmentation
# et en reconstruisant le graphe avec les mêmes couches (donc mêmes poids)
layer_map = {}  # tensor ref d'origine -> nouveau tensor

for layer in model.layers:
    # Skip la couche Input (on a notre propre input)
    if isinstance(layer, tf.keras.layers.InputLayer):
        # Mapper l'output de l'InputLayer original vers notre nouvel input
        layer_map[layer.output.ref()] = inference_inputs
        continue

    # Skip la couche data_augmentation
    if layer.name == 'data_augmentation':
        # L'output de l'augmentation pointe vers le même tensor que l'input
        # (on bypass l'augmentation)
        layer_map[layer.output.ref()] = inference_inputs
        continue

    # Déterminer les entrées de cette couche
    inbound = layer.input
    if isinstance(inbound, list):
        layer_input = [layer_map[t.ref()] for t in inbound]
    else:
        layer_input = layer_map[inbound.ref()]

    # Reconstruire le graphe en réutilisant la même couche (mêmes poids)
    out = layer(layer_input)

    # Enregistrer le output
    if isinstance(layer.output, list):
        for orig_t, new_t in zip(layer.output, out if isinstance(out, list) else [out]):
            layer_map[orig_t.ref()] = new_t
    else:
        layer_map[layer.output.ref()] = out

# Récupérer la sortie finale
inference_output = layer_map[model.output.ref()]

inference_model = models.Model(inference_inputs, inference_output, name='CNN_Genre_Augmentation_Inference')
print(f"Modèle d'inférence créé : {inference_model.count_params():,} paramètres")

# Vérifier que les prédictions sont identiques
print("\nVérification des prédictions (modèle entraîné vs modèle d'inférence) :")
test_img = X_test[:5]
pred_original = model.predict(test_img, verbose=0)
pred_inference = inference_model.predict(test_img, verbose=0)
diff = np.max(np.abs(pred_original - pred_inference))
print(f"  Diff max entre les 2 modèles : {diff:.8f}")
assert diff < 1e-5, f"ERREUR : les prédictions diffèrent de {diff}"
print("  Prédictions identiques !")

# Export TFLite
tflite_path = os.path.join(OUTPUT_PATH, 'gender_augmentation.tflite')

converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
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
"""## 14. Résumé final"""

print("=" * 60)
print("RÉSUMÉ FINAL - MODÈLE 3 : DATA AUGMENTATION GENRE")
print("=" * 60)
print(f"""
Architecture :
  - Type : CNN custom ResNet + SE-Net + SeparableConv + Data Augmentation
  - Bloc 1 : Conv2D(32) → BN → ReLU → SE(4) + Skip → MaxPool → Dropout(0.25)
  - Bloc 2 : SeparableConv2D(64) → BN → ReLU → SE(8) + Skip → MaxPool → Dropout(0.25)
  - Bloc 3 : SeparableConv2D(128) → BN → ReLU → SE(8) + Skip → MaxPool → Dropout(0.3)
  - Bloc 4 : SeparableConv2D(256) → BN → ReLU → SE(16) + Skip → MaxPool → Dropout(0.3)
  - Head : GAP → Dense(256) → Dropout(0.5) → Dense(1, sigmoid)
  - Input : RGB {IMG_SIZE}x{IMG_SIZE}
  - Paramètres totaux : {model.count_params():,}

CHANGEMENT vs Modèle 2 :
  → Data augmentation intégrée au modèle :
    - RandomFlip horizontal
    - RandomRotation ±18°
    - RandomZoom ±10%
    - RandomBrightness ±15%
    - RandomContrast ±15%

Entraînement :
  - Loss : binary_crossentropy
  - Optimizer : Adam (lr=0.001)
  - Epochs : {len(history.history['loss'])} (max 60)
  - EarlyStopping : patience=12, start_from_epoch=20
  - Class weights : {class_weight_dict}

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
print("  - gender_model_augmentation.keras")
print("  - gender_augmentation.tflite")
print("  - training_curves_genre_augmentation.png")
print("  - confusion_matrix_genre_augmentation.png")
print("  - metrics_per_class_genre_augmentation.png")
print("  - gradcam_genre_augmentation.png")
print("  - confidence_distribution_genre_augmentation.png")
