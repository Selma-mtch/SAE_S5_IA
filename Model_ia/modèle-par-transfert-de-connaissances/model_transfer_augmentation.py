# %% [code]
# -*- coding: utf-8 -*-
"""
# Modèle 3 : Data Augmentation - MobileNetV2

**Entraînement sur Kaggle**

**Changement unique vs Modèle 2 :** Ajout de data augmentation via couches Keras
- RandomFlip horizontal (les visages sont symétriques)
- RandomRotation ±18° (légère inclinaison)
- RandomZoom ±10%
- RandomBrightness ±10% (conservateur car la couleur = feature)
- RandomContrast ±10%

**Pourquoi :** L'augmentation aide à régulariser le modèle et à mieux généraliser,
surtout avec un dataset de seulement ~20k images. Les augmentations sont choisies
spécifiquement pour les visages (pas de flip vertical, rotations modérées).

**Architecture :** Identique au Modèle 2
- MobileNetV2 + fine-tuning 2 phases → GAP → Dense(256) → Dropout(0.4) → Dense(5)

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
"""## 2. Préparation des données (Age + Genre + Ethnicité)"""

X = images
y_age = labels[:, 0].astype('float32')
y_gender = labels[:, 1].astype('float32')
y_ethnicity = labels[:, 2]

eth_labels = ['Blanc', 'Noir', 'Asiatique', 'Indien', 'Autre']
print("\nDistribution des classes (ethnicité) :")
for i in range(5):
    count = np.sum(y_ethnicity == i)
    print(f"  {eth_labels[i]} : {count}")

print(f"\nAge - min: {y_age.min()}, max: {y_age.max()}, mean: {y_age.mean():.1f}")
print(f"Gender - distribution: {Counter(y_gender.astype(int))}")

X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test, y_eth_train, y_eth_test = train_test_split(
    X, y_age, y_gender, y_ethnicity,
    test_size=0.2,
    random_state=SEED,
    stratify=y_ethnicity
)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(f"\nX_train : {X_train.shape}")
print(f"X_test : {X_test.shape}")

y_eth_train_cat = to_categorical(y_eth_train, num_classes=5)
y_eth_test_cat = to_categorical(y_eth_test, num_classes=5)

# Aliases pour compatibilité avec les sections d'évaluation
y_train = y_eth_train
y_test = y_eth_test
y_train_cat = y_eth_train_cat
y_test_cat = y_eth_test_cat

# %% [code]
"""## 3. Data Augmentation (CHANGEMENT CLÉ vs Modèle 2)

Couches Keras intégrées au modèle, actives uniquement pendant l'entraînement.
Augmentations choisies spécifiquement pour les visages :
- Pas de flip vertical (les visages ne sont jamais à l'envers)
- Rotations modérées (les visages UTKFace sont déjà alignés)
- Brightness/contraste conservateurs (la couleur de peau est informative)
"""

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),          # ±18°
    layers.RandomZoom((-0.1, 0.1)),       # ±10%
    layers.RandomBrightness(0.1),         # ±10% (conservateur)
    layers.RandomContrast(0.1),           # ±10%
], name='data_augmentation')

print("Couches d'augmentation :")
for layer in data_augmentation.layers:
    print(f"  - {layer.name}")

# %% [code]
"""## 4. Construction du modèle"""

base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# CHANGEMENT : ajout de l'augmentation avant le base model
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)

shared = layers.Dense(256, activation='relu')(x)
shared = layers.Dropout(0.4)(shared)

# Branche Age (régression)
age_branch = layers.Dense(128, activation='relu')(shared)
age_branch = layers.Dense(64, activation='relu')(age_branch)
age_output = layers.Dense(1, activation='linear', name='age')(age_branch)

# Branche Genre (classification binaire)
gender_branch = layers.Dense(128, activation='relu')(shared)
gender_branch = layers.Dropout(0.3)(gender_branch)
gender_output = layers.Dense(1, activation='sigmoid', name='gender')(gender_branch)

# Branche Ethnicité (classification multi-classe)
eth_branch = layers.Dense(128, activation='relu')(shared)
eth_branch = layers.Dropout(0.4)(eth_branch)
ethnicity_output = layers.Dense(5, activation='softmax', name='ethnicity')(eth_branch)

model = models.Model(inputs, [age_output, gender_output, ethnicity_output], name='Transfer_MobileNetV2_Augmentation')

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss={
        'age': tf.keras.losses.Huber(),
        'gender': 'binary_crossentropy',
        'ethnicity': 'categorical_crossentropy'
    },
    loss_weights={'age': 0.4, 'gender': 1.0, 'ethnicity': 1.0},
    metrics={
        'age': ['mae'],
        'gender': ['accuracy'],
        'ethnicity': ['accuracy']
    }
)

model.summary()

# %% [code]
"""## 5. Phase 1 : Entraînement du head seul"""

# Split manuel train/val pour pouvoir passer des dict labels
val_split = 0.2
n_val = int(len(X_train) * val_split)
indices = np.random.RandomState(SEED).permutation(len(X_train))
val_idx, tr_idx = indices[:n_val], indices[n_val:]

X_tr, X_val = X_train[tr_idx], X_train[val_idx]
y_age_tr, y_age_val = y_age_train[tr_idx], y_age_train[val_idx]
y_gender_tr, y_gender_val = y_gender_train[tr_idx], y_gender_train[val_idx]
y_eth_tr_cat, y_eth_val_cat = y_eth_train_cat[tr_idx], y_eth_train_cat[val_idx]

early_stop = EarlyStopping(monitor='val_ethnicity_accuracy', mode='max', patience=10, restore_best_weights=True, start_from_epoch=30, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

print("\n" + "=" * 60)
print("PHASE 1 : ENTRAÎNEMENT DU HEAD SEUL + AUGMENTATION")
print("=" * 60)
print("  - Epochs : 50 (minimum 30)")

history1 = model.fit(
    X_tr,
    {'age': y_age_tr, 'gender': y_gender_tr, 'ethnicity': y_eth_tr_cat},
    epochs=50,
    batch_size=32,
    validation_data=(X_val, {'age': y_age_val, 'gender': y_gender_val, 'ethnicity': y_eth_val_cat}),
    callbacks=[early_stop, reduce_lr]
)

print(f"\nPhase 1 terminée - Ethnicity Accuracy val : {max(history1.history['val_ethnicity_accuracy'])*100:.2f}%")

# %% [code]
"""## 6. Phase 2 : Fine-tuning"""

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss={
        'age': tf.keras.losses.Huber(),
        'gender': 'binary_crossentropy',
        'ethnicity': 'categorical_crossentropy'
    },
    loss_weights={'age': 0.4, 'gender': 1.0, 'ethnicity': 1.0},
    metrics={
        'age': ['mae'],
        'gender': ['accuracy'],
        'ethnicity': ['accuracy']
    }
)

print(f"Phase 2 - Paramètres entraînables : {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")

early_stop2 = EarlyStopping(monitor='val_ethnicity_accuracy', mode='max', patience=10, restore_best_weights=True, start_from_epoch=30, verbose=1)
reduce_lr2 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

print("\n" + "=" * 60)
print("PHASE 2 : FINE-TUNING + AUGMENTATION")
print("=" * 60)
print("  - Epochs : 50 (minimum 30)")

history2 = model.fit(
    X_tr,
    {'age': y_age_tr, 'gender': y_gender_tr, 'ethnicity': y_eth_tr_cat},
    epochs=50,
    batch_size=32,
    validation_data=(X_val, {'age': y_age_val, 'gender': y_gender_val, 'ethnicity': y_eth_val_cat}),
    callbacks=[early_stop2, reduce_lr2]
)

print(f"\nPhase 2 terminée - Ethnicity Accuracy val : {max(history2.history['val_ethnicity_accuracy'])*100:.2f}%")

# %% [code]
"""## 7. Évaluation du modèle"""

predictions = model.predict(X_test)
y_pred_age = predictions[0].flatten()
y_pred_gender = (predictions[1].flatten() > 0.5).astype(int)
y_pred_proba = predictions[2]
y_pred = y_pred_proba.argmax(axis=1)

age_mae = np.mean(np.abs(y_pred_age - y_age_test))
gender_acc = np.mean(y_pred_gender == y_gender_test)
accuracy = np.mean(y_pred == y_test)

print(f"\n--- Résultats Multi-tâches ---")
print(f"Age MAE : {age_mae:.2f} ans")
print(f"Gender Accuracy : {gender_acc*100:.2f}%")
print(f"Ethnicity Accuracy : {accuracy*100:.2f}%")

auc_score = roc_auc_score(y_test_cat, y_pred_proba, multi_class='ovr', average='macro')
ap_score = average_precision_score(y_test_cat, y_pred_proba, average='macro')
print(f"Ethnicity AUC (macro) : {auc_score:.4f}")
print(f"Ethnicity AP (macro) : {ap_score:.4f}")

print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=eth_labels))

# %% [code]
"""## 8. Graphiques d'entraînement (2 phases combinées)"""

all_loss = history1.history['loss'] + history2.history['loss']
all_val_loss = history1.history['val_loss'] + history2.history['val_loss']
all_acc = history1.history['ethnicity_accuracy'] + history2.history['ethnicity_accuracy']
all_val_acc = history1.history['val_ethnicity_accuracy'] + history2.history['val_ethnicity_accuracy']
phase1_epochs = len(history1.history['loss'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(all_loss, label='Train', linewidth=2, marker='o', markersize=3)
axes[0].plot(all_val_loss, label='Validation', linewidth=2, marker='s', markersize=3)
axes[0].axvline(x=phase1_epochs - 0.5, color='red', linestyle='--', alpha=0.7, label='Début fine-tuning')
axes[0].set_title('Loss durant l\'entraînement', fontsize=12)
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

plt.suptitle('Modèle 3 : Fine-tuning + Data Augmentation', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'training_curves_transfer_augmentation.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [code]
"""## 9. Matrice de confusion"""

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
plt.title('Matrice de confusion - Modèle 3 : Data Augmentation')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix_transfer_augmentation.png'), dpi=150)
plt.show()

# %% [code]
"""## 10. Performances par classe"""

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.arange(len(eth_labels))
width = 0.25

bars1 = ax.bar(x_pos - width, precision * 100, width, label='Precision', color='steelblue')
bars2 = ax.bar(x_pos, recall * 100, width, label='Recall', color='teal')
bars3 = ax.bar(x_pos + width, f1 * 100, width, label='F1-Score', color='coral')

ax.set_ylabel('Score (%)')
ax.set_title('Performances par classe - Modèle 3 : Data Augmentation')
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
plt.savefig(os.path.join(OUTPUT_PATH, 'metrics_per_class_transfer_augmentation.png'), dpi=150)
plt.show()

# %% [code]
"""## 11. Grad-CAM : Visualiser comment le modèle réfléchit"""


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

    # Couches du head = tout après le base_model (on skip aussi l'augmentation)
    head_layers = []
    found_base = False
    for layer in model.layers:
        if layer.name == base_model.name:
            found_base = True
            continue
        if found_base:
            head_layers.append(layer)

    with tf.GradientTape() as tape:
        # Pas d'augmentation pour le Grad-CAM (on passe directement dans le base model)
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

    preds = model.predict(img_array, verbose=0)
    pred_eth_proba = preds[2]  # ethnicity output
    pred_class = np.argmax(pred_eth_proba)
    true_class = y_test[idx]
    confidence = pred_eth_proba[0][pred_class] * 100

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

plt.suptitle('Grad-CAM - Modèle 3 : Data Augmentation\n(Zones chaudes = où le modèle regarde)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'gradcam_transfer_augmentation.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [code]
"""## 12. Distribution de confiance des prédictions"""

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
ax.set_title('Distribution de confiance - Modèle 3 : Data Augmentation')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confidence_distribution_transfer_augmentation.png'), dpi=150)
plt.show()

print(f"Confiance moyenne (correctes) : {y_pred_max_proba[correct_mask].mean()*100:.1f}%")
print(f"Confiance moyenne (incorrectes) : {y_pred_max_proba[incorrect_mask].mean()*100:.1f}%")

# %% [code]
"""## 13. Sauvegarde du modèle"""

model.save(os.path.join(OUTPUT_PATH, 'multitask_model_transfer_augmentation.keras'))
print(f"Modèle sauvegardé : {OUTPUT_PATH}/multitask_model_transfer_augmentation.keras")

# %% [code]
"""## 14. Export TensorFlow Lite

Le modèle contient des couches d'augmentation (RandomFlip, RandomRotation, etc.)
qui ne sont pas supportées par TFLite. On crée un modèle d'inférence propre
sans augmentation, on copie les poids, puis on exporte.
"""

print("\n" + "=" * 60)
print("EXPORT TENSORFLOW LITE")
print("=" * 60)

# Construire un modèle d'inférence SANS augmentation (3 sorties)
# On reconstruit le graphe en sautant les couches d'augmentation
inference_inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inference_inputs, training=False)

# Reconstruire le head multi-tâches en réutilisant les couches entraînées
# On parcourt les couches après le base_model en recréant le graphe
layer_map = {}  # nom du tensor d'origine -> nouveau tensor
found_base = False
for layer in model.layers:
    if layer.name == base_model.name:
        found_base = True
        # Le base_model output est maintenant x
        continue
    if not found_base:
        continue

    # Déterminer les entrées de cette couche dans le modèle original
    inbound = layer.input if not isinstance(layer.input, list) else layer.input
    if isinstance(inbound, list):
        layer_inputs = [layer_map.get(t.ref(), x) for t in inbound]
        out = layer(layer_inputs)
    else:
        # Trouver le bon input pour cette couche
        input_ref = inbound.ref()
        inp = layer_map.get(input_ref, x)
        out = layer(inp)

    # Enregistrer le output
    if isinstance(layer.output, list):
        for t in layer.output:
            layer_map[t.ref()] = out
    else:
        layer_map[layer.output.ref()] = out

# Récupérer les 3 sorties par nom
age_out = None
gender_out = None
eth_out = None
for layer in model.layers:
    if layer.name == 'age':
        age_out = layer_map[layer.output.ref()]
    elif layer.name == 'gender':
        gender_out = layer_map[layer.output.ref()]
    elif layer.name == 'ethnicity':
        eth_out = layer_map[layer.output.ref()]

inference_model = models.Model(inference_inputs, [age_out, gender_out, eth_out], name='Transfer_Augmentation_Inference')
print(f"Modèle d'inférence créé : {inference_model.count_params():,} paramètres")

# Vérifier que les prédictions sont identiques
test_img = X_test[:1]
pred_original = model.predict(test_img, verbose=0)
pred_inference = inference_model.predict(test_img, verbose=0)
for i, name in enumerate(['age', 'gender', 'ethnicity']):
    diff = np.max(np.abs(np.array(pred_original[i]) - np.array(pred_inference[i])))
    print(f"  Vérification {name} : diff max = {diff:.6f}")

# Export TFLite
tflite_path = os.path.join(OUTPUT_PATH, 'multitask_transfer_augmentation.tflite')

converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
print(f"Modèle TFLite sauvegardé : {tflite_path} ({size_mb:.1f} MB)")

# Vérification
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"  Input  : shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
for i, od in enumerate(output_details):
    print(f"  Output {i} : shape={od['shape']}, dtype={od['dtype']}")

test_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
for i, od in enumerate(output_details):
    test_output = interpreter.get_tensor(od['index'])
    print(f"  Output {i} prediction : {test_output.flatten()}")
print("  TFLite OK (3 sorties) !")

# %% [code]
"""## 15. Résumé final"""

print("=" * 60)
print("RÉSUMÉ FINAL - MODÈLE 3 : MULTI-TÂCHES + DATA AUGMENTATION")
print("=" * 60)
print(f"""
Architecture :
  - Base model : MobileNetV2 (ImageNet)
  - Shared : GAP → BN → Dense(256) → Dropout(0.4)
  - Branche Age : Dense(128) → Dense(64) → Dense(1, linear)
  - Branche Genre : Dense(128) → Dropout(0.3) → Dense(1, sigmoid)
  - Branche Ethnicité : Dense(128) → Dropout(0.4) → Dense(5, softmax)
  - Input : RGB 128x128
  - Paramètres totaux : {model.count_params():,}

CHANGEMENT vs Modèle 2 :
  → Multi-tâches (age + genre + ethnicité)
  → Data augmentation intégrée au modèle :
    - RandomFlip horizontal
    - RandomRotation ±18°
    - RandomZoom ±10%
    - RandomBrightness ±10%
    - RandomContrast ±10%

Entraînement :
  - Phase 1 : Head seul (lr=0.001) - {phase1_epochs} epochs
  - Phase 2 : Fine-tuning 30 dernières couches (lr=0.0001) - {len(history2.history['loss'])} epochs
  - Loss : Huber (age) + binary_crossentropy (gender) + categorical_crossentropy (ethnicity)
  - Loss weights : age=0.4, gender=1.0, ethnicity=1.0

Résultats :
  - Age MAE : {age_mae:.2f} ans
  - Gender Accuracy : {gender_acc*100:.2f}%
  - Ethnicity Accuracy : {accuracy*100:.2f}%
  - Ethnicity AUC (macro) : {auc_score:.4f}
  - Ethnicity AP (macro) : {ap_score:.4f}
""")

print("Performances par classe :")
for i, label in enumerate(eth_labels):
    print(f"  {label:10s} - Precision: {precision[i]*100:5.1f}% | Recall: {recall[i]*100:5.1f}% | F1: {f1[i]*100:5.1f}% | Support: {support[i]}")

print(f"\n{'=' * 60}")
print("FICHIERS SAUVEGARDÉS")
print("=" * 60)
print("  - multitask_model_transfer_augmentation.keras")
print("  - training_curves_transfer_augmentation.png")
print("  - confusion_matrix_transfer_augmentation.png")
print("  - metrics_per_class_transfer_augmentation.png")
print("  - gradcam_transfer_augmentation.png")
print("  - confidence_distribution_transfer_augmentation.png")
print("  - multitask_transfer_augmentation.tflite")
