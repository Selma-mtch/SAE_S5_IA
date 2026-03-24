# %% [code]
# -*- coding: utf-8 -*-
"""
# Modèle 4 : Multi-tâche (Age + Genre + Ethnicité) - Focal Loss - MobileNetV2

**Entraînement sur Kaggle**

**Multi-tâche :** 3 sorties simultanées
- Age : régression (Huber loss, delta=8.0)
- Genre : classification binaire (binary crossentropy)
- Ethnicité : classification 5 classes (Focal Loss gamma=2.0, alpha pondéré)

**Architecture :** MobileNetV2 + shared dense + 3 branches spécialisées

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
"""## 2. Préparation des données (multi-tâche)"""

X = images
y_age = labels[:, 0].astype('float32')
y_gender = labels[:, 1].astype('float32')
y_ethnicity = labels[:, 2].astype('int32')

eth_labels = ['Blanc', 'Noir', 'Asiatique', 'Indien', 'Autre']
print("\nDistribution des classes d'ethnicité :")
for i in range(5):
    count = np.sum(y_ethnicity == i)
    print(f"  {eth_labels[i]} : {count}")

print(f"\nAge - min: {y_age.min()}, max: {y_age.max()}, mean: {y_age.mean():.1f}")
print(f"Genre - 0 (M): {np.sum(y_gender == 0)}, 1 (F): {np.sum(y_gender == 1)}")

# Split train/test avec stratification sur l'ethnicité
X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test, y_eth_train, y_eth_test = train_test_split(
    X, y_age, y_gender, y_ethnicity,
    test_size=0.2,
    random_state=SEED,
    stratify=y_ethnicity
)

# Split train/val depuis le train set
X_train, X_val, y_age_train, y_age_val, y_gender_train, y_gender_val, y_eth_train, y_eth_val = train_test_split(
    X_train, y_age_train, y_gender_train, y_eth_train,
    test_size=0.2,
    random_state=SEED,
    stratify=y_eth_train
)

X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode ethnicité
y_eth_train_cat = to_categorical(y_eth_train, num_classes=5)
y_eth_val_cat = to_categorical(y_eth_val, num_classes=5)
y_eth_test_cat = to_categorical(y_eth_test, num_classes=5)

print(f"\nX_train : {X_train.shape}")
print(f"X_val   : {X_val.shape}")
print(f"X_test  : {X_test.shape}")

# %% [code]
"""## 3. Focal Loss (pour la branche ethnicité)

La Focal Loss modifie la cross-entropy standard :
- FL(p) = -alpha * (1-p)^gamma * log(p)
- gamma > 0 : réduit la contribution des exemples bien classifiés
- alpha : poids par classe (classes rares -> poids plus élevé)
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


# Calcul des poids alpha à partir de y_eth_train
class_weights = compute_class_weight('balanced', classes=np.unique(y_eth_train), y=y_eth_train)

# Normaliser pour que la somme = nombre de classes
alpha_weights = list(class_weights)
alpha_sum = sum(alpha_weights)
alpha_weights = [w * 5 / alpha_sum for w in alpha_weights]

print("Poids alpha pour Focal Loss (ethnicité) :")
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
"""## 5. Construction du modèle multi-tâche"""

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

# Branche Ethnicité (classification 5 classes avec Focal Loss)
eth_branch = layers.Dense(128, activation='relu')(shared)
eth_branch = layers.Dropout(0.4)(eth_branch)
ethnicity_output = layers.Dense(5, activation='softmax', name='ethnicity')(eth_branch)

model = models.Model(inputs, [age_output, gender_output, ethnicity_output], name='Transfer_MobileNetV2_Focal')

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss={
        'age': tf.keras.losses.Huber(delta=8.0),
        'gender': 'binary_crossentropy',
        'ethnicity': focal_loss(gamma=2.0, alpha=alpha_weights),
    },
    loss_weights={'age': 0.4, 'gender': 1.0, 'ethnicity': 1.0},
    metrics={'age': ['mae'], 'gender': ['accuracy'], 'ethnicity': ['accuracy']}
)

model.summary()

# %% [code]
"""## 6. Phase 1 : Entraînement du head seul"""

early_stop = EarlyStopping(monitor='val_ethnicity_accuracy', mode='max', patience=10, restore_best_weights=True, start_from_epoch=30, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

print("\n" + "=" * 60)
print("PHASE 1 : HEAD SEUL + AUGMENTATION + MULTI-TÂCHE + FOCAL LOSS")
print("=" * 60)
print("  - Age loss : Huber (delta=8.0)")
print("  - Gender loss : Binary Crossentropy")
print("  - Ethnicity loss : Focal Loss (gamma=2.0, alpha=class weights)")
print("  - Epochs : 50")

train_labels = {'age': y_age_train, 'gender': y_gender_train, 'ethnicity': y_eth_train_cat}
val_labels = {'age': y_age_val, 'gender': y_gender_val, 'ethnicity': y_eth_val_cat}

history1 = model.fit(
    X_train, train_labels,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, val_labels),
    callbacks=[early_stop, reduce_lr]
)

print(f"\nPhase 1 terminée - Ethnicity accuracy val : {max(history1.history['val_ethnicity_accuracy'])*100:.2f}%")

# %% [code]
"""## 7. Phase 2 : Fine-tuning"""

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss={
        'age': tf.keras.losses.Huber(delta=8.0),
        'gender': 'binary_crossentropy',
        'ethnicity': focal_loss(gamma=2.0, alpha=alpha_weights),
    },
    loss_weights={'age': 0.4, 'gender': 1.0, 'ethnicity': 1.0},
    metrics={'age': ['mae'], 'gender': ['accuracy'], 'ethnicity': ['accuracy']}
)

print(f"Phase 2 - Paramètres entraînables : {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")

early_stop2 = EarlyStopping(monitor='val_ethnicity_accuracy', mode='max', patience=10, restore_best_weights=True, start_from_epoch=30, verbose=1)
reduce_lr2 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

print("\n" + "=" * 60)
print("PHASE 2 : FINE-TUNING + AUGMENTATION + MULTI-TÂCHE + FOCAL LOSS")
print("=" * 60)
print("  - Epochs : 50")

history2 = model.fit(
    X_train, train_labels,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, val_labels),
    callbacks=[early_stop2, reduce_lr2]
)

print(f"\nPhase 2 terminée - Ethnicity accuracy val : {max(history2.history['val_ethnicity_accuracy'])*100:.2f}%")

# %% [code]
"""## 8. Évaluation du modèle (multi-tâche)"""

test_labels = {'age': y_age_test, 'gender': y_gender_test, 'ethnicity': y_eth_test_cat}

# Prédictions multi-sorties
preds = model.predict(X_test)
age_preds = preds[0].flatten()
gender_preds_proba = preds[1].flatten()
eth_preds_proba = preds[2]

gender_preds = (gender_preds_proba >= 0.5).astype(int)
eth_preds = eth_preds_proba.argmax(axis=1)

# Évaluation globale
eval_results = model.evaluate(X_test, test_labels)
print("\nRésultats évaluation :")
for name, val in zip(model.metrics_names, eval_results):
    print(f"  {name} : {val:.4f}")

# Age MAE
age_mae = np.mean(np.abs(age_preds - y_age_test))
print(f"\nAge MAE : {age_mae:.2f} ans")

# Gender accuracy
gender_acc = np.mean(gender_preds == y_gender_test.astype(int))
print(f"Gender accuracy : {gender_acc*100:.2f}%")

# Ethnicity accuracy
eth_acc = np.mean(eth_preds == y_eth_test)
print(f"Ethnicity accuracy : {eth_acc*100:.2f}%")

# Ethnicity AUC
eth_auc_score = roc_auc_score(y_eth_test_cat, eth_preds_proba, multi_class='ovr', average='macro')
print(f"Ethnicity AUC (macro) : {eth_auc_score:.4f}")

# Gender AUC
gender_auc_score = roc_auc_score(y_gender_test.astype(int), gender_preds_proba)
print(f"Gender AUC : {gender_auc_score:.4f}")

print("\nRapport de classification (ethnicité) :")
print(classification_report(y_eth_test, eth_preds, target_names=eth_labels))

# %% [code]
"""## 9. Graphiques d'entraînement (2 phases combinées)"""

all_loss = history1.history['loss'] + history2.history['loss']
all_val_loss = history1.history['val_loss'] + history2.history['val_loss']
all_eth_acc = history1.history['ethnicity_accuracy'] + history2.history['ethnicity_accuracy']
all_val_eth_acc = history1.history['val_ethnicity_accuracy'] + history2.history['val_ethnicity_accuracy']
phase1_epochs = len(history1.history['loss'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(all_loss, label='Train', linewidth=2, marker='o', markersize=3)
axes[0].plot(all_val_loss, label='Validation', linewidth=2, marker='s', markersize=3)
axes[0].axvline(x=phase1_epochs - 0.5, color='red', linestyle='--', alpha=0.7, label='Début fine-tuning')
axes[0].set_title('Loss totale durant l\'entraînement', fontsize=12)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(all_eth_acc, label='Train', linewidth=2, marker='o', markersize=3)
axes[1].plot(all_val_eth_acc, label='Validation', linewidth=2, marker='s', markersize=3)
axes[1].axvline(x=phase1_epochs - 0.5, color='red', linestyle='--', alpha=0.7, label='Début fine-tuning')
axes[1].set_title('Ethnicity Accuracy durant l\'entraînement', fontsize=12)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Modèle Multi-tâche : Focal Loss (gamma=2.0)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'training_curves_transfer_focal.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [code]
"""## 10. Matrice de confusion (ethnicité)"""

cm_matrix = confusion_matrix(y_eth_test, eth_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=eth_labels,
    yticklabels=eth_labels
)
plt.title('Matrice de confusion - Ethnicité (Multi-tâche + Focal Loss)')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix_transfer_focal.png'), dpi=150)
plt.show()

# %% [code]
"""## 11. Performances par classe (ethnicité)"""

precision, recall, f1, support = precision_recall_fscore_support(y_eth_test, eth_preds)

fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.arange(len(eth_labels))
width = 0.25

bars1 = ax.bar(x_pos - width, precision * 100, width, label='Precision', color='steelblue')
bars2 = ax.bar(x_pos, recall * 100, width, label='Recall', color='teal')
bars3 = ax.bar(x_pos + width, f1 * 100, width, label='F1-Score', color='coral')

ax.set_ylabel('Score (%)')
ax.set_title('Performances par classe - Ethnicité (Multi-tâche + Focal Loss)')
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

    # Build a mini-model for the ethnicity branch from base output to ethnicity output
    head_layers = []
    found_base = False
    for layer in model.layers:
        if layer.name == base_model.name:
            found_base = True
            continue
        if found_base and layer.name != 'age' and layer.name != 'gender':
            head_layers.append(layer)

    with tf.GradientTape() as tape:
        conv_outputs, base_features = sub_base(img_array, training=False)
        tape.watch(conv_outputs)

        # Forward through the ethnicity path
        # We use the full model and grab the ethnicity output (index 2)
        all_preds = model(img_array, training=False)
        eth_predictions = all_preds[2]  # ethnicity output

        if pred_index is None:
            pred_index = tf.argmax(eth_predictions[0])
        class_channel = eth_predictions[:, pred_index]

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

    all_pred = model.predict(img_array, verbose=0)
    pred_eth_proba = all_pred[2]
    pred_class = np.argmax(pred_eth_proba)
    true_class = y_eth_test[idx]
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

plt.suptitle('Grad-CAM - Multi-tâche + Focal Loss\n(Zones chaudes = où le modèle regarde)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'gradcam_transfer_focal.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [code]
"""## 13. Distribution de confiance des prédictions (ethnicité)"""

y_pred_max_proba = np.max(eth_preds_proba, axis=1)
correct_mask = (eth_preds == y_eth_test)
incorrect_mask = ~correct_mask

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(y_pred_max_proba[correct_mask], bins=30, alpha=0.7,
        label=f'Correct ({correct_mask.sum()})', color='green', edgecolor='darkgreen')
ax.hist(y_pred_max_proba[incorrect_mask], bins=30, alpha=0.7,
        label=f'Incorrect ({incorrect_mask.sum()})', color='red', edgecolor='darkred')

ax.set_xlabel('Probabilité de la classe prédite')
ax.set_ylabel('Nombre de prédictions')
ax.set_title('Distribution de confiance (ethnicité) - Multi-tâche + Focal Loss')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confidence_distribution_transfer_focal.png'), dpi=150)
plt.show()

print(f"Confiance moyenne (correctes) : {y_pred_max_proba[correct_mask].mean()*100:.1f}%")
print(f"Confiance moyenne (incorrectes) : {y_pred_max_proba[incorrect_mask].mean()*100:.1f}%")

# %% [code]
"""## 14. Sauvegarde du modèle"""

model.save(os.path.join(OUTPUT_PATH, 'multitask_model_transfer_focal.keras'))
print(f"Modèle sauvegardé : {OUTPUT_PATH}/multitask_model_transfer_focal.keras")

# %% [code]
"""## 15. Export TensorFlow Lite (3 sorties)"""

print("\n" + "=" * 60)
print("EXPORT TENSORFLOW LITE (MULTI-TÂCHE)")
print("=" * 60)

# Modèle d'inférence sans augmentation - 3 sorties
# On réutilise les couches existantes du modèle entraîné (poids partagés)
inference_inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inference_inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)

# Retrouver les couches du modèle entraîné par type et position
bn_layer = [l for l in model.layers if isinstance(l, layers.BatchNormalization) and l.name != base_model.name][0]
dense_256 = [l for l in model.layers if isinstance(l, layers.Dense) and l.units == 256][0]
dropout_layers_all = [l for l in model.layers if isinstance(l, layers.Dropout)]
dense_128_layers = [l for l in model.layers if isinstance(l, layers.Dense) and l.units == 128]
dense_64_layers = [l for l in model.layers if isinstance(l, layers.Dense) and l.units == 64]

age_output_layer = model.get_layer('age')
gender_output_layer = model.get_layer('gender')
eth_output_layer = model.get_layer('ethnicity')

# Shared path (BN -> Dense(256) -> Dropout)
x_inf = bn_layer(x)
x_inf = dense_256(x_inf)
x_inf = dropout_layers_all[0](x_inf)  # shared dropout

# Age branch : Dense(128) -> Dense(64) -> Dense(1)
age_inf = dense_128_layers[0](x_inf)
age_inf = dense_64_layers[0](age_inf)
age_inf_out = age_output_layer(age_inf)

# Gender branch : Dense(128) -> Dropout -> Dense(1)
gender_inf = dense_128_layers[1](x_inf)
gender_inf = dropout_layers_all[1](gender_inf)
gender_inf_out = gender_output_layer(gender_inf)

# Ethnicity branch : Dense(128) -> Dropout -> Dense(5)
eth_inf = dense_128_layers[2](x_inf)
eth_inf = dropout_layers_all[2](eth_inf)
eth_inf_out = eth_output_layer(eth_inf)

inference_model = models.Model(
    inference_inputs,
    [age_inf_out, gender_inf_out, eth_inf_out],
    name='Transfer_Focal_Inference'
)

# Vérification
test_img = X_test[:1]
pred_original = model.predict(test_img, verbose=0)
pred_inference = inference_model.predict(test_img, verbose=0)
print(f"Vérification des poids (age) : diff max = {np.max(np.abs(pred_original[0] - pred_inference[0])):.6f}")
print(f"Vérification des poids (gender) : diff max = {np.max(np.abs(pred_original[1] - pred_inference[1])):.6f}")
print(f"Vérification des poids (ethnicity) : diff max = {np.max(np.abs(pred_original[2] - pred_inference[2])):.6f}")

tflite_path = os.path.join(OUTPUT_PATH, 'multitask_transfer_focal.tflite')

converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
print(f"Modèle TFLite sauvegardé : {tflite_path} ({size_mb:.1f} MB)")

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
    print(f"  Test prediction output {i} : {test_output.flatten()}")
print("  TFLite OK !")

# %% [code]
"""## 16. Résumé final"""

print("=" * 60)
print("RÉSUMÉ FINAL - MODÈLE MULTI-TÂCHE (AGE + GENRE + ETHNICITÉ)")
print("=" * 60)
print(f"""
Architecture :
  - Base model : MobileNetV2 (ImageNet)
  - Shared : GAP -> BN -> Dense(256) -> Dropout(0.4)
  - Age branch : Dense(128) -> Dense(64) -> Dense(1, linear)
  - Gender branch : Dense(128) -> Dropout(0.3) -> Dense(1, sigmoid)
  - Ethnicity branch : Dense(128) -> Dropout(0.4) -> Dense(5, softmax)
  - Input : RGB {IMG_SIZE}x{IMG_SIZE}
  - Paramètres totaux : {model.count_params():,}

Losses :
  - Age : Huber (delta=8.0), poids=0.4
  - Gender : Binary Crossentropy, poids=1.0
  - Ethnicity : Focal Loss (gamma=2.0, alpha pondéré), poids=1.0

  Poids alpha (ethnicité) :""")
for i, label in enumerate(eth_labels):
    print(f"    {label} : {alpha_weights[i]:.3f}")

print(f"""
Entraînement :
  - Phase 1 : Head seul (lr=0.001) - {phase1_epochs} epochs
  - Phase 2 : Fine-tuning 30 dernières couches (lr=0.0001) - {len(history2.history['loss'])} epochs
  - Augmentation : RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast

Résultats :
  - Age MAE : {age_mae:.2f} ans
  - Gender accuracy : {gender_acc*100:.2f}%
  - Gender AUC : {gender_auc_score:.4f}
  - Ethnicity accuracy : {eth_acc*100:.2f}%
  - Ethnicity AUC (macro) : {eth_auc_score:.4f}
""")

print("Performances par classe (ethnicité) :")
for i, label in enumerate(eth_labels):
    print(f"  {label:10s} - Precision: {precision[i]*100:5.1f}% | Recall: {recall[i]*100:5.1f}% | F1: {f1[i]*100:5.1f}% | Support: {support[i]}")

print(f"\n{'=' * 60}")
print("FICHIERS SAUVEGARDÉS")
print("=" * 60)
print("  - multitask_model_transfer_focal.keras")
print("  - multitask_transfer_focal.tflite")
print("  - training_curves_transfer_focal.png")
print("  - confusion_matrix_transfer_focal.png")
print("  - metrics_per_class_transfer_focal.png")
print("  - gradcam_transfer_focal.png")
print("  - confidence_distribution_transfer_focal.png")
