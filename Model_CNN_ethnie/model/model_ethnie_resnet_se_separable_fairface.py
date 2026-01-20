# %% [code]
# -*- coding: utf-8 -*-
"""
# Modèle d'ethnicité UTKFace + FairFace - Architecture de base + ResNet + SE-Net + Separable Conv

**Entraînement sur Kaggle**

**Datasets combinés :**
- UTKFace (jangedoo/utkface-new)
- FairFace (aibloy/fairface)

**5 classes d'ethnicité :**
- European (White)
- African (Black)
- East Asian (Asian)
- South Asian (Indian)
- Other (Latino + Middle Eastern)

**Architecture de base (3 blocs) améliorée avec :**

1. **ResNet (Skip Connections)** :
   - Connexions résiduelles autour de chaque bloc conv
   - Meilleur flux de gradient

2. **SE-Net (Squeeze-and-Excitation)** :
   - Attention sur les channels après chaque convolution
   - Le modèle apprend quels filtres sont importants

3. **Separable Convolutions** :
   - Remplace Conv2D par SeparableConv2D
   - ~8x moins de paramètres par couche

**Structure identique au modèle de base :**
- Bloc 1 : 32 filtres
- Bloc 2 : 64 filtres
- Bloc 3 : 128 filtres
- Dense : 256 → 5

**Preprocessing :**
- Images en niveaux de gris (1 canal)
- Redimensionnement 128x128
- Normalisation /255
- Class weights pour équilibrage

## 1. Configuration et imports
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

# Chemins Kaggle
KAGGLE_INPUT_UTK = "/kaggle/input/utkface-new"
KAGGLE_INPUT_FAIRFACE = "/kaggle/input/fairface"
OUTPUT_PATH = "/kaggle/working"

# Labels d'ethnicité (5 classes)
ETHNICITY_LABELS = [
    'European',      # 0 - White
    'African',       # 1 - Black
    'East Asian',    # 2 - Asian
    'South Asian',   # 3 - Indian
    'Other'          # 4 - Other (inclut Latino + Middle Eastern)
]

NUM_CLASSES = len(ETHNICITY_LABELS)
print(f"Nombre de classes : {NUM_CLASSES}")
print(f"Labels : {ETHNICITY_LABELS}")

"""## 2. Chargement des données UTKFace"""

# Mapping UTKFace vers nos 5 classes
UTK_TO_NEW_MAPPING = {
    0: 0,  # White -> European
    1: 1,  # Black -> African
    2: 2,  # Asian -> East Asian
    3: 3,  # Indian -> South Asian
    4: 4,  # Other -> Other
}

# Trouver le dossier UTKFace
possible_folders = ["UTKFace", "utkface_aligned_cropped", "crop_part1", ""]
utk_folder = None

for folder in possible_folders:
    test_path = os.path.join(KAGGLE_INPUT_UTK, folder) if folder else KAGGLE_INPUT_UTK
    if os.path.exists(test_path):
        files = os.listdir(test_path)
        jpg_files = [f for f in files if f.endswith(".jpg")]
        if jpg_files:
            utk_folder = test_path
            print(f"Dossier UTKFace trouvé : {utk_folder}")
            break

images_utk = []
labels_utk = []

if utk_folder:
    image_files = [f for f in os.listdir(utk_folder) if f.endswith(".jpg")]
    print(f"Fichiers UTKFace trouvés : {len(image_files)}")

    for file in image_files:
        try:
            parts = file.split("_")
            age = int(parts[0])
            gender = int(parts[1])
            race = int(parts[2])

            if race in UTK_TO_NEW_MAPPING:
                img = Image.open(os.path.join(utk_folder, file)).convert("L").resize((128, 128))
                images_utk.append(np.array(img))
                labels_utk.append(UTK_TO_NEW_MAPPING[race])
        except:
            continue

    print(f"Images UTKFace chargées : {len(images_utk)}")
else:
    print("Dossier UTKFace non trouvé")

"""## 3. Chargement des données FairFace (aibloy/fairface)"""

# Mapping FairFace vers nos 5 classes
FAIRFACE_TO_NEW_MAPPING = {
    'White': 0,                # -> European
    'Black': 1,                # -> African
    'East Asian': 2,           # -> East Asian
    'Southeast Asian': 2,      # -> East Asian (combiné)
    'Indian': 3,               # -> South Asian
    'Latino_Hispanic': 4,      # -> Other
    'Middle Eastern': 4        # -> Other
}

images_fairface = []
labels_fairface = []

# Structure du dataset aibloy/fairface : /kaggle/input/fairface/FairFace/
FAIRFACE_ROOT = os.path.join(KAGGLE_INPUT_FAIRFACE, "FairFace")

# Fichiers CSV
fairface_train_csv = os.path.join(FAIRFACE_ROOT, "train_labels.csv")
fairface_val_csv = os.path.join(FAIRFACE_ROOT, "val_labels.csv")

# Chemins des images
fairface_train_path = os.path.join(FAIRFACE_ROOT, "train")
fairface_val_path = os.path.join(FAIRFACE_ROOT, "val")


def load_fairface_data(csv_path, base_path):
    """Charge les données FairFace depuis un fichier CSV"""
    images = []
    labels = []

    if not os.path.exists(csv_path):
        print(f"Fichier non trouvé : {csv_path}")
        return images, labels

    df = pd.read_csv(csv_path)
    print(f"Entrées dans {os.path.basename(csv_path)} : {len(df)}")
    print(f"Colonnes : {list(df.columns)}")

    for _, row in df.iterrows():
        try:
            race = row['race']
            if race in FAIRFACE_TO_NEW_MAPPING:
                file_name = row['file']

                # Essayer plusieurs chemins possibles
                possible_paths = [
                    os.path.join(base_path, file_name),
                    os.path.join(base_path, os.path.basename(file_name)),
                    os.path.join(FAIRFACE_ROOT, file_name),
                ]

                img_path = None
                for p in possible_paths:
                    if os.path.exists(p):
                        img_path = p
                        break

                if img_path:
                    img = Image.open(img_path).convert("L").resize((128, 128))
                    images.append(np.array(img))
                    labels.append(FAIRFACE_TO_NEW_MAPPING[race])
        except:
            continue

    return images, labels


# Charger train et val de FairFace
imgs_train, lbls_train = load_fairface_data(fairface_train_csv, fairface_train_path)
imgs_val, lbls_val = load_fairface_data(fairface_val_csv, fairface_val_path)

images_fairface = imgs_train + imgs_val
labels_fairface = lbls_train + lbls_val

print(f"Images FairFace chargées : {len(images_fairface)}")

"""## 4. Fusion des datasets"""

# Combiner les deux datasets
all_images = images_utk + images_fairface
all_labels = labels_utk + labels_fairface

images = np.array(all_images)
labels = np.array(all_labels)

print(f"\n{'='*50}")
print("DATASET COMBINÉ")
print(f"{'='*50}")
print(f"Total images : {len(images)}")
print(f"Shape des images : {images.shape}")

# Distribution des classes
print(f"\nDistribution par classe :")
for i, label in enumerate(ETHNICITY_LABELS):
    count = np.sum(labels == i)
    pct = count / len(labels) * 100
    print(f"  {label:15s} : {count:6d} ({pct:5.1f}%)")

"""## 5. Imports TensorFlow et préparation des données"""

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score
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

print(f"\nTensorFlow version : {tf.__version__}")
print(f"GPU disponible : {tf.config.list_physical_devices('GPU')}")

X = images
y_ethnicity = labels

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
print(f"\nClass weights : {class_weight_dict}")

y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)

print(f"\nX_train : {X_train.shape}")
print(f"X_test : {X_test.shape}")

"""## 6. Définition des blocs (SE-Block)"""

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


"""## 7. Construction du modèle (Architecture de base + 3 techniques)"""

def build_model(input_shape=(128, 128, 1), num_classes=5):
    """
    Architecture de base avec ResNet + SE-Net + Separable Conv

    Structure :
    - Bloc 1 : 32 filtres
    - Bloc 2 : 64 filtres
    - Bloc 3 : 128 filtres
    - Dense : 256 → 5

    Modifications :
    - Conv2D → SeparableConv2D (sauf première couche)
    - Ajout SE-Block après chaque conv
    - Ajout Skip Connection autour de chaque bloc
    """
    inputs = Input(shape=input_shape)

    # ================================================================
    # BLOC 1 : 32 filtres (comme le modèle de base)
    # ================================================================
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = se_block(x)

    shortcut = Conv2D(32, (1, 1), padding='same')(inputs)
    x = Add()([x, shortcut])

    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # ================================================================
    # BLOC 2 : 64 filtres
    # ================================================================
    shortcut = x

    x = SeparableConv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = se_block(x)

    shortcut = Conv2D(64, (1, 1), padding='same')(shortcut)
    x = Add()([x, shortcut])

    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # ================================================================
    # BLOC 3 : 128 filtres
    # ================================================================
    shortcut = x

    x = SeparableConv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = se_block(x)

    shortcut = Conv2D(128, (1, 1), padding='same')(shortcut)
    x = Add()([x, shortcut])

    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # ================================================================
    # CLASSIFICATION (256 → 5 classes)
    # ================================================================
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='Base_ResNet_SE_Separable_FairFace')
    return model


# Créer le modèle
model = build_model(input_shape=(128, 128, 1), num_classes=NUM_CLASSES)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Informations sur le modèle
print(f"\n{'='*60}")
print("MODÈLE UTKFace + FairFace - 5 CLASSES D'ETHNICITÉ")
print(f"{'='*60}")
print(f"""
Datasets combinés :
  - UTKFace (toutes les classes)
  - FairFace (toutes les classes)

Classes d'ethnicité (5) :
  0. European      (UTK White + FairFace White)
  1. African       (UTK Black + FairFace Black)
  2. East Asian    (UTK Asian + FairFace East/Southeast Asian)
  3. South Asian   (UTK Indian + FairFace Indian)
  4. Other         (UTK Other + FairFace Latino + Middle Eastern)

Architecture : Base + ResNet + SE-Net + Separable Conv
Paramètres : {model.count_params():,}
""")

"""## 8. Entraînement"""

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

"""## 9. Visualisation de l'entraînement"""

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
plt.savefig(os.path.join(OUTPUT_PATH, 'training_curves_base_fairface.png'), dpi=150)
plt.show()

print("=" * 50)
print("RÉSUMÉ DE L'ENTRAÎNEMENT")
print("=" * 50)
print(f"Nombre d'epochs effectuées : {len(history.history['loss'])}")
print(f"\nMeilleure accuracy validation : {max(history.history['val_accuracy'])*100:.2f}%")
print(f"Meilleure loss validation : {min(history.history['val_loss']):.4f}")
print(f"\nAccuracy finale (train) : {history.history['accuracy'][-1]*100:.2f}%")
print(f"Accuracy finale (val) : {history.history['val_accuracy'][-1]*100:.2f}%")

"""## 10. Évaluation du modèle"""

y_pred_proba = model.predict(X_test)
y_pred = y_pred_proba.argmax(axis=1)

loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"\nAccuracy sur le test set : {accuracy*100:.2f}%")

# AUC et AP (métriques multi-classes)
auc_score = roc_auc_score(y_test_cat, y_pred_proba, multi_class='ovr', average='macro')
ap_score = average_precision_score(y_test_cat, y_pred_proba, average='macro')
print(f"AUC (macro) : {auc_score:.4f}")
print(f"AP (macro) : {ap_score:.4f}")

print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=ETHNICITY_LABELS))

"""## 11. Matrice de confusion"""

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=ETHNICITY_LABELS,
    yticklabels=ETHNICITY_LABELS
)
plt.title('Matrice de confusion - UTKFace + FairFace (5 classes)')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix_base_fairface.png'), dpi=150)
plt.show()

"""## 12. Sauvegarde du modèle"""

model.save(os.path.join(OUTPUT_PATH, 'ethnicity_model_base_fairface.keras'))
print(f"Modèle sauvegardé : {OUTPUT_PATH}/ethnicity_model_base_fairface.keras")

print("\n→ Les fichiers sont disponibles dans l'onglet 'Output' de Kaggle pour téléchargement.")

"""## 13. Analyse des performances par classe"""

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(ETHNICITY_LABELS))
width = 0.25

bars1 = ax.bar(x - width, precision * 100, width, label='Precision', color='steelblue')
bars2 = ax.bar(x, recall * 100, width, label='Recall', color='teal')
bars3 = ax.bar(x + width, f1 * 100, width, label='F1-Score', color='coral')

ax.set_ylabel('Score (%)')
ax.set_title('Performances par classe - UTKFace + FairFace (5 classes)')
ax.set_xticks(x)
ax.set_xticklabels(ETHNICITY_LABELS, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'metrics_per_class_base_fairface.png'), dpi=150)
plt.show()

# Résumé final
print("=" * 60)
print("RÉSUMÉ FINAL - UTKFace + FairFace (5 CLASSES)")
print("=" * 60)
print(f"\nDatasets :")
print(f"  - UTKFace : {len(images_utk)} images")
print(f"  - FairFace : {len(images_fairface)} images")
print(f"  - Total : {len(images)} images")
print(f"\nArchitecture :")
print(f"  - 3 blocs convolutionnels")
print(f"  - Filtres : 32 → 64 → 128")
print(f"  - Dense : 256 → 5")
print(f"  - Paramètres : {model.count_params():,}")
print(f"\nTechniques :")
print(f"  - ResNet : Skip connections")
print(f"  - SE-Net : Channel attention")
print(f"  - Separable Conv : Réduction paramètres")
print(f"\nAccuracy globale : {accuracy*100:.2f}%")
print(f"\nPerformances par classe :")
for i, label in enumerate(ETHNICITY_LABELS):
    print(f"  {label:15s} - Precision: {precision[i]*100:5.1f}% | Recall: {recall[i]*100:5.1f}% | F1: {f1[i]*100:5.1f}% | Support: {support[i]}")

print(f"\n" + "=" * 60)
print("FICHIERS SAUVEGARDÉS")
print("=" * 60)
print("  - ethnicity_model_base_fairface.keras")
print("  - training_curves_base_fairface.png")
print("  - confusion_matrix_base_fairface.png")
print("  - metrics_per_class_base_fairface.png")
