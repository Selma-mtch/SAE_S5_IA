# %% [code]
# -*- coding: utf-8 -*-
"""
# Modèle d'ethnicité UTKFace + FairFace - Architecture Complexe + ResNet + SE-Net + Separable Conv

**Entraînement sur Kaggle**

**Datasets combinés :**
- UTKFace (jangedoo/utkface-new)
- FairFace (dataset équilibré par ethnie)

**5 classes d'ethnicité :**
- European (White)
- African (Black)
- East Asian (Asian)
- South Asian (Indian)
- Other (Latino + Middle Eastern)

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
# UTKFace : 0=White, 1=Black, 2=Asian, 3=Indian, 4=Other
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

            # Mapper vers nouvelles classes (toutes les classes incluses)
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

# Fichiers CSV dans FairFace/
# Noms réels : train_labels.csv et val_labels.csv
possible_train_csv = [
    os.path.join(FAIRFACE_ROOT, "train_labels.csv"),
]
possible_val_csv = [
    os.path.join(FAIRFACE_ROOT, "val_labels.csv"),
]

fairface_train_csv = None
fairface_val_csv = None

for path in possible_train_csv:
    if os.path.exists(path):
        fairface_train_csv = path
        print(f"CSV train trouvé : {path}")
        break

for path in possible_val_csv:
    if os.path.exists(path):
        fairface_val_csv = path
        print(f"CSV val trouvé : {path}")
        break

if fairface_train_csv is None:
    print(f"[!] fairface_label_train.csv non trouvé")
    print(f"\n--- Exploration de la structure FairFace ---")
    print(f"Chemin de base : {KAGGLE_INPUT_FAIRFACE}")

    # Explorer et afficher toute la structure
    for root, dirs, files in os.walk(KAGGLE_INPUT_FAIRFACE):
        level = root.replace(KAGGLE_INPUT_FAIRFACE, '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")

        # Afficher les fichiers CSV
        csv_files = [f for f in files if f.endswith('.csv')]
        for f in csv_files:
            print(f"{indent}  [CSV] {f}")

        # Afficher le nombre d'images
        img_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
        if img_files:
            print(f"{indent}  [{len(img_files)} images]")

        # Limiter la profondeur
        if level >= 3:
            dirs[:] = []

    print(f"--- Fin exploration ---\n")


def load_fairface_data(csv_path, base_path):
    """Charge les données FairFace depuis un fichier CSV"""
    images = []
    labels = []

    if csv_path is None or not os.path.exists(csv_path):
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
                    os.path.join(base_path, file_name),                      # FairFace/train/file.jpg
                    os.path.join(base_path, os.path.basename(file_name)),   # FairFace/train/file.jpg (sans sous-dossier)
                    os.path.join(FAIRFACE_ROOT, file_name),                 # FairFace/train/file.jpg
                    os.path.join(KAGGLE_INPUT_FAIRFACE, file_name),         # fairface/train/file.jpg
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


# Chemins des images (structure : FairFace/train/ et FairFace/val/)
fairface_train_path = os.path.join(FAIRFACE_ROOT, "train")
fairface_val_path = os.path.join(FAIRFACE_ROOT, "val")

# Charger train et val de FairFace
if fairface_train_csv:
    imgs_train, lbls_train = load_fairface_data(fairface_train_csv, fairface_train_path)
else:
    imgs_train, lbls_train = [], []

if fairface_val_csv:
    imgs_val, lbls_val = load_fairface_data(fairface_val_csv, fairface_val_path)
else:
    imgs_val, lbls_val = [], []

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
print(f"y_train_cat : {y_train_cat.shape}")
print(f"y_test_cat : {y_test_cat.shape}")

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


"""## 7. Construction du modèle (Architecture Complexe + 3 techniques)"""


def build_model(input_shape=(128, 128, 1), num_classes=5):
    """
    Architecture Complexe (VGG-style) avec ResNet + SE-Net + Separable Conv

    Structure :
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
    # BLOC 4 : 512 filtres (double conv)
    # ================================================================
    shortcut = x

    x = conv_block_with_se(x, 512, use_separable=True)  # +Separable

    # Skip connection : adapter 256 → 512 channels
    shortcut = Conv2D(512, (1, 1), padding='same')(shortcut)
    x = Add()([x, shortcut])  # +ResNet

    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # ================================================================
    # CLASSIFICATION (512 → 256 → 5 classes)
    # ================================================================
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='Complexe_ResNet_SE_Separable_FairFace')
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

Architecture : VGG-style + ResNet + SE-Net + Separable Conv
Paramètres : {model.count_params():,}
""")

"""## 8. Entraînement"""

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
plt.savefig(os.path.join(OUTPUT_PATH, 'training_curves_complexe_fairface.png'), dpi=150)
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
plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix_complexe_fairface.png'), dpi=150)
plt.show()

"""## 12. Sauvegarde du modèle"""

model.save(os.path.join(OUTPUT_PATH, 'ethnicity_model_complexe_fairface.keras'))
print(f"Modèle sauvegardé : {OUTPUT_PATH}/ethnicity_model_complexe_fairface.keras")

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
plt.savefig(os.path.join(OUTPUT_PATH, 'metrics_per_class_complexe_fairface.png'), dpi=150)
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
print(f"  - 4 blocs convolutionnels (double conv style VGG)")
print(f"  - Filtres : 64 → 128 → 256 → 512")
print(f"  - Dense : 512 → 256 → 5")
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
print("  - ethnicity_model_complexe_fairface.keras")
print("  - training_curves_complexe_fairface.png")
print("  - confusion_matrix_complexe_fairface.png")
print("  - metrics_per_class_complexe_fairface.png")
