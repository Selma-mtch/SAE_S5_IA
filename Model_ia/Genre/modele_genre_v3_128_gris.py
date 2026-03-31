# ============================================================
# modele_genre_v3_128_gris.py
# Prediction du sexe a partir d'images UTKFace
# Resolution : 128x128, niveaux de gris (1 canal)
# Genere depuis : modele_genre_v3_128_gris.ipynb
# ============================================================

# # Notebook final - Predire le sexe a partir de UTKFace (128x128, niveaux de gris)
#
# ## 1. Contexte et objectif
#
# L'objectif de ce notebook est de construire un modele final de classification binaire du sexe
# a partir du dataset UTKFace.
#
# Le pipeline est volontairement centre sur les outils explicitement presents dans les cours :
# CNN Keras, convolutions, pooling, flatten, couches denses, regularisation, dropout,
# optimisation avec Adam, suivi de l'accuracy, courbes d'apprentissage, matrice de confusion
# et metriques de classification.
#
# Le notebook est pense pour etre execute tel quel sur Kaggle avec GPU.

import gc
from pathlib import Path

import kagglehub
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, callbacks

# ## 2. Contraintes pedagogiques
#
# Ce notebook respecte les regles suivantes :
#
# - uniquement des techniques explicitement citees dans les cours fournis ;
# - TensorFlow / Keras pour le modele ;
# - aucune methode "hors programme" ajoutee pour gagner artificiellement en performance ;
# - priorite donnee a la conformite pedagogique avant toute sophistication.
#
# Choix importants de conformite (version corrigee) :
#
# [CORRECTION 1] sortie finale en 1 neurone + sigmoid (Cours 3 : "Sigmoide si binaire") ;
# [CORRECTION 1] fonction de perte binary_crossentropy (Cours 3 : classification binaire) ;
# [CORRECTION 2] standardisation par mean/std du train (Cours 4 : StandardScaler) ;
# [# [CORRECTION 3] Data augmentation via RandomFlip et RandomRotation uniquement
#                (Cours 4 : "Data augmentation surtout pour images")
#                RandomBrightness RETIRE : inutile pour le genre (signal geometrique,
#                pas photometrique) et nocif avec BN (degrade les running stats)
# [CORRECTION 4] BatchNormalization ajoutee dans les couches cachees
#                (Cours 4 : "Batch Normalization : normalise l'activation, accelere l'apprentissage") ;
# - Adam comme optimiseur ;
# - accuracy pendant l'entrainement ;
# - regularisation L2, Dropout, Early stopping et ReduceLROnPlateau car ces
#   techniques apparaissent explicitement dans le cours ;
# - pas de transfert learning.

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_DIR = None
DATASET_HANDLE = "jangedoo/utkface-new"
OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

IMAGE_SIZE = 128
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 32  # Reduit pour gerer la memoire avec resolution 128
EPOCHS = 25
LEARNING_RATE = 1e-3
L2_FACTOR = 1e-4
DROPOUT_RATE = 0.40

EARLY_STOPPING_PATIENCE = 5
LR_PATIENCE = 2
LR_REDUCTION_FACTOR = 0.5

N_SAMPLE_IMAGES = 12
LABEL_NAMES = {0: "Homme", 1: "Femme"}

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

gpu_devices = tf.config.list_physical_devices("GPU")
print(f"TensorFlow : {tf.__version__}")
print(f"GPU detecte(s) : {gpu_devices}")
print(f"Dossier de sortie : {OUTPUT_DIR.resolve()}")

# ## 3. Imports
#
# Les bibliotheques utilisees restent simples :
#
# - numpy pour la manipulation des tableaux ;
# - PIL pour lire les images ;
# - matplotlib pour les visualisations ;
# - tensorflow.keras pour le modele CNN et l'entrainement ;
# - kagglehub pour recuperer le dataset UTKFace sur Kaggle.


# ## 4. Parametres globaux
#
# Choix retenus :
#
# - IMAGE_SIZE = 128 : resolution augmentee (images en niveaux de gris) ;
# - BATCH_SIZE = 64 : valeur coherente avec les exemples CNN du cours ;
# - Adam avec learning_rate = 1e-3 : choix robuste et explicitement vu en cours ;
# - L2 et Dropout : regularisation pour limiter le surapprentissage ;
# - EarlyStopping et ReduceLROnPlateau : optimisations explicitement citees dans le cours.

def parse_utkface_filename(filename):
    """Extrait age, sexe et race a partir du nom de fichier UTKFace."""
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 4:
        return None

    try:
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])
    except ValueError:
        return None

    if gender not in (0, 1):
        return None

    return {
        "filename": filename,
        "age": age,
        "gender": gender,
        "race": race,
    }


def find_utkface_directory(dataset_dir=None, dataset_handle="jangedoo/utkface-new"):
    """Charge UTKFace via kagglehub ou utilise un chemin fourni manuellement."""
    if dataset_dir is not None:
        dataset_path = Path(dataset_dir)
    else:
        downloaded_path = Path(kagglehub.dataset_download(dataset_handle))
        dataset_path = downloaded_path

    candidate_dirs = [
        dataset_path / "UTKFace",
        dataset_path / "crop_part1",
        dataset_path,
    ]

    for candidate in candidate_dirs:
        if not candidate.exists():
            continue

        image_files = [
            path for path in candidate.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not image_files:
            continue

        valid_examples = sum(
            parse_utkface_filename(path.name) is not None
            for path in image_files[:20]
        )

        if valid_examples >= 10:
            return candidate

    raise FileNotFoundError(
        "Impossible de trouver un dossier d'images UTKFace valide apres le telechargement."
    )


def load_images_and_labels(records, image_dir, image_size):
    """Charge toutes les images en niveaux de gris, les redimensionne et recupere les labels de sexe."""
    # MODIFICATION : conversion en niveaux de gris -> 1 canal au lieu de 3
    images = np.empty((len(records), image_size, image_size, 1), dtype=np.uint8)
    labels = np.empty(len(records), dtype=np.int32)

    for index, record in enumerate(records):
        image_path = image_dir / record["filename"]

        with Image.open(image_path) as image:
            # MODIFICATION : "L" = niveaux de gris (Luminance), 1 seul canal
            image = image.convert("L").resize((image_size, image_size))
            images[index] = np.asarray(image, dtype=np.uint8)[..., np.newaxis]

        labels[index] = record["gender"]

        if (index + 1) % 5000 == 0 or index == len(records) - 1:
            print(f"Images chargees : {index + 1}/{len(records)}")

    return images, labels


def stratified_split_indices(labels, train_ratio, val_ratio, test_ratio, seed=42):
    """Construit un split stratifie simple sans dependance externe."""
    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []
    test_indices = []

    for class_id in np.unique(labels):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)

        n_samples = len(class_indices)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train:n_train + n_val])
        test_indices.extend(class_indices[n_train + n_val:])

    train_indices = np.array(train_indices, dtype=np.int32)
    val_indices = np.array(val_indices, dtype=np.int32)
    test_indices = np.array(test_indices, dtype=np.int32)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return train_indices, val_indices, test_indices


def summarize_split(name, labels):
    """Affiche la repartition des classes dans un sous-ensemble."""
    counts = np.bincount(labels, minlength=2)
    total = counts.sum()

    print(f"{name} : {total} images")
    for class_id, count in enumerate(counts):
        ratio = count / total if total else 0.0
        print(f"  - {LABEL_NAMES[class_id]} ({class_id}) : {count} ({ratio:.2%})")


# [CORRECTION 3] Data augmentation integree directement dans le modele Keras.
# Avantage : elle est automatiquement desactivee pendant model.evaluate et model.predict.
# Justification cours 4 : "Data augmentation (surtout pour images) : genere plus de
# donnees pour ameliorer la robustesse."
# Choix delibere pour le GENRE : les caracteristiques importantes (forme de la machoire,
# des yeux) sont geometriques, pas liees a la luminosite.
# RandomBrightness est RETIRE car :
# 1. Inutile : le genre ne depend pas de l'eclairage.
# 2. Nocif avec BatchNormalization : change les statistiques de pixels par batch,
#    empeche le BN d'estimer des running_mean/var stables -> val_loss explose.
# RandomFlip et RandomRotation preservent la distribution de pixels, pas de conflit BN.
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),   # un visage a 40 ans retourne reste reconnaissable
    layers.RandomRotation(0.05),       # legere inclinaison de tete (+-5%)
], name="data_augmentation")


def build_gender_cnn(input_shape):
    """
    Construit le CNN final avec uniquement des briques autorisees par le cours.

    MODIFICATION : Adapte pour des images en niveaux de gris (1 canal).
    La resolution est passee en 128x128.

    Corrections appliquees par rapport a la version initiale :

    [CORRECTION 1] Couche de sortie : Dense(1, sigmoid) + binary_crossentropy
                   au lieu de Dense(2, softmax) + sparse_categorical_crossentropy.
                   Justification : Cours 3, explicitement "Sigmoide si binaire".

    [CORRECTION 3] Data augmentation ajoutee en premiere couche du modele.
                   Justification : Cours 4, "Data augmentation surtout pour images".

    [CORRECTION 4] BatchNormalization ajoutee apres chaque bloc Conv + activation.
                   Justification : Cours 4, "Batch Normalization : normalise l'activation
                   de chaque couche, accelere l'apprentissage".
    """
    inputs = layers.Input(shape=input_shape)

    # [CORRECTION 3] Augmentation active seulement en training
    x = data_augmentation(inputs)

    # Bloc 1
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)   # [CORRECTION 4]
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloc 2
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)   # [CORRECTION 4]
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloc 3
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)   # [CORRECTION 4]
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloc 4
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)   # [CORRECTION 4]
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)

    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_FACTOR)
    )(x)
    x = layers.BatchNormalization()(x)   # [CORRECTION 4]
    x = layers.Dropout(DROPOUT_RATE)(x)

    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_FACTOR)
    )(x)

    # [CORRECTION 1] : 1 neurone + sigmoid pour classification BINAIRE
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def compute_metrics_from_confusion_matrix(conf_matrix):
    """Calcule accuracy, precision, recall et F1 a partir de la matrice de confusion."""
    total = conf_matrix.sum()
    accuracy = np.trace(conf_matrix) / total if total else 0.0

    per_class = []
    precisions = []
    recalls = []
    f1_scores = []

    for class_id in range(conf_matrix.shape[0]):
        tp = conf_matrix[class_id, class_id]
        fp = conf_matrix[:, class_id].sum() - tp
        fn = conf_matrix[class_id, :].sum() - tp
        support = conf_matrix[class_id, :].sum()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )

        per_class.append({
            "label": LABEL_NAMES[class_id],
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": int(support),
        })
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
    }

# ## 5. Chargement du dataset
#
# Le chargement utilise kagglehub :
#
# - telechargement du dataset jangedoo/utkface-new ;
# - recuperation du sous-dossier UTKFace ;
# - verification que les noms de fichiers suivent bien le format attendu.

image_dir = find_utkface_directory(DATASET_DIR, DATASET_HANDLE)
all_image_paths = sorted([
    path for path in image_dir.iterdir()
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
])

print(f"Dossier d'images detecte : {image_dir}")
print(f"Nombre de fichiers image trouves : {len(all_image_paths)}")

# ## 6. Extraction des labels de sexe
#
# UTKFace encode les informations dans le nom du fichier :
# age_gender_race_date.jpg
# Ici, seule la variable gender est retenue comme cible de classification.

records = []
ignored_files = 0

for path in all_image_paths:
    parsed = parse_utkface_filename(path.name)
    if parsed is None:
        ignored_files += 1
        continue
    records.append(parsed)

if not records:
    raise ValueError("Aucune image UTKFace valide n'a ete trouvee.")

ages = np.array([record["age"] for record in records], dtype=np.int32)
genders = np.array([record["gender"] for record in records], dtype=np.int32)
races = np.array([record["race"] for record in records], dtype=np.int32)

print(f"Images valides conservees : {len(records)}")
print(f"Images ignorees : {ignored_files}")
print(f"Repartition du sexe : {np.bincount(genders, minlength=2)}")

# ## 7. Verifications sur les donnees

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.hist(ages, bins=range(0, int(ages.max()) + 5, 5), color="steelblue", edgecolor="black")
plt.title("Distribution des ages")
plt.xlabel("Age")
plt.ylabel("Nombre d'images")
plt.grid(alpha=0.25)

plt.subplot(1, 3, 2)
gender_counts = np.bincount(genders, minlength=2)
plt.bar([LABEL_NAMES[0], LABEL_NAMES[1]], gender_counts, color=["royalblue", "salmon"], edgecolor="black")
plt.title("Distribution du sexe")
plt.ylabel("Nombre d'images")
plt.grid(axis="y", alpha=0.25)

plt.subplot(1, 3, 3)
race_counts = np.bincount(races)
plt.bar(range(len(race_counts)), race_counts, color="slategray", edgecolor="black")
plt.title("Distribution de la race")
plt.xlabel("Code race")
plt.ylabel("Nombre d'images")
plt.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.show()

# ## 8. Chargement des images brutes (uint8)
#
# Les images sont chargees en niveaux de gris (uint8, 0-255) avant normalisation.
# MODIFICATION : chaque image est convertie en niveaux de gris (1 canal au lieu de 3).
# La normalisation sera appliquee apres le split, uniquement sur la base du train.

images_uint8, labels = load_images_and_labels(records, image_dir, IMAGE_SIZE)

print(f"Shape des images apres chargement : {images_uint8.shape}  # (N, 128, 128, 1) attendu")
print(f"Shape des labels : {labels.shape}")

rng = np.random.default_rng(SEED)
sample_count = min(N_SAMPLE_IMAGES, len(images_uint8))
sample_indices = rng.choice(len(images_uint8), size=sample_count, replace=False)

plt.figure(figsize=(14, 8))
for plot_index, image_index in enumerate(sample_indices, start=1):
    plt.subplot(3, 4, plot_index)
    plt.imshow(images_uint8[image_index, :, :, 0], cmap="gray")  # MODIFICATION: 1 canal -> squeeze + colormap gris
    plt.title(
        f"{LABEL_NAMES[int(labels[image_index])]} | {ages[image_index]} ans | race {races[image_index]}",
        fontsize=9
    )
    plt.axis("off")

plt.suptitle("Exemples d'images UTKFace apres redimensionnement", fontsize=13)
plt.tight_layout()
plt.show()

# ## 9. Split train / validation / test
#
# Le split est realise en conservant l'equilibre des deux classes dans chaque sous-ensemble.

train_idx, val_idx, test_idx = stratified_split_indices(
    labels,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO,
    seed=SEED
)

# Conversion float32 pour la normalisation
x_train_raw = images_uint8[train_idx].astype("float32")
x_val_raw   = images_uint8[val_idx].astype("float32")
x_test_raw  = images_uint8[test_idx].astype("float32")

y_train = labels[train_idx]
y_val   = labels[val_idx]
y_test  = labels[test_idx]

del images_uint8
gc.collect()

# ## 10. Normalisation par standardisation (mean / std)
#
# [CORRECTION 2] Remplacement de la normalisation /255 par une standardisation.
#
# Justification : le Cours 4 (California Housing) utilise explicitement StandardScaler
# et insiste que mean et std sont calcules SUR LE TRAIN UNIQUEMENT, puis appliques
# a val et test. Cela garantit qu'aucune information du val/test ne fuit dans le
# preprocessing (data leakage).
#
# La standardisation centre les pixels autour de 0 avec ecart-type 1, ce qui
# accelere la convergence d'Adam et est plus robuste aux variations de luminosite
# presentes dans UTKFace.

# Calcul de la moyenne et l'ecart-type UNIQUEMENT sur le train
train_mean = x_train_raw.mean()
train_std  = x_train_raw.std()

print(f"Moyenne calculee sur le train : {train_mean:.4f}")
print(f"Ecart-type calcule sur le train : {train_std:.4f}")

# Application de la meme standardisation sur train, val et test
x_train = (x_train_raw - train_mean) / train_std
x_val   = (x_val_raw   - train_mean) / train_std
x_test  = (x_test_raw  - train_mean) / train_std

del x_train_raw, x_val_raw, x_test_raw
gc.collect()

print(f"\nApres standardisation :")
print(f"x_train  mean={x_train.mean():.4f}  std={x_train.std():.4f}  shape={x_train.shape}")
print(f"x_val    mean={x_val.mean():.4f}  std={x_val.std():.4f}  shape={x_val.shape}")
print(f"x_test   mean={x_test.mean():.4f}  std={x_test.std():.4f}  shape={x_test.shape}")

summarize_split("Train", y_train)
summarize_split("Validation", y_val)
summarize_split("Test", y_test)

# ## 11. Construction du modele
#
# Le modele integre les 4 corrections par rapport a la version initiale :
#
# [CORRECTION 1] Sortie sigmoid + binary_crossentropy (classification binaire).
# [CORRECTION 3] Data augmentation en premiere couche (RandomFlip, RandomRotation).
# [CORRECTION 4] BatchNormalization apres chaque bloc convolutif et apres la premiere Dense.

# MODIFICATION : 1 canal (niveaux de gris) au lieu de 3
model = build_gender_cnn((IMAGE_SIZE, IMAGE_SIZE, 1))
model.summary()

# ## 12. Compilation
#
# [CORRECTION 1] Passage a binary_crossentropy + sigmoid, conformement au Cours 3
# qui distingue explicitement :
# - "Sigmoide si binaire" + Binary Crossentropy
# - "Softmax si multiclasse" + Categorical Crossentropy
#
# Les metriques precision et recall sont ajoutees car citees dans le Cours 3
# dans la section metriques de classification.

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",          # [CORRECTION 1]
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ]
)

training_callbacks = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=LR_REDUCTION_FACTOR,
        patience=LR_PATIENCE,
        verbose=1
    ),
]

print("Callbacks retenus : EarlyStopping + ReduceLROnPlateau")

# ## 13. Entrainement
#
# L'entrainement est realise sur le jeu train avec suivi du jeu validation.
# La data augmentation (RandomFlip, RandomRotation) est active
# automatiquement pendant le training et desactivee pendant la validation.

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=training_callbacks,
    verbose=1
)

# ## 14. Evaluation
#
# L'evaluation finale se fait sur le jeu test, jamais utilise pendant l'entrainement.
# [CORRECTION 1] Les predictions sont obtenues par seuillage a 0.5 de la sortie sigmoid
# (au lieu de argmax sur softmax).

test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
    x_test, y_test, verbose=0
)

# [CORRECTION 1] : sigmoid produit une proba entre 0 et 1 -> seuil a 0.5
y_pred_proba = model.predict(x_test, verbose=0).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

conf_matrix = tf.math.confusion_matrix(y_test, y_pred, num_classes=2).numpy()
metrics_summary = compute_metrics_from_confusion_matrix(conf_matrix)

print(f"Loss sur le test      : {test_loss:.4f}")
print(f"Accuracy sur le test  : {test_accuracy:.4f}")
print(f"Precision sur le test : {test_precision:.4f}")
print(f"Recall sur le test    : {test_recall:.4f}\n")

for class_metrics in metrics_summary["per_class"]:
    print(
        f"{class_metrics['label']} -> "
        f"precision={class_metrics['precision']:.4f} | "
        f"recall={class_metrics['recall']:.4f} | "
        f"f1={class_metrics['f1_score']:.4f} | "
        f"support={class_metrics['support']}"
    )

print("\nMoyennes macro :")
print(f"Precision macro : {metrics_summary['macro_precision']:.4f}")
print(f"Recall macro    : {metrics_summary['macro_recall']:.4f}")
print(f"F1 macro        : {metrics_summary['macro_f1']:.4f}")

# ## 15. Visualisations des performances

history_dict = history.history
epochs_range = range(1, len(history_dict["loss"]) + 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history_dict["loss"], label="Train")
plt.plot(epochs_range, history_dict["val_loss"], label="Validation")
plt.title("Evolution de la loss (binary_crossentropy)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history_dict["accuracy"], label="Train")
plt.plot(epochs_range, history_dict["val_accuracy"], label="Validation")
plt.title("Evolution de l'accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
plt.imshow(conf_matrix, cmap="Blues")
plt.title("Matrice de confusion - test")
plt.xlabel("Prediction")
plt.ylabel("Verite")
plt.xticks([0, 1], [LABEL_NAMES[0], LABEL_NAMES[1]])
plt.yticks([0, 1], [LABEL_NAMES[0], LABEL_NAMES[1]])

for row in range(conf_matrix.shape[0]):
    for col in range(conf_matrix.shape[1]):
        plt.text(col, row, conf_matrix[row, col], ha="center", va="center", color="black")

plt.tight_layout()
plt.show()

prediction_count = min(N_SAMPLE_IMAGES, len(x_test))
prediction_indices = rng.choice(len(x_test), size=prediction_count, replace=False)

# On denormalise pour l'affichage (retour en [0,1] pour imshow)
x_test_display = x_test * train_std + train_mean
x_test_display = np.clip(x_test_display / 255.0, 0.0, 1.0)

plt.figure(figsize=(14, 8))
for plot_index, test_index in enumerate(prediction_indices, start=1):
    true_label = LABEL_NAMES[int(y_test[test_index])]
    pred_label = LABEL_NAMES[int(y_pred[test_index])]
    confidence = float(y_pred_proba[test_index]) if y_pred[test_index] == 1 else 1.0 - float(y_pred_proba[test_index])
    is_correct = int(y_test[test_index]) == int(y_pred[test_index])
    title_color = "green" if is_correct else "red"

    plt.subplot(3, 4, plot_index)
    # MODIFICATION : squeeze le canal unique (niveaux de gris) pour imshow
    plt.imshow(x_test_display[test_index, :, :, 0], cmap="gray")
    plt.title(
        f"Vrai: {true_label}\nPred: {pred_label} ({confidence:.0%})",
        fontsize=9,
        color=title_color
    )
    plt.axis("off")

plt.suptitle("Exemples de predictions sur le jeu de test", fontsize=13)
plt.tight_layout()
plt.show()

# ## 16. Analyse des resultats

best_epoch = int(np.argmax(history_dict["val_accuracy"]) + 1)
best_val_accuracy = float(np.max(history_dict["val_accuracy"]))

print(f"Meilleure epoch selon la validation : {best_epoch}")
print(f"Meilleure val_accuracy             : {best_val_accuracy:.4f}")
print(f"Accuracy finale sur le test        : {test_accuracy:.4f}")

if len(history_dict["accuracy"]) >= 2:
    final_gap = history_dict["accuracy"][-1] - history_dict["val_accuracy"][-1]
    print(f"Ecart train/validation en fin d'entrainement : {final_gap:.4f}")

print("\nLecture rapide :")
print("- Si les courbes train et validation restent proches, le surapprentissage est mieux controle.")
print("- Si precision, recall et F1 restent proches entre les deux classes, le modele est plus equilibre.")
print("- La matrice de confusion permet d'identifier d'eventuels biais vers une classe.")

# ## 17. Sauvegarde du meilleur modele

model_path = OUTPUT_DIR / "modele_sexe_utkface_128_gris.keras"
model.save(model_path)
print(f"Modele sauvegarde : {model_path}")

# Sauvegarde des parametres de standardisation pour une utilisation future
np.save(OUTPUT_DIR / "train_mean.npy", np.array([train_mean]))
np.save(OUTPUT_DIR / "train_std.npy",  np.array([train_std]))
print(f"Parametres de normalisation sauvegardes : mean={train_mean:.4f}, std={train_std:.4f}")

# ## 18. Verification finale de conformite (version corrigee)
#
# Toutes les techniques utilisees sont explicitement citees dans les cours :
#
# [CORRECTION 1] Dense(1, sigmoid) + binary_crossentropy
#   -> Cours 3 : "Sigmoide si binaire" + Binary Crossentropy
#
# [CORRECTION 2] Standardisation mean/std calculee sur le train uniquement
#   -> Cours 4 : StandardScaler avec fit_transform sur train, transform sur val/test
#
# [CORRECTION 3] Data augmentation (RandomFlip, RandomRotation) - RandomBrightness retire
#   -> Cours 4 : "Data augmentation (surtout pour images) : genere plus de donnees
#   -> RandomBrightness retire : inutile pour le genre + conflit avec BatchNormalization
#      pour ameliorer la robustesse"
#
# [CORRECTION 4] BatchNormalization apres chaque bloc Conv et Dense
#   -> Cours 4 : "Batch Normalization : normalise l'activation de chaque couche,
#      accelere l'apprentissage"
#
# Reste du pipeline inchange et toujours conforme :
#   Conv2D, MaxPooling2D, Flatten, Dense, Dropout, L2,
#   Adam, accuracy, precision, recall, F1-score, matrice de confusion,
#   EarlyStopping, ReduceLROnPlateau, split 70/15/15 stratifie,
#   images RGB 64x64.