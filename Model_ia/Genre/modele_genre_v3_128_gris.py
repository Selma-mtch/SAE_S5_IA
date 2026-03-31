# -*- coding: utf-8 -*-
"""
Modele Genre - Classification binaire (Homme/Femme)
Input : 128x128 grayscale [0, 1]
Sortie : Dense(1, sigmoid) -> 0=Homme, 1=Femme
Dataset : UTKFace (jangedoo/utkface-new)
"""

import gc
import os
from pathlib import Path

import kagglehub
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, callbacks

# --- 0. Configuration ---

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3
L2_FACTOR = 1e-4
DROPOUT_RATE = 0.40
LABEL_NAMES = {0: "Homme", 1: "Femme"}

print(f"TensorFlow : {tf.__version__}")
print(f"GPU : {tf.config.list_physical_devices('GPU')}")

# --- 1. Chargement du dataset ---

def find_utkface_directory():
    kaggle_input = "/kaggle/input"
    if os.path.exists(kaggle_input):
        for root, dirs, files in os.walk(kaggle_input):
            if sum(1 for f in files if f.endswith(".jpg")) > 100:
                return root

    downloaded_path = Path(kagglehub.dataset_download("jangedoo/utkface-new"))
    for candidate in [downloaded_path / "UTKFace", downloaded_path]:
        if candidate.exists():
            jpgs = [f for f in candidate.iterdir() if f.suffix == ".jpg"]
            if len(jpgs) > 100:
                return str(candidate)

    raise FileNotFoundError("Dataset UTKFace introuvable")


image_dir = Path(find_utkface_directory())
image_files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
print(f"Images trouvees : {len(image_files)}")

# --- 2. Extraction des labels et chargement des images ---

images = []
labels = []
ages = []
races = []

for path in image_files:
    try:
        parts = path.stem.split("_")
        if len(parts) < 4:
            continue
        age = int(parts[0])
        gender = int(parts[1])
        try:
            race = int(parts[2])
        except:
            race = 4
        if gender not in (0, 1):
            continue

        img = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
        images.append(np.array(img, dtype=np.uint8))
        labels.append(gender)
        ages.append(age)
        races.append(race)
    except:
        continue

images = np.array(images)[..., np.newaxis]  # (N, 128, 128, 1)
labels = np.array(labels, dtype=np.int32)
ages = np.array(ages, dtype=np.int32)
races = np.array(races, dtype=np.int32)
print(f"Images valides : {len(images)}, shape : {images.shape}")
print(f"Repartition : Homme={np.sum(labels==0)}, Femme={np.sum(labels==1)}")

# --- 2b. Visualisation de la distribution des donnees ---

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.hist(ages, bins=range(0, int(ages.max()) + 5, 5), color="steelblue", edgecolor="black")
plt.title("Distribution des ages")
plt.xlabel("Age")
plt.ylabel("Nombre d'images")
plt.grid(alpha=0.25)

plt.subplot(1, 3, 2)
gender_counts = np.bincount(labels, minlength=2)
plt.bar(["Homme", "Femme"], gender_counts, color=["royalblue", "salmon"], edgecolor="black")
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

# --- 3. Split train/val/test (70/15/15 stratifie) ---

def stratified_split(labels, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


train_idx, val_idx, test_idx = stratified_split(labels)

# Normalisation [0, 1] (compatible avec l'app Android)
x_train = images[train_idx].astype("float32") / 255.0
x_val = images[val_idx].astype("float32") / 255.0
x_test = images[test_idx].astype("float32") / 255.0
y_train = labels[train_idx]
y_val = labels[val_idx]
y_test = labels[test_idx]

del images
gc.collect()

print(f"Train : {len(x_train)}, Val : {len(x_val)}, Test : {len(x_test)}")
print(f"x_train min/max : {x_train.min():.2f} / {x_train.max():.2f}")

# --- 4. Data Augmentation ---

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
], name="data_augmentation")

# --- 5. Construction du modele ---

def build_gender_cnn():
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    x = data_augmentation(inputs)

    # Bloc 1
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloc 2
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloc 3
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloc 4
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(L2_FACTOR))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(L2_FACTOR))(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs=inputs, outputs=outputs)


model = build_gender_cnn()
model.summary()

# --- 6. Compilation ---

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
)

# --- 7. Entrainement ---

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ],
    verbose=1
)

# --- 8. Evaluation ---

def compute_metrics_from_confusion_matrix(conf_matrix):
    total = conf_matrix.sum()
    accuracy = np.trace(conf_matrix) / total if total else 0.0
    per_class = []
    precisions, recalls, f1_scores = [], [], []

    for class_id in range(conf_matrix.shape[0]):
        tp = conf_matrix[class_id, class_id]
        fp = conf_matrix[:, class_id].sum() - tp
        fn = conf_matrix[class_id, :].sum() - tp
        support = conf_matrix[class_id, :].sum()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_class.append({"label": LABEL_NAMES[class_id], "precision": precision,
                          "recall": recall, "f1": f1, "support": int(support)})
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {"accuracy": accuracy, "per_class": per_class,
            "macro_precision": float(np.mean(precisions)),
            "macro_recall": float(np.mean(recalls)),
            "macro_f1": float(np.mean(f1_scores))}


test_loss, test_acc, test_prec, test_recall = model.evaluate(x_test, y_test, verbose=0)
y_pred_proba = model.predict(x_test, verbose=0).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)
conf_matrix = tf.math.confusion_matrix(y_test, y_pred, num_classes=2).numpy()
metrics = compute_metrics_from_confusion_matrix(conf_matrix)

print(f"\nResultats sur le test :")
print(f"  Loss     : {test_loss:.4f}")
print(f"  Accuracy : {test_acc:.4f}")
print(f"  Precision: {test_prec:.4f}")
print(f"  Recall   : {test_recall:.4f}")

print(f"\nPar classe :")
for c in metrics["per_class"]:
    print(f"  {c['label']} -> precision={c['precision']:.4f} | recall={c['recall']:.4f} | f1={c['f1']:.4f} | support={c['support']}")

print(f"\nMoyennes macro :")
print(f"  Precision : {metrics['macro_precision']:.4f}")
print(f"  Recall    : {metrics['macro_recall']:.4f}")
print(f"  F1        : {metrics['macro_f1']:.4f}")

print(f"\nMatrice de confusion :")
print(conf_matrix)

# --- 9. Graphiques ---

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

epochs_range = range(1, len(history.history["loss"]) + 1)

axes[0].plot(epochs_range, history.history["loss"], label="Train")
axes[0].plot(epochs_range, history.history["val_loss"], label="Validation")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history.history["accuracy"], label="Train")
axes[1].plot(epochs_range, history.history["val_accuracy"], label="Validation")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].imshow(conf_matrix, cmap="Blues")
axes[2].set_title("Matrice de confusion")
axes[2].set_xlabel("Prediction")
axes[2].set_ylabel("Verite")
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(["Homme", "Femme"])
axes[2].set_yticks([0, 1])
axes[2].set_yticklabels(["Homme", "Femme"])
for r in range(2):
    for c in range(2):
        axes[2].text(c, r, conf_matrix[r, c], ha="center", va="center")

plt.suptitle("Modele Genre - 128x128 Grayscale", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "courbes_genre_128_gris.png", dpi=150, bbox_inches="tight")
plt.show()

# --- 10. Sauvegarde ---

model.save(OUTPUT_DIR / "modele_genre_128_gris.keras")
print(f"Modele sauvegarde : {OUTPUT_DIR}/modele_genre_128_gris.keras")

# --- 11. Export TFLite ---

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = OUTPUT_DIR / "model_gender.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
print(f"TFLite sauvegarde : {tflite_path} ({size_mb:.1f} Mo)")

# Verification TFLite
interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
outs = interpreter.get_output_details()
print(f"  Input  : shape={inp['shape']}, dtype={inp['dtype']}")
for i, o in enumerate(outs):
    print(f"  Output {i}: shape={o['shape']}, dtype={o['dtype']}, name={o['name']}")
print("Export TFLite OK")
