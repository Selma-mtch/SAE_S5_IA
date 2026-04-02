# =============================================================================
# Modele CNN - Prediction d'age (UTKFace)
# =============================================================================

import gc
from pathlib import Path

import kagglehub
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, callbacks

# -----------------------------------------------------------------------------
# Parametres globaux
# -----------------------------------------------------------------------------

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_DIR    = None
DATASET_HANDLE = "jangedoo/utkface-new"
OUTPUT_DIR     = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

IMAGE_SIZE       = 128
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

BATCH_SIZE    = 32
EPOCHS        = 50
LEARNING_RATE = 3e-4
L2_FACTOR     = 1e-4
DROPOUT_RATE  = 0.25  # Reduit de 0.40 a 0.25 pour la regression fine

EARLY_STOPPING_PATIENCE = 10
LR_PATIENCE             = 3
LR_REDUCTION_FACTOR     = 0.5

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

# -----------------------------------------------------------------------------
# Fonctions utilitaires
# -----------------------------------------------------------------------------

def parse_utkface_filename(filename):
    """Extrait age, sexe et race a partir du nom de fichier UTKFace."""
    stem  = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    try:
        age    = int(parts[0])
        gender = int(parts[1])
        race   = int(parts[2])
    except ValueError:
        return None
    if age < 0:
        return None
    return {"filename": filename, "age": age, "gender": gender, "race": race}


def find_utkface_directory(dataset_dir=None, dataset_handle="jangedoo/utkface-new"):
    """Charge UTKFace via kagglehub ou utilise un chemin fourni manuellement."""
    dataset_path = Path(dataset_dir) if dataset_dir is not None \
                   else Path(kagglehub.dataset_download(dataset_handle))

    for candidate in [dataset_path / "UTKFace", dataset_path / "crop_part1", dataset_path]:
        if not candidate.exists():
            continue
        image_files = [p for p in candidate.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        if not image_files:
            continue
        if sum(parse_utkface_filename(p.name) is not None for p in image_files[:20]) >= 10:
            return candidate

    raise FileNotFoundError("Impossible de trouver un dossier d'images UTKFace valide.")


def load_images_and_labels(records, image_dir, image_size):
    """Charge toutes les images, les redimensionne et recupere les labels d'age."""
    images = np.empty((len(records), image_size, image_size, 3), dtype=np.uint8)
    ages   = np.empty(len(records), dtype=np.float32)
    for index, record in enumerate(records):
        with Image.open(image_dir / record["filename"]) as img:
            images[index] = np.asarray(img.convert("RGB").resize((image_size, image_size)), dtype=np.uint8)
        ages[index] = float(record["age"])
        if (index + 1) % 5000 == 0 or index == len(records) - 1:
            print(f"Images chargees : {index + 1}/{len(records)}")
    return images, ages


def random_split_indices(n_samples, train_ratio, val_ratio, test_ratio, seed=42):
    """Construit un split aleatoire simple."""
    rng     = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_train = int(n_samples * train_ratio)
    n_val   = int(n_samples * val_ratio)
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]


def compute_regression_metrics(y_true_years, y_pred_years):
    """Calcule MSE, MAE et R2 en annees."""
    y_true = y_true_years.astype(np.float32).reshape(-1)
    y_pred = y_pred_years.astype(np.float32).reshape(-1)
    mse    = float(np.mean((y_true - y_pred) ** 2))
    mae    = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return {"mse": mse, "mae": mae, "r2": r2}

# -----------------------------------------------------------------------------
# Data augmentation
# Data augmentation geometrique uniquement : compatible Standardisation + BN.
# RandomFlip et RandomRotation ne modifient pas la distribution des pixels.
# -----------------------------------------------------------------------------

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
], name="data_augmentation")

# -----------------------------------------------------------------------------
# Architecture du modele
# -----------------------------------------------------------------------------

def build_age_cnn(input_shape):
    """
    CNN de regression pour la prediction d'age.

    Architecture : 4 blocs Conv2D (32->64->128->256) + BN + MaxPool,
    suivi GlobalAveragePooling2D, deux couches Dense avec L2 + BN + Dropout,
    sortie Dense(1) sans activation (regression lineaire).
    """
    inputs = layers.Input(shape=input_shape)
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
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(L2_FACTOR))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(L2_FACTOR))(x)

    # Sortie lineaire : prediction en espace normalise
    outputs = layers.Dense(1)(x)

    return models.Model(inputs=inputs, outputs=outputs)

# -----------------------------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Chargement du dataset
    image_dir       = find_utkface_directory(DATASET_DIR, DATASET_HANDLE)
    all_image_paths = sorted([p for p in image_dir.iterdir()
                               if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])

    records = [r for p in all_image_paths
               if (r := parse_utkface_filename(p.name)) is not None]
    print(f"Images valides : {len(records)}")

    images_uint8, age_targets = load_images_and_labels(records, image_dir, IMAGE_SIZE)

    # 2. Split train / val / test
    train_idx, val_idx, test_idx = random_split_indices(
        len(age_targets), TRAIN_RATIO, VAL_RATIO, TEST_RATIO, seed=SEED
    )

    x_train_raw   = images_uint8[train_idx].astype("float32")
    x_val_raw     = images_uint8[val_idx].astype("float32")
    x_test_raw    = images_uint8[test_idx].astype("float32")
    y_train_years = age_targets[train_idx].astype("float32")
    y_val_years   = age_targets[val_idx].astype("float32")
    y_test_years  = age_targets[test_idx].astype("float32")

    del images_uint8
    gc.collect()

    # 3. Standardisation des images (fit sur train uniquement)
    train_img_mean = x_train_raw.mean()
    train_img_std  = x_train_raw.std()

    x_train = (x_train_raw - train_img_mean) / train_img_std
    x_val   = (x_val_raw   - train_img_mean) / train_img_std
    x_test  = (x_test_raw  - train_img_mean) / train_img_std

    del x_train_raw, x_val_raw, x_test_raw
    gc.collect()

    # 4. Normalisation des labels (fit sur y_train uniquement)
    age_mean = float(y_train_years.mean())
    age_std  = float(y_train_years.std())

    y_train = (y_train_years - age_mean) / age_std
    y_val   = (y_val_years   - age_mean) / age_std
    y_test  = (y_test_years  - age_mean) / age_std

    # 5. Construction et compilation du modele
    model = build_age_cnn((IMAGE_SIZE, IMAGE_SIZE, 3))
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )

    training_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_REDUCTION_FACTOR,
            patience=LR_PATIENCE,
            verbose=1,
        ),
    ]

    # 6. Entrainement
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=training_callbacks,
        verbose=1,
    )

    # 7. Evaluation sur le test
    y_pred_norm  = model.predict(x_test, verbose=0).reshape(-1)
    y_pred_years = y_pred_norm * age_std + age_mean
    metrics      = compute_regression_metrics(y_test_years, y_pred_years)
    print(f"\nTest  MSE={metrics['mse']:.2f}  MAE={metrics['mae']:.2f}  R2={metrics['r2']:.4f}")

    # 8. Sauvegarde du modele et des parametres de normalisation
    model_path = OUTPUT_DIR / "modele_age_utkface.keras"
    model.save(model_path)
    print(f"Modele sauvegarde : {model_path}")

    np.save(OUTPUT_DIR / "age_train_img_mean.npy", np.array([train_img_mean]))
    np.save(OUTPUT_DIR / "age_train_img_std.npy",  np.array([train_img_std]))
    np.save(OUTPUT_DIR / "age_label_mean.npy",     np.array([age_mean]))
    np.save(OUTPUT_DIR / "age_label_std.npy",      np.array([age_std]))

    # 9. Graphiques
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 9a. Courbes d'entrainement (MAE normalise)
    axes[0, 0].plot(history.history['mae'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_mae'], label='Validation', linewidth=2)
    axes[0, 0].set_title('MAE durant l\'entrainement')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MAE (normalise)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 9b. Loss
    axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Loss (MSE) durant l\'entrainement')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 9c. Age reel vs predit
    axes[1, 0].scatter(y_test_years, y_pred_years, alpha=0.3, s=10, color='teal')
    axes[1, 0].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Prediction parfaite')
    axes[1, 0].set_xlabel('Age reel')
    axes[1, 0].set_ylabel('Age predit')
    axes[1, 0].set_title(f'Age reel vs predit (MAE={metrics["mae"]:.2f} ans)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 9d. Distribution des erreurs
    errors = y_pred_years - y_test_years
    axes[1, 1].hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Erreur (annees)')
    axes[1, 1].set_ylabel('Nombre')
    axes[1, 1].set_title(f'Distribution des erreurs (MAE={metrics["mae"]:.2f} ans)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_curves_age.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 10. Export TFLite avec preprocessing + denormalisation integres
    print("\n" + "=" * 50)
    print("EXPORT TENSORFLOW LITE")
    print("=" * 50)

    # Construire un modele d'inference : RGB [0,255] -> age en annees
    inference_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="input_image")

    # Bake la standardisation des images dans le modele
    x_inf = (inference_input - train_img_mean) / train_img_std

    # Passer a travers le modele entraine (sans augmentation)
    for layer in model.layers:
        if layer.name == 'data_augmentation' or isinstance(layer, layers.InputLayer):
            continue
        x_inf = layer(x_inf)

    # Denormaliser la sortie : age_normalise -> age en annees
    age_output = x_inf * age_std + age_mean
    inference_model = models.Model(inputs=inference_input, outputs=age_output, name="age_inference")

    # Verifier que les predictions matchent
    test_img = x_test_raw[:5] if 'x_test_raw' in dir() else (x_test * train_img_std + train_img_mean)[:5]
    pred_original = y_pred_years[:5]
    pred_inference = inference_model.predict(test_img, verbose=0).flatten()
    for i in range(5):
        diff = abs(pred_original[i] - pred_inference[i])
        print(f"  Verification {i}: original={pred_original[i]:.1f}, inference={pred_inference[i]:.1f}, diff={diff:.2f}")

    # Conversion TFLite
    tflite_path = OUTPUT_DIR / 'model_age.tflite'
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = tflite_path.stat().st_size / (1024 * 1024)
    print(f"\nModele TFLite sauvegarde : {tflite_path} ({size_mb:.1f} MB)")

    # Verification TFLite
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"  Input  : shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
    print(f"  Output : shape={output_details[0]['shape']}, dtype={output_details[0]['dtype']}")

    # Test sur une vraie image
    test_sample = (x_test[:1] * train_img_std + train_img_mean).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_sample)
    interpreter.invoke()
    tflite_age = interpreter.get_tensor(output_details[0]['index']).flatten()[0]
    real_age = y_test_years[0]
    print(f"  Test : reel={real_age:.0f} ans, TFLite={tflite_age:.1f} ans")
    print("  TFLite OK!")

    print("\n" + "=" * 50)
    print("FICHIERS SAUVEGARDES")
    print("=" * 50)
    print("  - modele_age_utkface.keras")
    print("  - model_age.tflite")
    print("  - training_curves_age.png")
    print("  - age_train_img_mean.npy / age_train_img_std.npy")
    print("  - age_label_mean.npy / age_label_std.npy")