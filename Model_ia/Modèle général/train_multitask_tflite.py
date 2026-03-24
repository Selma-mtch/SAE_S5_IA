"""
FaceAI - Modele Multi-Tache (Age, Genre, Ethnicite)
Entraine sur UTKFace, exporte en TensorFlow Lite pour Android.

Fonctionne directement sur Kaggle (detecte le dataset automatiquement)
ou en local avec --dataset.

Le fichier model_multitask.tflite sera genere dans /kaggle/working/
Copiez-le dans app/IA_ethnie/app/src/main/assets/
"""

import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


# ============================================================
# 1. CHARGEMENT DU DATASET
# ============================================================

def find_utkface_path(base):
    """Cherche le dossier contenant les images UTKFace."""
    for root, dirs, files in os.walk(base):
        if any(f.endswith(".jpg") for f in files):
            return root
    return None


def load_dataset(dataset_path, img_size=128):
    """Charge les images UTKFace et extrait les labels."""
    image_folder = find_utkface_path(dataset_path)
    if image_folder is None:
        raise FileNotFoundError(f"Dataset introuvable dans {dataset_path}")

    print(f"Dataset trouve : {image_folder}")

    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
    print(f"Fichiers .jpg trouves : {len(image_files)}")

    images = []
    labels = []

    for file in image_files:
        try:
            parts = file.split("_")
            age = int(parts[0])
            gender = int(parts[1])
            try:
                race = int(parts[2])
            except (ValueError, IndexError):
                race = 4

            img = Image.open(os.path.join(image_folder, file)) \
                       .convert("L") \
                       .resize((img_size, img_size))

            images.append(np.array(img))
            labels.append([age, gender, race])
        except Exception:
            continue

    images = np.array(images)
    labels = np.array(labels)
    print(f"Images chargees : {len(images)}")
    return images, labels


# ============================================================
# 2. PREPARATION DES DONNEES
# ============================================================

def prepare_data(images, labels):
    """Split train/val/test et encode les labels."""
    X = images.reshape(images.shape[0], 128, 128, 1).astype("float32") / 255.0

    y_age = labels[:, 0].astype(np.float32)
    y_gender = labels[:, 1].astype(np.float32)
    y_ethnicity = labels[:, 2].astype(np.int32)

    # Train 70% / Temp 30%
    X_train, X_temp, \
    y_age_train, y_age_temp, \
    y_gender_train, y_gender_temp, \
    y_eth_train, y_eth_temp = train_test_split(
        X, y_age, y_gender, y_ethnicity,
        test_size=0.3, random_state=42, stratify=y_ethnicity
    )

    # Val 15% / Test 15%
    X_val, X_test, \
    y_age_val, y_age_test, \
    y_gender_val, y_gender_test, \
    y_eth_val, y_eth_test = train_test_split(
        X_temp, y_age_temp, y_gender_temp, y_eth_temp,
        test_size=0.5, random_state=42, stratify=y_eth_temp
    )

    # Versions tournees pour test
    X_test_90 = np.rot90(X_test, k=1, axes=(1, 2))
    X_test_180 = np.rot90(X_test, k=2, axes=(1, 2))

    # Encodage ethnicite
    y_eth_train_cat = to_categorical(y_eth_train, num_classes=5)
    y_eth_val_cat = to_categorical(y_eth_val, num_classes=5)
    y_eth_test_cat = to_categorical(y_eth_test, num_classes=5)

    # Class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_eth_train),
        y=y_eth_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights (ethnicite) : {class_weight_dict}")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "X_test_90": X_test_90, "X_test_180": X_test_180,
        "y_age_train": y_age_train, "y_age_val": y_age_val, "y_age_test": y_age_test,
        "y_gender_train": y_gender_train, "y_gender_val": y_gender_val, "y_gender_test": y_gender_test,
        "y_eth_train_cat": y_eth_train_cat, "y_eth_val_cat": y_eth_val_cat, "y_eth_test_cat": y_eth_test_cat,
        "class_weight_dict": class_weight_dict,
    }


# ============================================================
# 3. DATA AUGMENTATION (via tf.data, pas dans le modele)
# ============================================================

def random_90_rotation(x):
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    return tf.image.rot90(x, k)


def augment_image(image):
    """Applique l'augmentation sur une image."""
    image = random_90_rotation(image)
    image = tf.image.random_brightness(image, 0.1)
    return image


def create_augmented_dataset(X, y_dict, batch_size=128):
    """Cree un tf.data.Dataset avec augmentation."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y_dict))

    def augment_fn(image, labels):
        image = random_90_rotation(image)
        # Random rotation +-180
        angle = tf.random.uniform([], -0.5, 0.5)
        image = tf.keras.layers.RandomRotation(0.5)(tf.expand_dims(image, 0), training=True)[0]
        return image, labels

    dataset = dataset.shuffle(len(X))
    dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ============================================================
# 4. MODELE MULTI-TACHE (un seul modele, sans augmentation dedans)
# ============================================================

def build_model():
    """Modele unique pour entrainement ET export TFLite."""
    input_img = layers.Input(shape=(128, 128, 1))

    x = input_img

    # Bloc CNN 1
    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloc CNN 2
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloc CNN 3
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloc CNN 4
    x = layers.Conv2D(256, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Features partagees
    shared = layers.Dense(256, activation="relu")(x)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.4)(shared)

    # Branche AGE (regression)
    age_branch = layers.Dense(128, activation="relu")(shared)
    age_branch = layers.Dense(64, activation="relu")(age_branch)
    age_output = layers.Dense(1, activation="linear", name="age")(age_branch)

    # Branche GENDER (classification binaire)
    gender_branch = layers.Dense(128, activation="relu")(shared)
    gender_branch = layers.Dropout(0.3)(gender_branch)
    gender_output = layers.Dense(1, activation="sigmoid", name="gender")(gender_branch)

    # Branche ETHNICITY (5 classes)
    eth_branch = layers.Dense(256, activation="relu")(shared)
    eth_branch = layers.Dense(128, activation="relu")(eth_branch)
    eth_branch = layers.Dropout(0.5)(eth_branch)
    ethnicity_output = layers.Dense(5, activation="softmax", name="ethnicity")(eth_branch)

    model = models.Model(
        inputs=input_img,
        outputs=[age_output, gender_output, ethnicity_output]
    )
    return model


# ============================================================
# 5. ENTRAINEMENT
# ============================================================

def train_model(model, data, epochs=80, batch_size=128):
    """Compile et entraine le modele avec augmentation via numpy."""
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss={
            "age": tf.keras.losses.Huber(delta=8.0),
            "gender": "binary_crossentropy",
            "ethnicity": "categorical_crossentropy",
        },
        loss_weights={"age": 0.4, "gender": 1.0, "ethnicity": 1.0},
        metrics={
            "age": ["mae"],
            "gender": ["accuracy"],
            "ethnicity": ["accuracy"],
        },
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Augmentation via numpy (rotations aleatoires de 90)
    X_train = data["X_train"]
    X_aug = X_train.copy()
    for i in range(len(X_aug)):
        k = np.random.randint(0, 4)
        X_aug[i] = np.rot90(X_aug[i], k=k, axes=(0, 1))

    X_combined = np.concatenate([X_train, X_aug], axis=0)
    y_age_combined = np.concatenate([data["y_age_train"], data["y_age_train"]], axis=0)
    y_gender_combined = np.concatenate([data["y_gender_train"], data["y_gender_train"]], axis=0)
    y_eth_combined = np.concatenate([data["y_eth_train_cat"], data["y_eth_train_cat"]], axis=0)

    print(f"Dataset augmente : {len(X_combined)} images (original + rotations)")

    history = model.fit(
        X_combined,
        {"age": y_age_combined, "gender": y_gender_combined, "ethnicity": y_eth_combined},
        validation_data=(
            data["X_val"],
            {"age": data["y_age_val"], "gender": data["y_gender_val"], "ethnicity": data["y_eth_val_cat"]},
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
    )

    return history


# ============================================================
# 6. EVALUATION
# ============================================================

def evaluate_model(model, data):
    """Evalue sur les jeux de test (original + rotations)."""
    test_labels = {
        "age": data["y_age_test"],
        "gender": data["y_gender_test"],
        "ethnicity": data["y_eth_test_cat"],
    }

    print("\n===== TEST ORIGINAL =====")
    model.evaluate(data["X_test"], test_labels)

    print("\n===== TEST 90° =====")
    model.evaluate(data["X_test_90"], test_labels)

    print("\n===== TEST 180° =====")
    model.evaluate(data["X_test_180"], test_labels)


def save_plots(history, output_dir):
    """Sauvegarde les graphiques d'entrainement."""
    os.makedirs(output_dir, exist_ok=True)

    # Loss globale
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss globale du modele multi-tache")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()

    # Accuracy genre
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["gender_accuracy"], label="Train Acc")
    plt.plot(history.history["val_gender_accuracy"], label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy - Genre")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "gender_accuracy.png"))
    plt.close()

    # Accuracy ethnicite
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["ethnicity_accuracy"], label="Train Acc")
    plt.plot(history.history["val_ethnicity_accuracy"], label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy - Ethnicite")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "ethnicity_accuracy.png"))
    plt.close()

    print(f"Graphiques sauvegardes dans {output_dir}/")


# ============================================================
# 7. EXPORT TENSORFLOW LITE
# ============================================================

def export_tflite(model, output_path="model_multitask.tflite"):
    """
    Exporte le modele directement en TFLite.
    Pas besoin de modele separee puisque l'augmentation
    n'est plus dans le modele.
    """
    print("\n===== EXPORT TFLITE =====")

    # Verification rapide que le modele fonctionne
    test_input = np.random.rand(1, 128, 128, 1).astype(np.float32)
    outputs = model(test_input, training=False)
    print("Verification pre-export:")
    for i, name in enumerate(["age", "gender", "ethnicity"]):
        print(f"  {name}: {outputs[i].numpy().flatten()}")

    # Conversion TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nModele TFLite sauvegarde : {output_path} ({size_mb:.1f} MB)")

    # Verifier le modele TFLite
    verify_tflite(output_path)

    return output_path


def verify_tflite(tflite_path):
    """Verifie que le modele TFLite fonctionne correctement."""
    print("\n===== VERIFICATION TFLITE =====")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\nInput:")
    for d in input_details:
        print(f"  {d['name']}: shape={d['shape']}, dtype={d['dtype']}")

    print(f"\nOutputs ({len(output_details)}):")
    for d in output_details:
        print(f"  {d['name']}: shape={d['shape']}, dtype={d['dtype']}")

    # Test avec une image aleatoire
    test_input = np.random.rand(1, 128, 128, 1).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()

    outputs = []
    for d in output_details:
        outputs.append(interpreter.get_tensor(d["index"]))

    # Identifier les sorties par leur shape
    for i, out in enumerate(outputs):
        shape = out.shape
        if shape == (1, 1) or shape == (1,):
            val = out.flatten()[0]
            if 0 <= val <= 1:
                print(f"  Output {i}: gender (sigmoid) = {val:.4f}")
            else:
                print(f"  Output {i}: age = {val:.1f}")
        elif shape[-1] == 5:
            print(f"  Output {i}: ethnicity (softmax) = {out.flatten()}")

    print("\nTFLite OK !")


# ============================================================
# 8. MAIN
# ============================================================

def detect_dataset_path():
    """Detecte automatiquement le dataset sur Kaggle ou en local."""
    # Chemins Kaggle courants
    kaggle_paths = [
        "/kaggle/input",
        "/kaggle/input/utkface-new",
        "/kaggle/input/utkface-new/UTKFace",
        "/kaggle/input/datasets/jangedoo/utkface-new/UTKFace",
    ]
    for p in kaggle_paths:
        if os.path.exists(p):
            found = find_utkface_path(p)
            if found:
                return found

    # Local
    local_paths = ["./UTKFace", "../UTKFace", "~/UTKFace"]
    for p in local_paths:
        expanded = os.path.expanduser(p)
        if os.path.exists(expanded):
            found = find_utkface_path(expanded)
            if found:
                return found

    return None


def main():
    # Sur Kaggle, sys.argv peut contenir des trucs inattendus
    # On utilise argparse seulement si des args sont passes
    import sys

    dataset_path = None
    epochs = 80
    batch_size = 128
    output = "/kaggle/working/model_multitask.tflite" if os.path.exists("/kaggle/working") else "model_multitask.tflite"
    plots_dir = "/kaggle/working/plots" if os.path.exists("/kaggle/working") else "plots"
    save_keras = None

    # Parser les arguments si on est en mode CLI
    if len(sys.argv) > 1 and "--" in sys.argv[1]:
        parser = argparse.ArgumentParser(description="FaceAI - Entrainement multi-tache + export TFLite")
        parser.add_argument("--dataset", type=str, default=None)
        parser.add_argument("--epochs", type=int, default=80)
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--output", type=str, default=output)
        parser.add_argument("--plots-dir", type=str, default=plots_dir)
        parser.add_argument("--save-keras", type=str, default=None)
        args = parser.parse_args()
        dataset_path = args.dataset
        epochs = args.epochs
        batch_size = args.batch_size
        output = args.output
        plots_dir = args.plots_dir
        save_keras = args.save_keras

    # Detection automatique du dataset si pas specifie
    if dataset_path is None:
        print("Aucun --dataset specifie, detection automatique...")
        dataset_path = detect_dataset_path()
        if dataset_path is None:
            raise FileNotFoundError(
                "Dataset UTKFace introuvable. "
                "Sur Kaggle: ajoutez le dataset 'utkface-new' a votre notebook. "
                "En local: utilisez --dataset /chemin/vers/UTKFace"
            )
        print(f"Dataset detecte : {dataset_path}")

    # 1. Charger les donnees
    print("=" * 50)
    print("1. CHARGEMENT DU DATASET")
    print("=" * 50)
    images, labels = load_dataset(dataset_path)

    # 2. Preparer les donnees
    print("\n" + "=" * 50)
    print("2. PREPARATION DES DONNEES")
    print("=" * 50)
    data = prepare_data(images, labels)

    # 3. Construire le modele
    print("\n" + "=" * 50)
    print("3. CONSTRUCTION DU MODELE")
    print("=" * 50)
    model = build_model()
    model.summary()

    # 4. Entrainer
    print("\n" + "=" * 50)
    print("4. ENTRAINEMENT")
    print("=" * 50)
    history = train_model(model, data, epochs=epochs, batch_size=batch_size)

    # 5. Evaluer
    print("\n" + "=" * 50)
    print("5. EVALUATION")
    print("=" * 50)
    evaluate_model(model, data)

    # 6. Sauvegarder les graphiques
    save_plots(history, plots_dir)

    # 7. Sauvegarder le modele Keras si demande
    if save_keras:
        model.save(save_keras)
        print(f"Modele Keras sauvegarde : {save_keras}")

    # 8. Exporter en TFLite
    print("\n" + "=" * 50)
    print("6. EXPORT TFLITE")
    print("=" * 50)
    tflite_path = export_tflite(model, output)

    print("\n" + "=" * 50)
    print("TERMINE !")
    print("=" * 50)
    print(f"\nFichier TFLite : {tflite_path}")
    print(f"Copiez-le dans : app/IA_ethnie/app/src/main/assets/")
    print(f"\nEntree  : image 128x128 grayscale, float32, [0-1]")
    print(f"Sorties : age (float), gender (0=H/1=F), ethnicity (5 classes softmax)")


if __name__ == "__main__":
    main()
