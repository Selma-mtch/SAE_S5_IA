# -*- coding: utf-8 -*-
"""
Modèle Multi-Tâches CNN (Age + Genre + Ethnicité)

Dataset : UTKFace (jangedoo/utkface-new)
Input : Grayscale 128x128 [0, 1]
Sorties : age (régression), genre (sigmoid), ethnicité (softmax 5 classes)
"""

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Reproductibilité
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print(f"TensorFlow version : {tf.__version__}")
print(f"GPU disponible : {tf.config.list_physical_devices('GPU')}")

# --- 1. Chargement des données ---

def find_utkface_path(base="/kaggle/input"):
    for root, dirs, files in os.walk(base):
        if any(f.endswith(".jpg") for f in files):
            return root
    return None

image_folder = find_utkface_path()

if image_folder is None:
    try:
        import kagglehub
        print("Téléchargement du dataset via kagglehub...")
        base_path = kagglehub.dataset_download("jangedoo/utkface-new")
        image_folder = os.path.join(base_path, "UTKFace")
    except Exception:
        raise FileNotFoundError("Dataset UTKFace introuvable")

print(f"Dataset trouvé : {image_folder}")

image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
print(f"Nombre de fichiers .jpg : {len(image_files)}")

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

        img = Image.open(os.path.join(image_folder, file)) \
                   .convert("L") \
                   .resize((128, 128))

        images.append(np.array(img))
        labels.append([age, gender, race])
    except:
        continue

images = np.array(images)
labels = np.array(labels)
print(f"Images chargées : {len(images)}, shape : {images.shape}")

# --- 2. Préparation des données ---

X = images.reshape(images.shape[0], 128, 128, 1).astype('float32') / 255.0
print(f"X shape : {X.shape}, min/max : {X.min():.2f} / {X.max():.2f}")

y_age = labels[:, 0].astype(np.float32)  # âge brut en années
y_gender = labels[:, 1].astype(np.float32)
y_ethnicity = labels[:, 2].astype(np.int32)

eth_labels = ['Blanc', 'Noir', 'Asiatique', 'Indien', 'Autre']
gender_labels = ['Homme', 'Femme']

# Train (70%) / Val (15%) / Test (15%)
X_train, X_temp, \
y_age_train, y_age_temp, \
y_gender_train, y_gender_temp, \
y_eth_train, y_eth_temp = train_test_split(
    X, y_age, y_gender, y_ethnicity,
    test_size=0.3, random_state=42, stratify=y_ethnicity
)

X_val, X_test, \
y_age_val, y_age_test, \
y_gender_val, y_gender_test, \
y_eth_val, y_eth_test = train_test_split(
    X_temp, y_age_temp, y_gender_temp, y_eth_temp,
    test_size=0.5, random_state=42, stratify=y_eth_temp
)

# Versions tournées du test set
X_test_90 = np.rot90(X_test, k=1, axes=(1, 2))
X_test_180 = np.rot90(X_test, k=2, axes=(1, 2))

# Encodage ethnicité
y_eth_train_cat = to_categorical(y_eth_train, num_classes=5)
y_eth_val_cat = to_categorical(y_eth_val, num_classes=5)
y_eth_test_cat = to_categorical(y_eth_test, num_classes=5)

# Class weights -> sample weights pour l'ethnicité
class_weights = compute_class_weight('balanced', classes=np.unique(y_eth_train), y=y_eth_train)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights (ethnicité) : {class_weight_dict}")

# Loss custom qui intègre les class weights (sample_weight non supporté en Keras 3 multi-output)
class_weight_tensor = tf.constant([class_weight_dict[i] for i in range(5)], dtype=tf.float32)

def weighted_categorical_crossentropy(y_true, y_pred):
    weights = tf.reduce_sum(y_true * class_weight_tensor, axis=-1)
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return loss * weights

print(f"Train : {X_train.shape[0]}, Val : {X_val.shape[0]}, Test : {X_test.shape[0]}")

# --- 3. Data Augmentation ---

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.05),
    layers.RandomTranslation(0.02, 0.02),
])

# --- 4. Construction du modèle ---

input_img = layers.Input(shape=(128, 128, 1))
x = data_augmentation(input_img)

# Bloc CNN 1
x = layers.Conv2D(32, (3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D((2, 2))(x)

# Bloc CNN 2
x = layers.Conv2D(64, (3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D((2, 2))(x)

# Bloc CNN 3
x = layers.Conv2D(128, (3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D((2, 2))(x)

# Bloc CNN 4
x = layers.Conv2D(256, (3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

# Global Pooling + features partagées
x = layers.GlobalAveragePooling2D()(x)
shared = layers.Dense(256, activation='relu')(x)
shared = layers.BatchNormalization()(shared)
shared = layers.Dropout(0.4)(shared)

# Branche Age (régression)
age_branch = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(shared)
age_branch = Dropout(0.4)(age_branch)
age_branch = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(age_branch)
age_output = Dense(1, activation='linear', name='age')(age_branch)

# Branche Genre (classification binaire)
gender_branch = layers.Dense(128, activation='relu')(shared)
gender_branch = layers.Dropout(0.3)(gender_branch)
gender_output = layers.Dense(1, activation='sigmoid', name='gender')(gender_branch)

# Branche Ethnicité (classification 5 classes)
eth_branch = layers.Dense(256, activation='relu')(shared)
eth_branch = layers.Dense(128, activation='relu')(eth_branch)
eth_branch = layers.Dropout(0.5)(eth_branch)
ethnicity_output = layers.Dense(5, activation='softmax', name='ethnicity')(eth_branch)

model = models.Model(inputs=input_img, outputs=[age_output, gender_output, ethnicity_output])

# --- 5. Compilation ---

model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss={
        'age': tf.keras.losses.Huber(delta=8.0),
        'gender': 'binary_crossentropy',
        'ethnicity': weighted_categorical_crossentropy
    },
    loss_weights={'age': 0.05, 'gender': 0.8, 'ethnicity': 1.0},
    metrics={
        'age': ['mae'],
        'gender': ['accuracy'],
        'ethnicity': ['accuracy']
    }
)

model.summary()

# --- 5b. Vérifications avant entraînement ---
print("\n" + "=" * 50)
print("VÉRIFICATIONS AVANT ENTRAÎNEMENT")
print("=" * 50)

errors = []

# Vérif 1 : shapes des données
if X_train.shape[1:] != (128, 128, 1):
    errors.append(f"X_train shape incorrecte : {X_train.shape}, attendu (N, 128, 128, 1)")
if X_train.min() < -0.01 or X_train.max() > 1.01:
    errors.append(f"X_train hors range [0,1] : min={X_train.min():.4f}, max={X_train.max():.4f}")

# Vérif 2 : âge brut (pas normalisé)
if y_age_train.max() <= 1.0:
    errors.append(f"y_age_train semble normalisé [0,1] (max={y_age_train.max():.2f}). Attendu : âge brut en années")
if y_age_train.min() < 0:
    errors.append(f"y_age_train contient des valeurs négatives : min={y_age_train.min():.2f}")

# Vérif 3 : genre binaire
unique_genders = np.unique(y_gender_train)
if not np.array_equal(unique_genders, [0., 1.]):
    errors.append(f"y_gender_train pas binaire [0,1] : valeurs uniques = {unique_genders}")

# Vérif 4 : ethnicité one-hot
if y_eth_train_cat.shape[1] != 5:
    errors.append(f"y_eth_train_cat devrait avoir 5 classes, a {y_eth_train_cat.shape[1]}")

# Vérif 5 : outputs du modèle
output_names = [o.name for o in model.outputs]
print(f"  Outputs du modèle : {output_names}")
for o in model.outputs:
    print(f"    {o.name} : shape={o.shape}")

# Vérif 6 : test rapide forward pass
try:
    test_pred = model.predict(X_train[:2], verbose=0)
    print(f"  Forward pass OK : age={test_pred[0][0]}, gender={test_pred[1][0]}, eth={test_pred[2][0]}")
except Exception as e:
    errors.append(f"Forward pass échoué : {e}")

# Vérif 7 : loss weights vs loss magnitudes
print(f"  Loss weights : age=0.05, gender=0.8, ethnicity=1.0")
print(f"  Age range : [{y_age_train.min():.0f}, {y_age_train.max():.0f}] ans")
print(f"  Huber delta=8.0 → erreurs > 8 ans pénalisées linéairement")

if errors:
    print("\n ERREURS DÉTECTÉES :")
    for e in errors:
        print(f"  - {e}")
    raise ValueError("Vérifications échouées. Corrigez les erreurs ci-dessus avant d'entraîner.")
else:
    print("\n  Toutes les vérifications sont OK !")

print("=" * 50)

# --- 6. Entraînement ---

early_stop = EarlyStopping(
    monitor='val_loss', patience=12,
    restore_best_weights=True, verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5,
    patience=5, min_lr=1e-6, verbose=1
)

history = model.fit(
    X_train,
    {'age': y_age_train, 'gender': y_gender_train, 'ethnicity': y_eth_train_cat},
    validation_data=(
        X_val,
        {'age': y_age_val, 'gender': y_gender_val, 'ethnicity': y_eth_val_cat}
    ),
    epochs=80,
    batch_size=128,
    callbacks=[early_stop, lr_scheduler],
    shuffle=True
)

# --- 7. Évaluation ---

OUTPUT_PATH = "/kaggle/working"

print("\n===== TEST ORIGINAL =====")
model.evaluate(X_test, {'age': y_age_test, 'gender': y_gender_test, 'ethnicity': y_eth_test_cat})

print("\n===== TEST 90° =====")
model.evaluate(X_test_90, {'age': y_age_test, 'gender': y_gender_test, 'ethnicity': y_eth_test_cat})

print("\n===== TEST 180° =====")
model.evaluate(X_test_180, {'age': y_age_test, 'gender': y_gender_test, 'ethnicity': y_eth_test_cat})

# Métriques détaillées
results = model.evaluate(X_test, {'age': y_age_test, 'gender': y_gender_test, 'ethnicity': y_eth_test_cat}, verbose=0)
print("\nMétriques finales :")
for name, value in zip(model.metrics_names, results):
    print(f"  {name}: {value:.4f}")

# --- 8. Graphiques ---

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss globale
axes[0, 0].plot(history.history['loss'], label='Train')
axes[0, 0].plot(history.history['val_loss'], label='Validation')
axes[0, 0].set_title('Loss globale')
axes[0, 0].set_xlabel('Epochs')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# MAE Age
axes[0, 1].plot(history.history['age_mae'], label='Train MAE')
axes[0, 1].plot(history.history['val_age_mae'], label='Val MAE')
axes[0, 1].set_title('MAE - Âge (années)')
axes[0, 1].set_xlabel('Epochs')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Accuracy Genre
axes[1, 0].plot(history.history['gender_accuracy'], label='Train')
axes[1, 0].plot(history.history['val_gender_accuracy'], label='Validation')
axes[1, 0].set_title('Accuracy - Genre')
axes[1, 0].set_xlabel('Epochs')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Accuracy Ethnicité
axes[1, 1].plot(history.history['ethnicity_accuracy'], label='Train')
axes[1, 1].plot(history.history['val_ethnicity_accuracy'], label='Validation')
axes[1, 1].set_title('Accuracy - Ethnicité')
axes[1, 1].set_xlabel('Epochs')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Modèle Multi-Tâches CNN (ageMauvais)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'courbes_multitache_ageMauvais.png'), dpi=150, bbox_inches='tight')
plt.show()

# --- 9. Sauvegarde des métriques ---

metrics_df = pd.DataFrame(history.history)
metrics_df.to_csv(os.path.join(OUTPUT_PATH, 'metrics_training.csv'), index=False)

with open(os.path.join(OUTPUT_PATH, 'test_metrics.txt'), 'w') as f:
    for name, value in zip(model.metrics_names, results):
        f.write(f"{name}: {value:.4f}\n")

# --- 10. Export ---

model.save(os.path.join(OUTPUT_PATH, 'model_multitache_ageMauvais.keras'))
print(f"Modèle sauvegardé : {OUTPUT_PATH}/model_multitache_ageMauvais.keras")

# Export TFLite - modèle d'inférence SANS data augmentation
# On convertit directement le modèle entraîné.
# Les couches RandomRotation/RandomZoom/RandomTranslation sont des no-ops
# quand training=False (ce qui est le cas en inférence TFLite).
# On vérifie quand même que c'est bien le cas.

print("\nVérification : data augmentation désactivée en inference...")
test_input = X_test[:3]
preds = []
for _ in range(5):
    p = model.predict(test_input, verbose=0)
    preds.append(p[0][0][0])  # age du premier sample

if len(set([round(p, 4) for p in preds])) == 1:
    print("  OK : prédictions identiques (augmentation inactive en inference)")
else:
    print(f"  ATTENTION : prédictions différentes entre appels : {preds}")
    print("  L'augmentation est encore active ! Les prédictions TFLite seront instables.")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = os.path.join(OUTPUT_PATH, 'model_multitache_ageMauvais.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
print(f"TFLite sauvegardé : {tflite_path} ({size_mb:.1f} Mo)")

# Vérification TFLite
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"  Input  : shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
for i, od in enumerate(output_details):
    print(f"  Output {i} : name={od['name']}, shape={od['shape']}, dtype={od['dtype']}")
print("Conversion OK")
