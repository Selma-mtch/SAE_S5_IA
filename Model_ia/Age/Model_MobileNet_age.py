# =========================
# 0. IMPORTS
# =========================
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kagglehub  # pour télécharger UTKFace

# =========================
# 1. CHARGEMENT DU DATASET UTKFACE
# =========================
path = kagglehub.dataset_download("jangedoo/utkface-new")
data_dir = os.path.join(path, "UTKFace")

files = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]

data = []
for f in files:
    try:
        age = int(f.split("_")[0])
        if age <= 95:
            data.append({
                "path": os.path.join(data_dir, f),
                "age": age
            })
    except:
        pass

df = pd.DataFrame(data).sample(frac=1, random_state=42)
train_df = df.iloc[:int(0.8 * len(df))]
val_df   = df.iloc[int(0.8 * len(df)):]

# =========================
# 2. GÉNÉRATEURS DE DONNÉES
# =========================
IMG_SIZE = 128
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1]
)

train_gen = datagen.flow_from_dataframe(
    train_df,
    x_col="path",
    y_col="age",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="raw"
)

val_gen = datagen.flow_from_dataframe(
    val_df,
    x_col="path",
    y_col="age",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="raw"
)

# =========================
# 3. MOBILE NET V2 - TRANSFERT LEARNING
# =========================
base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",   # ✅ Téléchargement automatique sur Colab
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # Phase 1 : feature extraction

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation="linear")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss="huber",
    metrics=["mae"]
)

model.summary()

# =========================
# 4. CALLBACKS
# =========================
early_stop = callbacks.EarlyStopping(patience=6, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-6)

# =========================
# 5. ENTRAÎNEMENT - PHASE 1
# =========================
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[early_stop, reduce_lr]
)

# =========================
# 6. FINE-TUNING - PHASE 2
# =========================
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Débloquer seulement les dernières couches
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(1e-4),  # LR plus faible pour fine-tuning
    loss="huber",
    metrics=["mae"]
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[early_stop, reduce_lr]
)

# =========================
# 7. VISUALISATION DES RÉSULTATS
# =========================
plt.figure(figsize=(12,4))

# MAE Train / Validation
plt.subplot(1,2,1)
plt.plot(history1.history["mae"] + history2.history["mae"], label="Train MAE")
plt.plot(history1.history["val_mae"] + history2.history["val_mae"], label="Val MAE")
plt.legend()
plt.title("MAE (années)")

# Exemple de prédiction
plt.subplot(1,2,2)
batch_x, batch_y = next(val_gen)
preds = model.predict(batch_x)

plt.imshow(batch_x[0])
plt.title(f"Réel : {batch_y[0]} | Prédit : {preds[0][0]:.1f}")
plt.axis("off")

plt.show()
