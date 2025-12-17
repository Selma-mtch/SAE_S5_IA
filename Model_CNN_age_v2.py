import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import kagglehub

# --- 1. PRÉPARATION DES DONNÉES ---
path = kagglehub.dataset_download("jangedoo/utkface-new")
data_dir = os.path.join(path, 'UTKFace')

files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
data = [{'path': os.path.join(data_dir, f), 'age': int(f.split('_')[0])} for f in files if int(f.split('_')[0]) <= 95]
df = pd.DataFrame(data).sample(frac=1, random_state=42)

# Division 80/20
train_df = df.iloc[:int(0.8 * len(df))]
val_df = df.iloc[int(0.8 * len(df)):]

# --- 2. GÉNÉRATEURS (Streaming pour économiser la RAM) ---
# On utilise du 128x128 pour un bon compromis vitesse/précision sans modèle pré-entraîné
IMG_SIZE = 128
datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, brightness_range=[0.9, 1.1])

train_gen = datagen.flow_from_dataframe(train_df, x_col='path', y_col='age', 
                                        target_size=(IMG_SIZE, IMG_SIZE), batch_size=32, class_mode='raw')
val_gen = datagen.flow_from_dataframe(val_df, x_col='path', y_col='age', 
                                      target_size=(IMG_SIZE, IMG_SIZE), batch_size=32, class_mode='raw')

# --- 3. ARCHITECTURE RESNET CUSTOM ---
def residual_block(x, filters, kernel_size=3):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Add()([x, shortcut]) # Connexion résiduelle
    x = layers.Activation('relu')(x)
    return x

def build_custom_resnet(input_shape=(128, 128, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Entrée
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Blocs de convolution + Blocs résiduels
    for f in [64, 128, 256]:
        x = layers.Conv2D(f, (3, 3), strides=2, padding='same')(x) # Downsampling
        x = residual_block(x, f)
        x = layers.Dropout(0.2)(x)
    
    # Global Pooling au lieu de Flatten (réduit énormément le nombre de paramètres)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Tête de prédiction
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='linear')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(1e-3), loss='huber', metrics=['mae'])
    return model

model = build_custom_resnet()

# --- 4. ENTRAÎNEMENT ---
# Callbacks pour éviter le surapprentissage
early_stop = callbacks.EarlyStopping(patience=8, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(factor=0.5, patience=3)

history = model.fit(
    train_gen, 
    validation_data=val_gen, 
    epochs=40, 
    callbacks=[early_stop, reduce_lr]
)

# --- 5. VISUALISATION DES RÉSULTATS ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['mae'], label='MAE Train')
plt.plot(history.history['val_mae'], label='MAE Val')
plt.legend(); plt.title('Erreur Moyenne')

plt.subplot(1, 2, 2)
# Test sur quelques images
test_batch_x, test_batch_y = next(val_gen)
preds = model.predict(test_batch_x)

plt.imshow(test_batch_x[0])
plt.title(f"Réel: {test_batch_y[0]} | Prédit: {preds[0][0]:.1f}")
plt.axis('off')
plt.show()
