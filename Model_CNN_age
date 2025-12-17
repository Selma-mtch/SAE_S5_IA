# ==================== IMPORTS ====================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import kagglehub

# ==================== TÉLÉCHARGEMENT DU DATASET ====================
print("Téléchargement du dataset UTKFace depuis Kaggle...")
path = kagglehub.dataset_download("jangedoo/utkface-new")
print("Path to dataset files:", path)

# ==================== CHARGEMENT DES DONNÉES ====================
def load_utkface_data(data_dir, img_size=(128, 128), max_samples=None):
    """
    Charge le jeu de données UTKFace
    Format du nom de fichier: [age]_[gender]_[race]_[date&time].jpg
    """
    images = []
    ages = []
    
    files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if max_samples:
        files = files[:max_samples]
    
    print(f"Chargement de {len(files)} images...")
    
    for i, filename in enumerate(files):
        if i % 1000 == 0:
            print(f"Progression: {i}/{len(files)}")
        
        try:
            # Extraire l'âge du nom de fichier
            age = int(filename.split('_')[0])
            
            # Filtrer les âges aberrants
            if age < 0 or age > 116:
                continue
            
            # Charger et redimensionner l'image
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            
            images.append(img)
            ages.append(age)
        except Exception as e:
            continue
    
    images = np.array(images, dtype='float32') / 255.0
    ages = np.array(ages, dtype='float32')
    
    print(f"\n✓ {len(images)} images chargées avec succès!")
    
    return images, ages

# ==================== VISUALISATION ====================
def visualize_samples(X, y, n_samples=10):
    """
    Visualise quelques échantillons du dataset
    """
    plt.figure(figsize=(15, 3))
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[idx])
        plt.title(f"Âge: {int(y[idx])} ans")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ==================== ARCHITECTURE CNN ====================
def create_age_cnn(input_shape=(128, 128, 3)):
    """
    Crée un modèle CNN pour la prédiction d'âge
    """
    model = models.Sequential([
        # Bloc 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloc 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloc 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloc 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Couches denses
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Sortie (régression pour l'âge)
        layers.Dense(1, activation='linear')
    ])
    
    return model

# ==================== VISUALISATION DE L'ENTRAÎNEMENT ====================
def plot_training_history(history):
    """
    Visualise l'historique d'entraînement
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # MAE
    axes[0].plot(history.history['mae'], label='Train MAE')
    axes[0].plot(history.history['val_mae'], label='Val MAE')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MAE (années)')
    axes[0].set_title('Mean Absolute Error')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# ==================== ENTRAÎNEMENT ====================
def train_model(data_dir, epochs=50, batch_size=32, img_size=(128, 128)):
    """
    Entraîne le modèle CNN sur UTKFace
    """
    # Vérifier le GPU
    print("GPU disponible:", tf.config.list_physical_devices('GPU'))
    
    print("\n" + "="*50)
    print("CHARGEMENT DES DONNÉES")
    print("="*50)
    X, y = load_utkface_data(data_dir, img_size=img_size)
    
    print(f"\n📊 Statistiques du dataset:")
    print(f"   • Nombre d'images: {len(X)}")
    print(f"   • Âge min: {y.min():.0f} ans")
    print(f"   • Âge max: {y.max():.0f} ans")
    print(f"   • Âge moyen: {y.mean():.2f} ans")
    print(f"   • Écart-type: {y.std():.2f} ans")
    
    # Visualiser quelques échantillons
    print("\n📸 Visualisation d'échantillons:")
    visualize_samples(X, y)
    
    # Division train/validation/test
    print("\n" + "="*50)
    print("DIVISION DES DONNÉES")
    print("="*50)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"   • Train: {len(X_train)} images")
    print(f"   • Validation: {len(X_val)} images")
    print(f"   • Test: {len(X_test)} images")
    
    # Création du modèle
    print("\n" + "="*50)
    print("CRÉATION DU MODÈLE")
    print("="*50)
    model = create_age_cnn(input_shape=(*img_size, 3))
    
    # Compilation
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mae',
        metrics=['mae', 'mse']
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'best_age_model.h5',
            monitor='val_mae',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Augmentation des données
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2]
    )
    
    # Entraînement
    print("\n" + "="*50)
    print("ENTRAÎNEMENT")
    print("="*50)
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Visualiser l'historique
    print("\n📈 Historique d'entraînement:")
    plot_training_history(history)
    
    # Évaluation
    print("\n" + "="*50)
    print("ÉVALUATION SUR LE TEST SET")
    print("="*50)
    test_results = model.evaluate(X_test, y_test, verbose=0)
    print(f"   • Test MAE: {test_results[1]:.2f} ans")
    print(f"   • Test MSE: {test_results[2]:.2f}")
    print(f"   • Test RMSE: {np.sqrt(test_results[2]):.2f} ans")
    
    # Prédictions exemples avec visualisation
    print("\n🎯 Exemples de prédictions:")
    n_examples = 10
    indices = np.random.choice(len(X_test), n_examples, replace=False)
    y_pred = model.predict(X_test[indices], verbose=0)
    
    plt.figure(figsize=(15, 6))
    for i in range(n_examples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[indices[i]])
        real_age = int(y_test[indices[i]])
        pred_age = int(y_pred[i][0])
        error = abs(real_age - pred_age)
        color = 'green' if error <= 5 else 'orange' if error <= 10 else 'red'
        plt.title(f"Réel: {real_age}\nPrédit: {pred_age}\nErreur: {error}", 
                 color=color, fontsize=9)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return model, history

# ==================== EXÉCUTION ====================
# Le dataset est automatiquement téléchargé dans 'path'
# Trouver le dossier contenant les images
DATA_DIR = path
if os.path.isdir(os.path.join(path, 'UTKFace')):
    DATA_DIR = os.path.join(path, 'UTKFace')

print(f"\nUtilisation du dossier: {DATA_DIR}")
print(f"Nombre de fichiers: {len(os.listdir(DATA_DIR))}")

# Lancer l'entraînement
model, history = train_model(
    DATA_DIR, 
    epochs=50, 
    batch_size=32,
    img_size=(128, 128)
)

print("\n" + "="*50)
print("✓ ENTRAÎNEMENT TERMINÉ!")
print("="*50)
