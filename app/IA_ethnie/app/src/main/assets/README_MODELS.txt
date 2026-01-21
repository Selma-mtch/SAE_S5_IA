MODELES TENSORFLOW LITE
=======================

Placez vos fichiers .tflite dans ce dossier:

1. model_multitask.tflite    - Modèle multi-tâches (âge, genre, ethnicité)
2. model_ethnicity.tflite    - Modèle spécialisé ethnicité
3. model_age.tflite          - Modèle spécialisé âge
4. model_gender.tflite       - Modèle spécialisé genre

Pour convertir vos modèles Keras en TFLite, utilisez:

```python
import tensorflow as tf

# Charger le modèle Keras
model = tf.keras.models.load_model('mon_model.h5')

# Convertir en TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Sauvegarder
with open('mon_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

Format d'entrée attendu:
- Taille: 128x128 pixels
- Canaux: 1 (niveaux de gris)
- Normalisation: [0, 1]

Sorties (modèle multi-tâches):
- Âge: float (régression)
- Genre: 2 classes (Homme, Femme)
- Ethnicité: 5 classes (Blanc, Noir, Asiatique, Indien, Autre)
