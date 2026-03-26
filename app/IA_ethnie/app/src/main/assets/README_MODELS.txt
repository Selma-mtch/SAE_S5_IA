MODELES TENSORFLOW LITE
=======================

Placez vos fichiers .tflite dans ce dossier:

1. model_multitask.tflite    - Modele multi-taches (age, genre, ethnicite)
2. model_ethnicity.tflite    - Modele specialise ethnicite
3. model_age.tflite          - Modele specialise age
4. model_gender.tflite       - Modele specialise genre
5. model_transfer.tflite     - Modele par transfert (MobileNetV2)

Format d'entree par modele:
---------------------------

MULTI_TASK (model_multitask.tflite):
- Taille: 128x128 pixels
- Canaux: 1 (niveaux de gris)
- Normalisation: [0, 1]
- Sorties: age (float), genre (sigmoid 1 classe), ethnicite (softmax 5 classes)

SPECIALIZED:
- model_age.tflite:       128x128, 3 canaux (RGB), normalisation [-1, 1]
- model_gender.tflite:    128x128, 1 canal (niveaux de gris), normalisation [0, 1]
- model_ethnicity.tflite: 128x128, 1 canal (niveaux de gris), normalisation [0, 1]

TRANSFER (model_transfer.tflite):
- Taille: 128x128 pixels
- Canaux: 3 (RGB)
- Normalisation: [-1, 1] (MobileNetV2 preprocess_input)
- Sorties: age (float linear), genre (softmax 2 classes), ethnicite (softmax 5 classes)

Classes:
- Genre: 0=Homme, 1=Femme
- Ethnicite: 0=Blanc, 1=Noir, 2=Asiatique, 3=Indien, 4=Autre
