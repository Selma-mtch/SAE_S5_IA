# FaceScan - Analyse Faciale par Intelligence Artificielle

Application Android qui predit l'age, le genre et l'ethnicite a partir de visages en utilisant des modeles de deep learning entraines sur le dataset UTKFace.

## Fonctionnalites

- **Photo** : Capture ou import depuis la galerie
- **Temps reel** : Analyse continue sur le flux camera (~5 fps)
- **Detection de visage** : Localisation automatique du visage via Google ML Kit
- **3 modes de prediction** : Modeles specialises, multi-taches ou transfer learning
- **Historique** : Sauvegarde des analyses dans Firebase
- **Authentification** : Inscription et connexion via Firebase

## Modeles

| Modele | Input | Sorties |
|--------|-------|---------|
| Multi-taches | 128x128 RGB | age + genre + ethnicite |
| Transfer (MobileNetV2) | 128x128 RGB | age + genre + ethnicite |
| Age specialise | 128x128 RGB | age |
| Genre specialise | 128x128 grayscale | genre |
| Ethnicite specialise | 128x128 grayscale | ethnicite |

Classes d'ethnicite : Blanc, Noir, Asiatique, Indien, Autre

L'application detecte automatiquement le format d'entree (grayscale/RGB) et le type de sortie (sigmoid/softmax) de chaque modele.

## Technologies

- **Android** : Java, CameraX, Material Design 3, View Binding
- **IA** : TensorFlow Lite, Google ML Kit Face Detection
- **Backend** : Firebase Authentication, Firestore
- **Entrainement** : TensorFlow / Keras, dataset UTKFace (23 000+ images)

## Architecture

```
SAE_S5_IA/
├── app/IA_ethnie/                          # Application Android
│   ├── app/src/main/
│   │   ├── java/com/example/ia_ethnie/
│   │   │   ├── ui/
│   │   │   │   ├── auth/                   # Login, Register
│   │   │   │   ├── main/                   # Hub principal
│   │   │   │   ├── camera/                 # Capture et temps reel
│   │   │   │   ├── result/                 # Affichage des predictions
│   │   │   │   ├── history/                # Historique des analyses
│   │   │   │   └── info/                   # A propos
│   │   │   ├── ml/
│   │   │   │   ├── FaceAnalyzer.java       # Inference TFLite
│   │   │   │   └── FaceDetectorHelper.java # Detection de visage
│   │   │   └── utils/
│   │   │       └── SessionManager.java
│   │   └── assets/                         # Modeles .tflite
│   └── build.gradle.kts
│
├── Model_ia/                               # Scripts d'entrainement Python
│   ├── Age/                                # Prediction d'age (CNN, MobileNetV2)
│   ├── Genre/                              # Classification du genre
│   ├── Model_CNN_ethnie/                   # Classification ethnicite (Focal Loss, augmentation ciblee)
│   ├── Modele general/                     # Modeles multi-taches + export TFLite
│   └── modele-par-transfert-de-connaissances/  # Transfer learning MobileNetV2
│
└── README.md
```

## Flux de l'Application

```
Login → MainActivity → Camera → ML Kit detecte le visage → crop → TFLite → Resultats
                     → Temps reel → ML Kit + TFLite en continu → overlay
                     → Galerie → Resultats
                     → Historique (Firestore)
                     → Parametres (choix du modele)
```

## Installation

### Prerequis
- Android Studio (2023.1.1+)
- JDK 11+
- Appareil Android API 24+ ou emulateur

### Etapes

1. Cloner le repository
   ```bash
   git clone https://github.com/Selma-mtch/SAE_S5_IA.git
   cd SAE_S5_IA
   ```

2. Ouvrir le dossier `app/IA_ethnie/` dans Android Studio et attendre la synchronisation Gradle

3. Configurer Firebase :
   - Creer un projet sur [console.firebase.google.com](https://console.firebase.google.com)
   - Activer Authentication (Email/Password) et Firestore
   - Placer `google-services.json` dans `app/IA_ethnie/app/`

4. Build et run
   ```bash
   cd app/IA_ethnie
   ./gradlew assembleDebug
   ```

## Scripts d'Entrainement

Les scripts Python sont dans `Model_ia/` et s'executent sur Kaggle avec GPU.

| Script | Description |
|--------|-------------|
| `Age/Model_MobileNet_age.py` | Age par transfer learning MobileNetV2 |
| `Genre/modele_genre_v3_128_gris.py` | Genre CNN grayscale 128x128 |
| `Model_CNN_ethnie/model/model_ethnie_focal_augmentation_ciblee.py` | Ethnicite avec Focal Loss + augmentation ciblee |
| `Modele general/train_multitask_tflite.py` | Multi-taches avec export TFLite |
| `modele-par-transfert-de-connaissances/model_transfer_mobilenet.py` | Transfer learning MobileNetV2 multi-taches |

## Structure Firebase

```
users/{uid}
  - username, email, createdAt

predictions/{docId}
  - userId, age, gender, ethnicity
  - ageConfidence, genderConfidence, ethnicityConfidence
  - modelType (SPECIALIZED | MULTI_TASK | TRANSFER)
  - localImagePath, createdAt
```

## Equipe

Projet realise dans le cadre de la SAE S5 - Intelligence Artificielle.

## Licence

Projet a but educatif. Le dataset UTKFace est soumis a sa licence propre.
