# SAE S5 - Application d'Analyse Faciale par Intelligence Artificielle

Application Android utilisant l'intelligence artificielle pour analyser des visages et prédire l'âge, le genre et l'ethnicité à partir d'images.

## Fonctionnalités

### Authentification
- Inscription avec nom d'utilisateur, email et mot de passe
- Connexion sécurisée via Firebase Authentication
- Gestion de session persistante

### Analyse Faciale
- **Capture photo** : Prise de photo via la caméra du téléphone
- **Import galerie** : Sélection d'une image depuis la galerie
- **Analyse en temps réel** : Analyse continue avec affichage en overlay sur le flux caméra
- **Changement de caméra** : Basculement entre caméra frontale et arrière

### Résultats
- Prédiction de l'âge (en années)
- Classification du genre (Homme/Femme)
- Classification de l'ethnicité (5 catégories : Blanc, Noir, Asiatique, Indien, Autre)
- Scores de confiance affichés en pourcentage
- Partage des résultats

### Historique
- Consultation de toutes les analyses précédentes
- Affichage de la date, du type de modèle et des attributs prédits
- Suppression individuelle ou globale de l'historique
- Aperçu des images analysées

### Modèles IA
Trois types de modèles disponibles :
1. **Modèles spécialisés** : 3 CNN séparés (âge, genre, ethnicité)
2. **Multi-task Learning** : Modèle unifié ResNet + SE-Net
3. **Transfer Learning** : Basé sur MobileNetV2/EfficientNetB0

## Technologies Utilisées

### Application Android
| Technologie | Utilisation |
|-------------|-------------|
| Java | Langage de programmation principal |
| Android SDK 24-34 | Compatibilité Android 7.0 à 14 |
| CameraX | API caméra moderne |
| Material Design 3 | Interface utilisateur |
| View Binding | Liaison des vues |
| Glide | Chargement et cache d'images |

### Backend & Base de données
| Technologie | Utilisation |
|-------------|-------------|
| Firebase Authentication | Authentification utilisateurs |
| Firebase Firestore | Base de données NoSQL cloud |
| Firebase Storage | Stockage de fichiers |

### Intelligence Artificielle
| Technologie | Utilisation |
|-------------|-------------|
| TensorFlow Lite | Inférence sur mobile |
| TensorFlow Lite GPU | Accélération GPU optionnelle |
| UTKFace Dataset | Dataset d'entraînement |

## Architecture du Projet

```
SAE_S5_IA/
├── app/IA_ethnie/                    # Projet Android principal
│   ├── app/src/main/
│   │   ├── java/com/example/ia_ethnie/
│   │   │   ├── ui/
│   │   │   │   ├── auth/             # LoginActivity, RegisterActivity
│   │   │   │   ├── main/             # MainActivity (hub principal)
│   │   │   │   ├── camera/           # CameraActivity (capture & temps réel)
│   │   │   │   ├── result/           # ResultActivity (affichage résultats)
│   │   │   │   ├── history/          # HistoryActivity (historique)
│   │   │   │   └── info/             # ModelInfoActivity (documentation)
│   │   │   ├── ml/
│   │   │   │   └── FaceAnalyzer.java # Moteur d'inférence ML
│   │   │   └── utils/
│   │   │       └── SessionManager.java # Gestion de session
│   │   ├── res/                      # Ressources (layouts, strings, etc.)
│   │   └── assets/                   # Modèles TFLite
│   └── build.gradle.kts              # Dépendances
│
├── Model_ia/                         # Scripts d'entraînement Python
│   ├── Age/                          # Modèles de prédiction d'âge
│   └── Model_CNN_ethnie/             # Modèles CNN pour ethnicité
│
└── SAE2 (1).ipynb                    # Notebook d'exploration de données
```

## Flux de l'Application

```
LoginActivity
    ↓ (connexion réussie)
MainActivity (Hub avec 6 cartes)
    ├→ CameraActivity (capture) → ResultActivity → sauvegarde Firestore
    ├→ CameraActivity (temps réel) → affichage overlay
    ├→ Sélection galerie → ResultActivity → sauvegarde Firestore
    ├→ HistoryActivity → récupération depuis Firestore
    ├→ ModelInfoActivity → documentation
    └→ Paramètres → sélection du type de modèle
```

## Installation

### Prérequis
- Android Studio Hedgehog (2023.1.1) ou supérieur
- JDK 11 ou supérieur
- Un appareil Android (API 24+) ou émulateur

### Configuration

1. **Cloner le repository**
   ```bash
   git clone https://github.com/votre-repo/SAE_S5_IA.git
   cd SAE_S5_IA
   ```

2. **Ouvrir dans Android Studio**
   - Ouvrir le dossier `app/IA_ethnie/` dans Android Studio
   - Attendre la synchronisation Gradle

3. **Configurer Firebase**
   - Créer un projet Firebase sur [console.firebase.google.com](https://console.firebase.google.com)
   - Activer Authentication (Email/Password)
   - Activer Firestore Database
   - Télécharger `google-services.json` et le placer dans `app/IA_ethnie/app/`

4. **Ajouter les modèles TFLite**
   - Placer les fichiers `.tflite` dans `app/src/main/assets/`
   - Fichiers requis :
     - `model_age.tflite`
     - `model_gender.tflite`
     - `model_ethnicity.tflite`
     - `model_multitask.tflite` (optionnel)
     - `model_transfer.tflite` (optionnel)

5. **Build et Run**
   ```bash
   ./gradlew assembleDebug
   ```
   Ou utiliser le bouton Run dans Android Studio

## Structure Firebase

### Collection `users`
```
users/{uid}
  - username: string
  - email: string
  - createdAt: timestamp
```

### Collection `predictions`
```
predictions/{docId}
  - userId: string
  - age: integer
  - gender: string
  - ethnicity: string
  - ageConfidence: float
  - genderConfidence: float
  - ethnicityConfidence: float
  - modelType: string (SPECIALIZED|MULTI_TASK|TRANSFER)
  - localImagePath: string
  - createdAt: timestamp
```

## Modèles d'IA

### Dataset
Les modèles sont entraînés sur le dataset **UTKFace** contenant plus de 20 000 images de visages annotées avec l'âge, le genre et l'ethnicité.

### Architectures

#### Modèles Spécialisés
- 3 CNN indépendants optimisés pour chaque tâche
- Input : Image grayscale 128x128 normalisée [0,1]

#### Multi-Task Learning
- Architecture ResNet avec blocs Squeeze-Excitation (SE-Net)
- Apprentissage simultané des 3 attributs
- Précision ethnicité : ~74%

#### Transfer Learning
- Base : MobileNetV2 ou EfficientNetB0 pré-entraîné sur ImageNet
- Fine-tuning sur UTKFace
- Optimisé pour l'efficacité mobile

### Entraînement
Les scripts Python pour l'entraînement sont dans `Model_ia/` :
```bash
cd Model_ia/Model_CNN_ethnie
python model_resnet_se.py  # Exemple
```

## Améliorations Futures

- [ ] Connexion OAuth (Google, Facebook)
- [ ] Réinitialisation de mot de passe
- [ ] Dashboard statistiques
- [ ] Export des données (CSV/PDF)
- [ ] Détection multi-visages
- [ ] Estimation des émotions
- [ ] Mode hors-ligne
- [ ] Support multilingue

## Équipe

Projet réalisé dans le cadre de la SAE S5 - Intelligence Artificielle.

## Licence

Ce projet est à but éducatif. Les modèles et le dataset UTKFace sont soumis à leurs licences respectives.

---

*Développé avec Android Studio et TensorFlow Lite*
