# 📱 Application de Reconnaissance Faciale Android

Application mobile Android native utilisant l'apprentissage automatique pour la reconnaissance faciale en temps réel.

## Fonctionnalités

- Reconnaissance faciale en temps réel via la caméra
- Détection de visages avec ML Kit de Google
- Classification avec modèle TensorFlow Lite personnalisé
- Interface utilisateur intuitive
- Support caméra avant/arrière
- Fonctionnement 100% hors ligne

## 🛠️ Technologies utilisées

### Android
- **Langage** : Kotlin
- **Minimum SDK** : 24 (Android 7.0)
- **Target SDK** : 34 (Android 14)
- **Architecture** : Native Android avec View Binding

### Machine Learning
- **TensorFlow Lite** : 2.16.1
- **ML Kit Face Detection** : 16.1.6
- **Modèle** : Classificateur d'embeddings (512D → 2 classes)
- **Précision** : 98.11% sur validation

### Caméra
- **CameraX** : 1.3.1
- **Format** : YUV_420_888
- **Résolution** : Adaptative

## Installation

### Prérequis
- Android Studio Hedgehog | 2023.1.1 ou supérieur
- JDK 17
- Android SDK 34
- Python 3.8+ (pour la conversion du modèle)

### Étapes

1. **Cloner le dépôt**
```bash
git clone https://github.com/VOTRE_USERNAME/face-recognition-android.git
cd face-recognition-android
```

2. **Convertir votre modèle d'embeddings**

   **Option A : Format ONNX** (recommandé, compatible Python 3.13)
   ```bash
   # Installer les dépendances Python
   pip install scikit-learn numpy skl2onnx onnxruntime

   # Placer votre face_model.pkl dans le dossier racine
   # Puis convertir
   python convert_model_to_onnx.py
   ```

   **Option B : Format TensorFlow Lite** (nécessite Python 3.11 ou inférieur)
   ```bash
   # Installer les dépendances Python
   pip install tensorflow scikit-learn numpy

   # Placer votre face_model.pkl dans le dossier racine
   # Puis convertir
   python convert_model_to_tflite.py
   ```

3. **Ouvrir dans Android Studio**
   - File → Open → Sélectionner le dossier `android/`
   - Attendre la synchronisation Gradle

4. **Compiler et exécuter**
   - Connecter un appareil Android ou lancer un émulateur
   - Cliquer sur ▶️ Run

## 📁 Structure du projet

```
.
├── android/                          # Application Android
│   ├── app/
│   │   ├── src/main/
│   │   │   ├── java/com/example/facerecognition/
│   │   │   │   ├── MainActivity.kt          # Activité principale
│   │   │   │   └── ml/
│   │   │   │       ├── FaceRecognitionModel.kt   # Modèle TFLite
│   │   │   │       └── FaceDetector.kt           # Détection ML Kit
│   │   │   ├── res/                  # Ressources UI
│   │   │   └── assets/               # Modèles ML
│   │   │       ├── face_recognition_model.tflite
│   │   │       └── face_recognition_metadata.json
│   │   └── build.gradle.kts          # Dépendances app
│   └── build.gradle.kts              # Configuration projet
├── convert_model_to_onnx.py          # Script conversion ONNX (recommandé)
├── convert_model_to_tflite.py        # Script conversion TFLite
├── face_model.pkl                    # Votre modèle d'embeddings (non inclus)
└── README.md
```

## 🔧 Configuration du modèle

**Script ONNX** (`convert_model_to_onnx.py`, recommandé) :
1. Charge les embeddings depuis `face_model.pkl`
2. Entraîne un classificateur scikit-learn (MLPClassifier) :
   - StandardScaler → MLP(256→128→64) → Softmax
3. Convertit en ONNX via skl2onnx
4. Génère les métadonnées JSON
5. **Précision typique : 97-98%**

**Script TensorFlow Lite** (`convert_model_to_tflite.py`) :
1. Charge les embeddings depuis `face_model.pkl`
2. Entraîne un classificateur Keras :
   - BatchNorm → Dense(256) → Dropout → Dense(128) → Dropout → Dense(64) → Softmax
3. Convertit en TensorFlow Lite avec optimisations
4. Génère les métadonnées JSON

### Format du face_model.pkl

```python
[
    {
        'embedding': np.array([512 dimensions]),
        'label': 'nom_personne'
    },
    ...
]
```

## 📱 Utilisation de l'application

1. **Lancer l'app** : L'application démarre avec la caméra avant
2. **Capturer** : Appuyez sur le bouton appareil photo
3. **Reconnaissance** : Le résultat s'affiche instantanément
4. **Changer de caméra** : Utilisez le bouton flip (🔄)

## 🎓 Apprentissage du modèle

Le modèle a été entraîné sur :
- **527 embeddings** (170 david, 357 manoah)
- **Split** : 80% train / 20% validation
- **Epochs** : ~30 avec early stopping
- **Accuracy finale** : 98.11%

## 🔒 Permissions

L'application nécessite :
- `CAMERA` : Accès à la caméra pour la capture

## 🐛 Dépannage

### Le modèle ne charge pas
- Vérifiez que `face_recognition_model.tflite` existe dans `android/app/src/main/assets/`
- Vérifiez les logs Logcat pour les erreurs TFLite

### Erreur "FULLY_CONNECTED version 12"
- Le modèle a été généré avec TensorFlow 2.20+ mais TFLite 2.16.1 ne supporte pas
- Reconvertir avec : `python convert_model_to_tflite.py`

### Caméra ne démarre pas
- Vérifier les permissions dans les paramètres Android
- Tester sur un appareil physique (certains émulateurs ont des problèmes)

## 📄 Licence

Ce projet est fourni à des fins éducatives.

## 👨‍💻 Auteur
David Mwangaza & NGOY Manoah

Projet de reconnaissance faciale Android avec TensorFlow Lite

## 🙏 Remerciements

- TensorFlow Lite pour l'inférence mobile
- ML Kit pour la détection de visages
- CameraX pour la gestion moderne de la caméra
