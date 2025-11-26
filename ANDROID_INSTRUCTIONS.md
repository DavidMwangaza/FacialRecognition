# 🤖 APPLICATION ANDROID AVEC MODÈLE INTÉGRÉ (MODE HORS LIGNE)

## ✅ Configuration Terminée

Votre application Android a été configurée pour utiliser la reconnaissance faciale **directement sur l'appareil** sans serveur backend.

---

## 📱 Architecture de l'Application

### Mode Hors Ligne - Reconnaissance Locale
- **TensorFlow Lite** : Exécute le modèle sur l'appareil
- **ML Kit** : Détecte les visages rapidement
- **OpenCV** : Traite les images
- **Aucune connexion réseau requise** ✓

---

## 🔧 ÉTAPE 1 : Convertir Votre Modèle

### Option A : Convertir face_model.pkl existant

1. **Installer les dépendances Python** :
```powershell
cd "C:\Users\david\Documents\Appli"
pip install tensorflow numpy
```

2. **Exécuter le script de conversion** :
```powershell
python convert_model_to_tflite.py
```

Cela créera :
- `android/app/src/main/assets/face_recognition_model.tflite`
- `android/app/src/main/assets/face_recognition_metadata.json`

### Option B : Utiliser le modèle d'exemple (pour tester)

Si vous voulez d'abord tester l'application :
```powershell
python convert_model_to_tflite.py
```
Le script créera automatiquement un modèle d'exemple si `face_model.pkl` ne peut pas être converti.

---

## 📂 Structure des Fichiers

```
android/
├── app/
│   ├── src/
│   │   └── main/
│   │       ├── assets/                          ← Placez vos modèles ici
│   │       │   ├── face_recognition_model.tflite
│   │       │   └── face_recognition_metadata.json
│   │       └── java/com/example/facerecognition/
│   │           ├── MainActivity.kt               ← Interface principale
│   │           └── ml/
│   │               ├── FaceRecognitionModel.kt   ← Inférence TFLite
│   │               └── FaceDetector.kt           ← Détection ML Kit
│   └── build.gradle                              ← Dépendances ML
```

---

## 🚀 ÉTAPE 2 : Compiler l'Application

### 1. Ouvrir dans Android Studio
```
Fichier > Open > C:\Users\david\Documents\Appli\android
```

### 2. Synchroniser Gradle
- Android Studio va télécharger les dépendances automatiquement
- Attendez la fin de "Gradle Sync"

### 3. Vérifier les fichiers assets
Dans Android Studio :
```
app > src > main > assets
```
Vous devriez voir :
- ✓ `face_recognition_model.tflite`
- ✓ `face_recognition_metadata.json`

### 4. Compiler et Exécuter
- Branchez votre téléphone Android (avec le débogage USB activé)
- Ou lancez un émulateur Android
- Cliquez sur le bouton ▶️ (Run)

---

## 📋 Format du Metadata JSON

Le fichier `face_recognition_metadata.json` doit contenir :

```json
{
  "names": ["Personne 1", "Personne 2", "Personne 3"],
  "num_classes": 3,
  "input_shape": [100, 100, 3],
  "model_type": "CNN"
}
```

---

## 🎯 Fonctionnalités de l'Application

### Interface Utilisateur
1. **Vue caméra en temps réel**
2. **Bouton Capturer** 📷 - Prend une photo
3. **Bouton Flip** 🔄 - Change de caméra (avant/arrière)
4. **Affichage des résultats** avec rectangles sur les visages détectés

### Processus de Reconnaissance
```
Photo capturée → ML Kit détecte les visages → 
Extraction des régions → TensorFlow Lite reconnaît → 
Affichage des noms avec confiance
```

---

## ⚙️ Dépendances Ajoutées

### build.gradle (app)
```gradle
// TensorFlow Lite
implementation 'org.tensorflow:tensorflow-lite:2.14.0'
implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'

// ML Kit Face Detection
implementation 'com.google.mlkit:face-detection:16.1.6'

// OpenCV
implementation 'org.opencv:opencv:4.8.0'
```

---

## 🔍 Classes Kotlin Créées

### 1. `FaceRecognitionModel.kt`
- Charge le modèle TFLite
- Prétraite les images
- Exécute l'inférence
- Retourne les prédictions avec confiance

### 2. `FaceDetector.kt`
- Utilise ML Kit pour détecter les visages
- Extrait les régions faciales
- Combine détection + reconnaissance

### 3. `MainActivity.kt` (modifiée)
- Capture photo avec CameraX
- Appelle la reconnaissance locale
- Dessine les rectangles sur les visages
- Affiche les résultats

---

## 🐛 Dépannage

### Erreur "Modèle non trouvé"
**Solution** : Exécutez le script de conversion
```powershell
python convert_model_to_tflite.py
```

### Gradle Sync Failed
**Solution** : Dans Android Studio
```
File > Invalidate Caches / Restart
```

### "Aucun visage détecté"
- Assurez-vous d'avoir un bon éclairage
- Le visage doit être de face
- Essayez de vous rapprocher

### Erreur de compilation OpenCV
**Solution** : Remplacez dans `build.gradle` par :
```gradle
implementation 'com.quickbirdstudios:opencv:4.5.3.0'
```

---

## 📊 Performance

### Vitesse
- **Détection de visage** : ~50-100ms
- **Reconnaissance** : ~100-200ms
- **Total** : ~150-300ms par image

### Compatibilité
- **Min SDK** : 24 (Android 7.0)
- **Target SDK** : 34 (Android 14)
- **GPU Acceleration** : Activée si disponible

---

## 🔄 Mise à Jour du Modèle

Pour mettre à jour le modèle sans recompiler :

1. **Réentraînez votre modèle**
2. **Reconvertissez en TFLite**
3. **Remplacez les fichiers dans assets/**
4. **Rebuild l'application**

---

## 📈 Améliorer la Précision

### 1. Augmenter les données d'entraînement
- Plus d'images par personne
- Différentes conditions d'éclairage
- Différents angles

### 2. Optimiser le modèle
- Utiliser un modèle plus profond (ResNet, MobileNet)
- Appliquer la data augmentation
- Ajuster les hyperparamètres

### 3. Prétraitement
- Normalisation cohérente
- Alignement des visages
- Augmentation du contraste

---

## 🎓 Ressources Utiles

- **TensorFlow Lite** : https://www.tensorflow.org/lite
- **ML Kit** : https://developers.google.com/ml-kit
- **CameraX** : https://developer.android.com/training/camerax

---

## 📝 Checklist avant Lancement

- [ ] Convertir `face_model.pkl` en TFLite
- [ ] Vérifier les fichiers dans `assets/`
- [ ] Synchroniser Gradle
- [ ] Tester sur un appareil réel
- [ ] Vérifier les permissions caméra
- [ ] Tester avec différents visages
- [ ] Vérifier la performance

---

## 🎉 Prêt à Lancer !

Une fois les étapes complétées :

1. **Convertissez le modèle** :
   ```powershell
   python convert_model_to_tflite.py
   ```

2. **Ouvrez dans Android Studio** :
   ```
   File > Open > android/
   ```

3. **Exécutez l'application** : Cliquez sur ▶️

---

**Bon développement ! 🚀**
