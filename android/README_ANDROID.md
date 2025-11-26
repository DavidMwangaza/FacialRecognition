# Application Mobile Android de Reconnaissance Faciale

Application Android native développée avec Kotlin pour la reconnaissance faciale en temps réel.

## 📁 Structure du Projet

```
android/
├── app/
│   ├── build.gradle                    # Configuration de l'application
│   ├── proguard-rules.pro             # Règles de minification
│   └── src/main/
│       ├── AndroidManifest.xml         # Manifeste de l'application
│       ├── java/com/example/facerecognition/
│       │   ├── MainActivity.kt         # Activité principale avec caméra
│       │   ├── ApiService.kt          # Interface API Retrofit
│       │   └── RetrofitClient.kt      # Client HTTP
│       └── res/
│           ├── layout/
│           │   └── activity_main.xml   # Layout de l'interface
│           ├── values/
│           │   ├── colors.xml         # Couleurs de l'application
│           │   ├── strings.xml        # Chaînes de texte
│           │   └── themes.xml         # Thèmes
│           └── xml/
│               ├── backup_rules.xml
│               └── data_extraction_rules.xml
├── build.gradle                        # Configuration Gradle projet
├── settings.gradle                     # Configuration modules
└── gradle.properties                   # Propriétés Gradle
```

## 🚀 Ouverture dans Android Studio

### 1. Ouvrir le projet
1. Lancez **Android Studio**
2. Cliquez sur **File** → **Open**
3. Sélectionnez le dossier `C:\Users\david\Documents\Appli\android`
4. Cliquez sur **OK**

### 2. Synchronisation Gradle
Android Studio va automatiquement synchroniser les dépendances. Si ce n'est pas le cas :
- Cliquez sur **File** → **Sync Project with Gradle Files**

### 3. Configuration de l'URL du Backend
Ouvrez `RetrofitClient.kt` et vérifiez/modifiez l'adresse IP :
```kotlin
private const val BASE_URL = "http://10.131.82.86:5000/"
```

## 📱 Exécution sur un Appareil

### Option A : Appareil Physique (Recommandé)

1. **Activer le mode développeur** sur votre téléphone Android :
   - Allez dans **Paramètres** → **À propos du téléphone**
   - Appuyez 7 fois sur **Numéro de build**

2. **Activer le débogage USB** :
   - Allez dans **Paramètres** → **Options de développeur**
   - Activez **Débogage USB**

3. **Connecter votre téléphone** :
   - Branchez votre téléphone via USB
   - Acceptez l'autorisation de débogage sur le téléphone

4. **Lancer l'application** :
   - Dans Android Studio, sélectionnez votre appareil dans la liste déroulante
   - Cliquez sur le bouton **Run** (▶️) ou appuyez sur **Shift + F10**

### Option B : Émulateur Android

1. **Créer un émulateur** :
   - Cliquez sur **Tools** → **Device Manager**
   - Cliquez sur **Create Device**
   - Sélectionnez un appareil (ex: Pixel 6)
   - Téléchargez une image système (Android 11+)
   - Nommez votre émulateur et créez-le

2. **Configurer la caméra** :
   - Dans la configuration de l'émulateur
   - Activez la caméra virtuelle

3. **Modifier l'URL** dans `RetrofitClient.kt` :
   ```kotlin
   private const val BASE_URL = "http://10.0.2.2:5000/"
   ```
   Note : `10.0.2.2` est l'adresse localhost depuis l'émulateur

4. **Lancer l'émulateur** :
   - Sélectionnez l'émulateur dans Android Studio
   - Cliquez sur **Run** (▶️)

## 🔧 Dépendances Principales

### Android
- **minSdk**: 24 (Android 7.0)
- **targetSdk**: 34 (Android 14)
- **Kotlin**: 1.9.20

### Bibliothèques
- **CameraX**: Gestion moderne de la caméra
- **Retrofit**: Client HTTP pour API REST
- **Gson**: Sérialisation/désérialisation JSON
- **Material Design**: Interface moderne
- **Coroutines**: Programmation asynchrone

## 🎯 Fonctionnalités

### Caméra
- ✅ Aperçu en temps réel
- ✅ Capture photo
- ✅ Basculement caméra avant/arrière
- ✅ Gestion automatique des permissions

### Reconnaissance
- ✅ Envoi image vers API Flask
- ✅ Détection de multiples visages
- ✅ Affichage des noms et confiance
- ✅ Indicateur de chargement

### Interface
- ✅ Design moderne Material Design
- ✅ Mode sombre
- ✅ Animations fluides
- ✅ Messages d'erreur clairs

## 📋 Permissions Nécessaires

L'application demande les permissions suivantes :

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />
```

## 🔐 Configuration Réseau

### WiFi
Assurez-vous que :
- Votre téléphone et PC sont sur le même réseau WiFi
- Le serveur Flask est actif
- L'adresse IP dans `RetrofitClient.kt` est correcte

### Vérifier la connexion
L'application teste automatiquement la connexion au démarrage.
Si vous voyez "✓ Serveur connecté", tout est OK !

## 🐛 Dépannage

### Erreur de build
```bash
# Nettoyer et rebuilder
./gradlew clean
./gradlew build
```

### Permissions caméra refusées
Allez dans **Paramètres** → **Applications** → **Reconnaissance Faciale** → **Permissions**
et accordez l'accès à la caméra.

### Impossible de se connecter au serveur
1. Vérifiez que le backend Flask est actif
2. Testez l'URL dans un navigateur : `http://10.131.82.86:5000/health`
3. Vérifiez le pare-feu Windows
4. Essayez de désactiver temporairement l'antivirus

### L'émulateur ne démarre pas
- Activez la virtualisation dans le BIOS (Intel VT-x ou AMD-V)
- Installez Intel HAXM (Hardware Accelerated Execution Manager)

## 📊 Tests

### Tester l'API manuellement
Dans Android Studio, ouvrez le **Logcat** pour voir les logs :
```
View → Tool Windows → Logcat
```

Filtrez par "FaceRecognition" pour voir les logs de l'application.

## 🚀 Build de Production

### Générer un APK
```bash
# APK Debug
./gradlew assembleDebug

# APK Release (nécessite une clé de signature)
./gradlew assembleRelease
```

L'APK sera dans : `app/build/outputs/apk/`

### Installer l'APK
```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

## 📝 Notes de Développement

### Architecture
- **MVVM** : Architecture recommandée (peut être ajoutée)
- **Coroutines** : Pour les opérations asynchrones
- **ViewBinding** : Pour l'accès aux vues

### Améliorations Possibles
- [ ] Ajout de ViewModel
- [ ] Repository pattern
- [ ] Persistance locale (Room)
- [ ] Sélection d'images depuis la galerie
- [ ] Historique des reconnaissances
- [ ] Mode batch (plusieurs photos)
- [ ] Export des résultats

## 🔗 URLs de Test

### Vérifier le serveur
```
http://10.131.82.86:5000/health
```

### Test avec Postman
```
POST http://10.131.82.86:5000/recognize
Body: {"image": "base64_encoded_image"}
```

## 📚 Documentation Complémentaire

- [CameraX Documentation](https://developer.android.com/training/camerax)
- [Retrofit Documentation](https://square.github.io/retrofit/)
- [Material Design Guidelines](https://material.io/design)

---

**Prêt à compiler et exécuter ! 🎉**

Ouvrez le projet dans Android Studio et appuyez sur Run !
