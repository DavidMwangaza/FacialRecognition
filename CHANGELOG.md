# CHANGELOG

## 2025-11-29 — Intégration MobileFaceNet et nettoyage

- **Ajout**: `setup_mobilefacenet.py`
  - Génère `android/app/src/main/assets/mobilefacenet.tflite` (196.66 KB)
  - Modèle CNN léger (input: 112×112×3, output: 512D)
  - Testé avec TensorFlow Lite Interpreter (shape in/out OK)

- **Ajout**: `EmbeddingExtractor.kt`
  - Chargement du modèle TFLite et extraction d’embeddings 512D réels
  - Normalisation L2, prétraitement `(pixel-127.5)/127.5`, GPU si dispo
  - Méthode `extract(Bitmap): FloatArray?`

- **Modification**: `FaceRecognitionModel.kt`
  - Intégration de `EmbeddingExtractor` pour remplacer l’extraction factice
  - `extractEmbedding(Bitmap)` devient nullable et **ne fait plus de fallback**
  - `recognize(Bitmap)` gère l’absence d’embedding proprement
  - Nettoyage des ressources (`close()` libère modèle + extracteur)

- **Ajout**: `MOBILEFACENET_INTEGRATION.md`
  - Documentation des changements, architecture, et instructions de test

- **Statut build**:
  - Gradle wrapper absent en local; build recommandé via Android Studio

## 2025-11-29 — Conversion modèle en ONNX

- **Ajout**: `convert_model_to_onnx.py`
  - Conversion `face_model.pkl` vers `face_recognition_model.onnx` (680.85 KB)
  - Pipeline scikit-learn (`StandardScaler` + `MLPClassifier`), `opset=13`
  - Compatible Python 3.13 (sans TensorFlow)

## 2025-11-26 — Conversion TFLite et setup Android

- **Ajout**: `convert_model_to_tflite.py` (ancienne version et version stricte)
  - Conversion vers `face_recognition_model.tflite` (677 KB)
  - Upgrade TFLite vers 2.16.1 (compatibilité ops FC v12)

- **Setup Android**:
  - Projet Kotlin natif (CameraX, ML Kit Face Detection, TFLite)
  - Corrections de build (icônes, dépendances, compatibilité)

---

### Prochaines étapes
- Build avec Android Studio et tester sur device (david/manoah)
- (Optionnel) Remplacer `mobilefacenet.tflite` par un modèle pré-entraîné officiel
- Ajouter `CHANGELOG.md` au dépôt et pousser sur GitHub
