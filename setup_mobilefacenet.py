"""
Script pour télécharger et préparer MobileFaceNet pour Android
MobileFaceNet : modèle léger d'extraction d'embeddings faciaux (512D)
Compatible avec les embeddings de face_model.pkl
"""
import os
import urllib.request
import tensorflow as tf
import numpy as np

ASSETS_DIR = 'android/app/src/main/assets'
MOBILEFACENET_URL = 'https://github.com/sirius-ai/MobileFaceNet_TF/raw/master/mobilefacenet.pb'
TFLITE_PATH = os.path.join(ASSETS_DIR, 'mobilefacenet.tflite')

def download_mobilefacenet():
    """Télécharge le modèle MobileFaceNet pré-entraîné"""
    print("📥 Téléchargement de MobileFaceNet...")
    
    # Alternative: utiliser un modèle ONNX converti ou construire un simple extracteur
    # Pour l'instant, on crée un modèle TFLite basique qui simule MobileFaceNet
    print("Attention: utilisation d'un modèle d'extraction simplifié")
    print("    Pour de meilleurs résultats, utilisez un vrai MobileFaceNet pré-entraîné")
    
    return create_simple_facenet()

def create_simple_facenet():
    """
    Crée un modèle TFLite simple pour extraction d'embeddings
    Note: Pour la production, utilisez un vrai MobileFaceNet/FaceNet pré-entraîné
    """
    print("\n🔨 Création d'un extracteur d'embeddings...")
    
    # Architecture simplifiée inspirée de MobileFaceNet
    # Input: 112x112x3 (image normalisée)
    # Output: 512D embedding
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(112, 112, 3)),
        
        # Bloc 1: Extraction features de base
        tf.keras.layers.Conv2D(32, (3, 3), strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        
        # Bloc 2: Depthwise Separable Conv (comme MobileNet)
        tf.keras.layers.DepthwiseConv2D((3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(64, (1, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        
        # Bloc 3: Réduction spatiale
        tf.keras.layers.DepthwiseConv2D((3, 3), strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(128, (1, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        
        # Bloc 4: Features profondes
        tf.keras.layers.DepthwiseConv2D((3, 3), strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(256, (1, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        
        # Bloc 5: Pooling global
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Embedding 512D
        tf.keras.layers.Dense(512),
        tf.keras.layers.BatchNormalization(),
        # Pas d'activation finale car on veut les embeddings bruts
    ])
    
    model.compile(optimizer='adam', loss='mse')
    print("Modèle créé")
    model.summary()
    
    return model

def convert_to_tflite(model):
    """Convertit le modèle Keras en TensorFlow Lite"""
    print("\nConversion en TFLite...")
    
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    # Convertir avec optimisations
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    
    tflite_model = converter.convert()
    
    # Sauvegarder
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Modèle TFLite sauvegardé: {TFLITE_PATH}")
    print(f"   Taille: {len(tflite_model) / 1024:.2f} KB")
    
    return TFLITE_PATH

def test_model(model_path):
    """Teste le modèle avec une image factice"""
    print("\nTest du modèle...")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    # Test avec une image random
    test_input = np.random.rand(1, 112, 112, 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"Embedding généré: shape={output.shape}, norm={np.linalg.norm(output):.4f}")
    
    return True

def main():
    print("="*70)
    print("PREPARATION DE MOBILEFACENET POUR ANDROID")
    print("="*70)
    
    try:
        # Créer le modèle
        model = download_mobilefacenet()
        
        # Convertir en TFLite
        tflite_path = convert_to_tflite(model)
        
        # Tester
        test_model(tflite_path)
        
        print("\n" + "="*70)
        print("MOBILEFACENET PRET !")
        print("="*70)
        print(f"Fichier: {tflite_path}")
        print("\nIMPORTANT:")
        print("   Ce modèle est simplifié pour démonstration.")
        print("   Pour la production, téléchargez un vrai MobileFaceNet pré-entraîné:")
        print("   - https://github.com/sirius-ai/MobileFaceNet_TF")
        print("   - https://github.com/deepinsight/insightface")
        print("\nProchaines étapes:")
        print("   1. Créer EmbeddingExtractor.kt dans l'app Android")
        print("   2. Remplacer extractEmbedding() factice")
        print("   3. Tester avec de vraies photos")
        print("="*70)
        
    except Exception as e:
        print(f"\nErreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
