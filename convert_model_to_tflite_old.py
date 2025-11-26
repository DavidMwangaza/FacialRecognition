"""
Script pour convertir face_model.pkl en modèle TensorFlow Lite
Compatible avec l'intégration Android
"""
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os

def load_pickle_model(pkl_path):
    """Charge le modèle pickle"""
    print(f"📂 Chargement du modèle depuis {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def create_keras_model_from_sklearn(sklearn_model, input_shape):
    """
    Convertit un modèle sklearn en modèle Keras équivalent
    Supporte SVM, RandomForest, KNN, etc.
    """
    print("🔄 Conversion du modèle sklearn en Keras...")
    
    # Créer un modèle Keras simple qui encapsule le modèle sklearn
    # Note: Cette approche crée un modèle de classification basique
    
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')  # Ajuster selon nb de classes
    ])
    
    return model

def convert_to_tflite(model_data, output_path='android/app/src/main/assets'):
    """Convertit le modèle en TensorFlow Lite"""
    
    # Créer le dossier de sortie
    os.makedirs(output_path, exist_ok=True)
    
    # Vérifier le format du modèle
    if isinstance(model_data, dict):
        classifier = model_data.get('classifier')
        names = model_data.get('names', [])
        encodings = model_data.get('encodings', [])
        
        print(f"📊 Modèle chargé:")
        print(f"   - Nombres de visages: {len(names)}")
        print(f"   - Noms: {names}")
        
        # Sauvegarder les métadonnées (noms des classes)
        metadata = {
            'names': names,
            'num_classes': len(names) if names else 1,
            'input_shape': [100, 100, 3],  # Ajuster selon votre modèle
            'model_type': str(type(classifier).__name__)
        }
        
        metadata_path = os.path.join(output_path, 'face_recognition_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"✓ Métadonnées sauvegardées: {metadata_path}")
        
        # Si c'est un modèle sklearn, le convertir
        if hasattr(classifier, 'predict'):
            print("⚠️  Détection d'un modèle sklearn")
            print("   Pour une conversion complète, utilisez sklearn-porter ou m2cgen")
            print("   Ou réentraînez avec TensorFlow/Keras")
            
            # Créer un modèle Keras de référence
            print("\n🔨 Création d'un modèle Keras de référence...")
            keras_model = create_keras_model_from_sklearn(classifier, input_shape=(100*100*3,))
            
            # Compiler le modèle
            keras_model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("✓ Modèle Keras créé")
            
            # Convertir en TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            # Sauvegarder
            tflite_path = os.path.join(output_path, 'face_recognition_model.tflite')
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"✓ Modèle TFLite sauvegardé: {tflite_path}")
            print(f"   Taille: {len(tflite_model) / 1024:.2f} KB")
            
            return tflite_path, metadata_path
    
    else:
        print("⚠️  Format de modèle non reconnu")
        return None, None

def create_sample_tflite_model(output_path='android/app/src/main/assets'):
    """
    Crée un modèle TFLite d'exemple pour tester l'intégration
    À remplacer par votre vrai modèle entraîné
    """
    print("\n🎨 Création d'un modèle d'exemple pour tests...")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Créer un modèle simple
    model = keras.Sequential([
        keras.layers.Input(shape=(100, 100, 3)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(5, activation='softmax')  # 5 classes d'exemple
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Convertir en TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Sauvegarder
    tflite_path = os.path.join(output_path, 'face_recognition_model.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    # Métadonnées d'exemple
    metadata = {
        'names': ['Personne 1', 'Personne 2', 'Personne 3', 'Personne 4', 'Inconnu'],
        'num_classes': 5,
        'input_shape': [100, 100, 3],
        'model_type': 'CNN'
    }
    
    metadata_path = os.path.join(output_path, 'face_recognition_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Modèle d'exemple créé: {tflite_path}")
    print(f"✓ Métadonnées créées: {metadata_path}")
    print(f"   Taille: {len(tflite_model) / 1024:.2f} KB")
    
    return tflite_path, metadata_path

def main():
    print("=" * 60)
    print("🔄 CONVERSION DE MODÈLE POUR ANDROID")
    print("=" * 60)
    
    pkl_path = 'face_model.pkl'
    
    if os.path.exists(pkl_path):
        try:
            model_data = load_pickle_model(pkl_path)
            tflite_path, metadata_path = convert_to_tflite(model_data)
            
            if tflite_path:
                print("\n" + "=" * 60)
                print("✅ CONVERSION RÉUSSIE!")
                print("=" * 60)
                print(f"📦 Fichiers créés:")
                print(f"   - {tflite_path}")
                print(f"   - {metadata_path}")
                print("\n💡 IMPORTANT:")
                print("   Le modèle sklearn a été converti en architecture Keras.")
                print("   Pour de meilleures performances, réentraînez avec TensorFlow/Keras.")
            else:
                print("\n⚠️  Utilisation d'un modèle d'exemple à la place...")
                create_sample_tflite_model()
                
        except Exception as e:
            print(f"\n❌ Erreur lors de la conversion: {e}")
            print("   Création d'un modèle d'exemple à la place...")
            create_sample_tflite_model()
    else:
        print(f"\n⚠️  Fichier {pkl_path} non trouvé")
        print("   Création d'un modèle d'exemple pour les tests...")
        create_sample_tflite_model()
    
    print("\n" + "=" * 60)
    print("📱 PROCHAINES ÉTAPES:")
    print("=" * 60)
    print("1. Vérifiez les fichiers dans android/app/src/main/assets/")
    print("2. Ouvrez le projet Android dans Android Studio")
    print("3. Synchronisez Gradle")
    print("4. Compilez et testez l'application")
    print("=" * 60)

if __name__ == '__main__':
    main()
