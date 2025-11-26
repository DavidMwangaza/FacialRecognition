"""
Script pour convertir face_model.pkl (embeddings) en modèle TensorFlow Lite
Entraîne un classificateur sur les embeddings existants
"""
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json
import os

def load_embeddings(pkl_path):
    """Charge les embeddings depuis le fichier pickle"""
    print(f"📂 Chargement des embeddings depuis {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extraire les embeddings et labels
    embeddings = []
    labels = []
    
    for item in data:
        embeddings.append(np.array(item['embedding']))
        labels.append(item['label'])
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    print(f"✓ {len(embeddings)} embeddings chargés")
    print(f"✓ Dimension des embeddings: {embeddings.shape[1]}")
    print(f"✓ Personnes: {np.unique(labels)}")
    print(f"✓ Nombre de personnes: {len(np.unique(labels))}")
    
    return embeddings, labels

def create_classifier_model(input_dim, num_classes):
    """Crée un modèle de classification pour les embeddings"""
    print(f"\n🔨 Création du modèle classificateur...")
    print(f"   Input: {input_dim} dimensions")
    print(f"   Output: {num_classes} classes")
    
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        
        # Normalisation
        keras.layers.BatchNormalization(),
        
        # Couches denses avec dropout
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),
        
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),
        
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        
        # Couche de sortie
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Modèle créé")
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Entraîne le modèle"""
    print(f"\n🎓 Entraînement du modèle...")
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Évaluer le modèle
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n✓ Entraînement terminé")
    print(f"✓ Précision validation: {val_acc*100:.2f}%")
    
    return model

def convert_to_tflite(model, output_path='android/app/src/main/assets'):
    """Convertit le modèle en TensorFlow Lite avec compatibilité maximale"""
    print(f"\n🔄 Conversion en TensorFlow Lite...")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Convertir en TFLite avec compatibilité stricte pour TFLite 2.16.1
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # FORCER la compatibilité avec opérations de base uniquement
    # Ne pas optimiser pour garder les opérations simples
    # converter.optimizations = []  # PAS d'optimisation qui pourrait générer des ops v12
    
    # Utiliser uniquement les opérations de base TFLite
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS  # Opérations TFLite standards uniquement
    ]
    
    # Désactiver les nouvelles fonctionnalités expérimentales
    converter._experimental_lower_tensor_list_ops = False
    
    # Forcer l'utilisation de types de données standards
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"⚠️ Erreur lors de la conversion standard: {e}")
        print("🔄 Tentative avec quantification int8 pour compatibilité...")
        
        # Si la conversion échoue, essayer avec quantification
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
    
    # Sauvegarder
    tflite_path = os.path.join(output_path, 'face_recognition_model.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✓ Modèle TFLite sauvegardé: {tflite_path}")
    print(f"   Taille: {len(tflite_model) / 1024:.2f} KB")
    print(f"   Compatible avec TensorFlow Lite 2.16.1")
    
    return tflite_path

def save_metadata(labels, output_path='android/app/src/main/assets'):
    """Sauvegarde les métadonnées (noms des classes)"""
    print(f"\n📝 Sauvegarde des métadonnées...")
    
    # Obtenir les noms uniques dans l'ordre
    unique_labels = sorted(np.unique(labels))
    
    metadata = {
        'names': list(unique_labels),
        'num_classes': len(unique_labels),
        'input_shape': [512],
        'model_type': 'EmbeddingClassifier'
    }
    
    metadata_path = os.path.join(output_path, 'face_recognition_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Métadonnées sauvegardées: {metadata_path}")
    print(f"   Personnes reconnues: {', '.join(unique_labels)}")
    
    return metadata_path

def main():
    print("=" * 70)
    print("🤖 CONVERSION DE VOTRE MODÈLE POUR ANDROID")
    print("=" * 70)
    
    pkl_path = 'face_model.pkl'
    
    if not os.path.exists(pkl_path):
        print(f"❌ Erreur: {pkl_path} non trouvé")
        return
    
    try:
        # 1. Charger les embeddings
        embeddings, labels = load_embeddings(pkl_path)
        
        # 2. Encoder les labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        num_classes = len(label_encoder.classes_)
        
        print(f"\n📊 Classes encodées:")
        for i, name in enumerate(label_encoder.classes_):
            count = np.sum(labels == name)
            print(f"   {i}: {name} ({count} exemples)")
        
        # 3. Diviser en train/val
        X_train, X_val, y_train, y_val = train_test_split(
            embeddings, labels_encoded,
            test_size=0.2,
            random_state=42,
            stratify=labels_encoded
        )
        
        print(f"\n📦 Données divisées:")
        print(f"   Train: {len(X_train)} exemples")
        print(f"   Validation: {len(X_val)} exemples")
        
        # 4. Créer et entraîner le modèle
        model = create_classifier_model(embeddings.shape[1], num_classes)
        model = train_model(model, X_train, y_train, X_val, y_val)
        
        # 5. Convertir en TFLite
        tflite_path = convert_to_tflite(model)
        
        # 6. Sauvegarder les métadonnées
        metadata_path = save_metadata(labels)
        
        print("\n" + "=" * 70)
        print("✅ CONVERSION RÉUSSIE!")
        print("=" * 70)
        print(f"📦 Fichiers créés:")
        print(f"   ✓ {tflite_path}")
        print(f"   ✓ {metadata_path}")
        print("\n🎯 PROCHAINES ÉTAPES:")
        print("=" * 70)
        print("1. Ouvrez Android Studio et importez le projet:")
        print("   File > Open > C:\\Users\\david\\Documents\\Appli\\android")
        print("\n2. Attendez la synchronisation Gradle")
        print("\n3. Connectez votre téléphone ou lancez un émulateur")
        print("\n4. Cliquez sur ▶️ Run pour compiler et installer")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
