"""
Script pour convertir face_model.pkl (embeddings) en modèle ONNX
1) Entraîne un classificateur scikit-learn (MLPClassifier) sur les embeddings
2) Convertit le modèle vers ONNX via skl2onnx (compatible Python 3.13)
"""
import os
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import json

ASSETS_DIR = 'android/app/src/main/assets'
ONNX_PATH = os.path.join(ASSETS_DIR, 'face_recognition_model.onnx')
METADATA_PATH = os.path.join(ASSETS_DIR, 'face_recognition_metadata.json')
PKL_PATH = 'face_model.pkl'


def load_embeddings(pkl_path: str):
    print(f"📂 Chargement des embeddings depuis {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    embeddings, labels = [], []
    for item in data:
        embeddings.append(np.array(item['embedding'], dtype=np.float32))
        labels.append(item['label'])
    embeddings = np.array(embeddings, dtype=np.float32)
    labels = np.array(labels)
    print(f"✓ {len(embeddings)} embeddings | dims={embeddings.shape[1]}")
    print(f"✓ Personnes: {np.unique(labels)}")
    return embeddings, labels


def create_classifier_model(input_dim: int, num_classes: int):
    print("\n🔨 Création du modèle classificateur scikit-learn (MLP)...")
    # Pipeline: StandardScaler + MLPClassifier
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=True
        ))
    ])
    print(f"✓ Pipeline créé: StandardScaler → MLP(256→128→64→{num_classes})")
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    print("\n🎓 Entraînement...")
    model.fit(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    print(f"✓ Précision validation: {val_acc*100:.2f}%")
    return model


def save_metadata(labels, input_dim: int):
    os.makedirs(ASSETS_DIR, exist_ok=True)
    unique_labels = sorted(np.unique(labels))
    metadata = {
        'names': list(unique_labels),
        'num_classes': len(unique_labels),
        'input_shape': [input_dim],
        'model_type': 'EmbeddingClassifier',
        'format': 'onnx'
    }
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ Métadonnées sauvegardées: {METADATA_PATH}")


def convert_to_onnx(model, sample_input_dim: int):
    print("\n🔄 Conversion en ONNX via skl2onnx...")
    # Définir le type d'entrée pour ONNX (batch_size, input_dim)
    initial_type = [('float_input', FloatTensorType([None, sample_input_dim]))]
    
    # Convertir le pipeline scikit-learn en ONNX
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=13
    )
    
    # Sauvegarder
    with open(ONNX_PATH, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"✓ Modèle ONNX sauvegardé: {ONNX_PATH}")
    print(f"   Taille: {os.path.getsize(ONNX_PATH) / 1024:.2f} KB")


def main():
    print("="*70)
    print("🤖 CONVERSION DE VOTRE MODÈLE VERS ONNX")
    print("="*70)
    if not os.path.exists(PKL_PATH):
        print(f"❌ Erreur: {PKL_PATH} non trouvé")
        return

    embeddings, labels = load_embeddings(PKL_PATH)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)

    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
    )

    model = create_classifier_model(embeddings.shape[1], num_classes)
    model = train_model(model, X_train, y_train, X_val, y_val)

    os.makedirs(ASSETS_DIR, exist_ok=True)
    convert_to_onnx(model, embeddings.shape[1])
    save_metadata(labels, embeddings.shape[1])

    print("\n✅ Conversion ONNX terminée.")
    print(f"📦 Fichiers: \n  - {ONNX_PATH}\n  - {METADATA_PATH}")


if __name__ == '__main__':
    main()
