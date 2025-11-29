#!/usr/bin/env python3
"""
Script pour télécharger et convertir MobileFaceNet en ONNX
Télécharge un modèle pré-entraîné et le convertit au format ONNX pour Android

Sources possibles:
1. TensorFlow Hub / GitHub repos publics (vérifier licence)
2. ONNX Model Zoo (modèles vérifiés)
3. Conversion depuis TFLite existant

Installation dépendances:
  pip install tensorflow onnx tf2onnx requests

Usage:
  python download_mobilefacenet_onnx.py --output android/app/src/main/assets/mobilefacenet.onnx
"""
import argparse
import os
import sys
import urllib.request
import tempfile

def download_from_url(url: str, output_path: str):
    """Télécharge un fichier ONNX depuis une URL"""
    print(f"Téléchargement depuis: {url}")
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(data)
        print(f"✓ Téléchargé: {output_path} ({len(data)} bytes)")
        return True
    except Exception as e:
        print(f"✗ Erreur téléchargement: {e}")
        return False

def convert_tflite_to_onnx(tflite_path: str, output_path: str):
    """Convertit un modèle TFLite en ONNX (nécessite tf2onnx)"""
    try:
        import tensorflow as tf
        import tf2onnx
        
        print(f"Conversion TFLite -> ONNX: {tflite_path}")
        
        # Charger TFLite
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Obtenir détails entrée/sortie
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"  Entrée: {input_details[0]['shape']}")
        print(f"  Sortie: {output_details[0]['shape']}")
        
        # Conversion (méthode alternative nécessaire - TFLite -> SavedModel -> ONNX)
        print("⚠ Conversion directe TFLite->ONNX complexe.")
        print("  Recommandation: utiliser modèle source TensorFlow/PyTorch si disponible")
        return False
        
    except ImportError:
        print("✗ tf2onnx non installé: pip install tf2onnx tensorflow")
        return False
    except Exception as e:
        print(f"✗ Erreur conversion: {e}")
        return False

def main():
    ap = argparse.ArgumentParser(description="Télécharge/convertit MobileFaceNet ONNX")
    ap.add_argument('--output', '-o', required=True, help='Chemin de sortie mobilefacenet.onnx')
    ap.add_argument('--url', help='URL du modèle ONNX pré-converti (optionnel)')
    ap.add_argument('--tflite', help='Chemin TFLite à convertir (expérimental)')
    args = ap.parse_args()

    if args.url:
        success = download_from_url(args.url, args.output)
    elif args.tflite:
        success = convert_tflite_to_onnx(args.tflite, args.output)
    else:
        print("=" * 70)
        print("OBTENTION DE MOBILEFACENET.ONNX")
        print("=" * 70)
        print()
        print("Options recommandées:")
        print()
        print("1. Télécharger depuis un repo GitHub (exemple):")
        print("   URL: https://github.com/onnx/models/raw/main/vision/body_analysis/...")
        print("   (Vérifier disponibilité et licence)")
        print()
        print("2. Convertir depuis PyTorch:")
        print("   - Charger modèle MobileFaceNet PyTorch")
        print("   - torch.onnx.export(model, dummy_input, 'mobilefacenet.onnx')")
        print()
        print("3. Convertir depuis TensorFlow:")
        print("   - python -m tf2onnx.convert --saved-model ./model --output mobilefacenet.onnx")
        print()
        print("4. Utiliser modèle pré-entraîné disponible:")
        print("   - Insightface (ArcFace/MobileFaceNet)")
        print("   - FaceNet-PyTorch")
        print()
        print("=" * 70)
        print("SPÉCIFICATIONS REQUISES:")
        print("=" * 70)
        print("  Entrée:  [1, 112, 112, 3] float32 NHWC normalisé (x-127.5)/127.5")
        print("  Sortie:  [1, 512] float32 (embedding)")
        print()
        print("Exemple commande avec URL:")
        print(f"  python {sys.argv[0]} --url <URL_DU_MODELE> --output {args.output}")
        print()
        success = False

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
