Ajoutez le fichier du modèle d'embeddings ONNX ici: mobilefacenet.onnx

Exigences:
- Nom exact attendu par le code: mobilefacenet.onnx
- Entrée: 1 x 112 x 112 x 3 (NHWC, float32 normalisé (x-127.5)/127.5)
- Sortie: 1 x 512 (embedding visage)

Obtention du modèle:
1. Depuis un repo public MobileFaceNet ONNX (vérifiez la licence avant usage commercial).
2. Ou re-convertir depuis un checkpoint TensorFlow original:
   - Charger le graph / checkpoint.
   - Utiliser tf2onnx: `python -m tf2onnx.convert --saved-model ./saved_model --output mobilefacenet.onnx --opset 13`.
   - Vérifier la forme d'entrée (NHWC) ou adapter le prétraitement si NCHW.

Validation rapide (PC):
- `pip install onnxruntime numpy pillow`
- Script:
```
import onnxruntime as ort, numpy as np
from PIL import Image
img = Image.open('face.jpg').resize((112,112))
arr = np.asarray(img, dtype=np.float32)
arr = (arr - 127.5) / 127.5
arr = np.expand_dims(arr, 0)  # NHWC
sess = ort.InferenceSession('mobilefacenet.onnx')
inp_name = sess.get_inputs()[0].name
out = sess.run(None, {inp_name: arr})[0]
print(out.shape, np.linalg.norm(out[0]))
```
Norme proche de ~15 avant normalisation interne; notre code normalise L2.

Après ajout du fichier:
- Recompiler l'app.
- Vérifier le log: "✓✓✓ Session ONNX prête" dans EmbeddingExtractor.

Si le modèle est NCHW (1x3x112x112):
- Adapter `EmbeddingExtractor.preprocessToFloatArray` pour réordonner en canaux d'abord.
- Remplacer la création du tenseur par shape `longArrayOf(1,3,112,112)`.

