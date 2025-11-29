package com.example.facerecognition.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Extracteur d'embeddings faciaux utilisant MobileFaceNet
 * Génère des vecteurs 512D à partir d'images de visages
 */
class EmbeddingExtractor(private val context: Context) {

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    
    companion object {
        private const val TAG = "EmbeddingExtractor"
        // Passage à ONNX: fournir un fichier 'mobilefacenet.onnx' dans assets
        private const val MODEL_FILE = "mobilefacenet.onnx"
        
        // Paramètres d'entrée du modèle
        private const val INPUT_SIZE = 112
        private const val PIXEL_SIZE = 3
        private const val IMAGE_MEAN = 127.5f
        private const val IMAGE_STD = 127.5f
        
        // Dimension de sortie
        private const val EMBEDDING_SIZE = 512
    }

    /**
     * Initialise le modèle ONNX (MobileFaceNet embeddings)
     */
    fun initialize(): Boolean {
        return try {
            Log.d(TAG, "[EmbeddingExtractor] Début initialisation ONNX...")
            Log.d(TAG, "[EmbeddingExtractor] Chargement de $MODEL_FILE...")
            
            // Vérifier que le fichier existe
            val assetList = context.assets.list("")
            Log.d(TAG, "[EmbeddingExtractor] Assets disponibles: ${assetList?.joinToString() ?: "aucun"}")
            
            if (assetList?.contains(MODEL_FILE) != true) {
                Log.e(TAG, "[EmbeddingExtractor] ✗ Fichier $MODEL_FILE introuvable dans assets")
                Log.e(TAG, "[EmbeddingExtractor] Fichiers .onnx présents: ${assetList?.filter { it.endsWith(".onnx") }?.joinToString()}")
                return false
            }
            
            Log.d(TAG, "[EmbeddingExtractor] ✓ Fichier $MODEL_FILE trouvé")
            // Charger le modèle ONNX en mémoire
            val modelBytes = context.assets.open(MODEL_FILE).use { it.readBytes() }
            Log.d(TAG, "[EmbeddingExtractor] Taille modèle: ${modelBytes.size} bytes")

            // Créer environnement & session
            ortEnv = OrtEnvironment.getEnvironment()
            ortSession = ortEnv!!.createSession(modelBytes)

            // Inspecter entrées/sorties
            val inputNames = ortSession!!.inputNames
            val outputNames = ortSession!!.outputNames
            Log.d(TAG, "[EmbeddingExtractor] Entrées: $inputNames")
            Log.d(TAG, "[EmbeddingExtractor] Sorties: $outputNames")

            Log.d(TAG, "[EmbeddingExtractor] ✓✓✓ Session ONNX prête ✓✓✓")
            
            true
        } catch (e: Exception) {
            Log.e(TAG, "[EmbeddingExtractor] ✗✗✗ Erreur init ONNX: ${e.message}", e)
            e.printStackTrace()
            false
        }
    }

    /**
     * Extrait un embedding 512D d'une image de visage
     * 
     * @param faceBitmap Image du visage (sera redimensionnée à 112x112)
     * @return FloatArray de 512 éléments (embedding normalisé L2)
     */
    fun extract(faceBitmap: Bitmap): FloatArray? {
        if (ortSession == null || ortEnv == null) {
            Log.e(TAG, "Session ONNX non initialisée")
            return null
        }
        return try {
            // Prétraiter l'image -> FloatArray [112*112*3]
            val inputFloats = preprocessToFloatArray(faceBitmap)

            val inputName = ortSession!!.inputNames.first()
            val inputTensor = OnnxTensor.createTensor(
                ortEnv!!,
                FloatBuffer.wrap(inputFloats),
                longArrayOf(1, INPUT_SIZE.toLong(), INPUT_SIZE.toLong(), PIXEL_SIZE.toLong())
            )

            val results = ortSession!!.run(mapOf(inputName to inputTensor))
            val outputTensor = results.values.first().value as OnnxTensor
            val rawEmbedding = outputTensor.floatBuffer.array()

            // Copie sûre (array peut être plus grand que EMBEDDING_SIZE)
            val embedding = if (rawEmbedding.size == EMBEDDING_SIZE) rawEmbedding
            else rawEmbedding.copyOf(EMBEDDING_SIZE)

            normalizeL2(embedding)
            Log.d(TAG, "Embedding ONNX extrait: ${embedding.size}D norm=${calculateNorm(embedding)}")

            // Libérer ressources temporaires
            inputTensor.close()
            results.close()
            outputTensor.close()

            embedding
        } catch (e: Exception) {
            Log.e(TAG, "Erreur extraction embedding ONNX: ${e.message}", e)
            null
        }
    }

    /** Prétraitement vers FloatArray NHWC */
    private fun preprocessToFloatArray(bitmap: Bitmap): FloatArray {
        val scaled = if (bitmap.width != INPUT_SIZE || bitmap.height != INPUT_SIZE) {
            Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        } else bitmap

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        val floats = FloatArray(INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE)
        var idx = 0
        for (p in pixels) {
            val r = (p shr 16 and 0xFF)
            val g = (p shr 8 and 0xFF)
            val b = (p and 0xFF)
            floats[idx++] = (r - IMAGE_MEAN) / IMAGE_STD
            floats[idx++] = (g - IMAGE_MEAN) / IMAGE_STD
            floats[idx++] = (b - IMAGE_MEAN) / IMAGE_STD
        }
        return floats
    }

    /**
     * Normalise un vecteur avec la norme L2
     * embedding = embedding / ||embedding||
     */
    private fun normalizeL2(embedding: FloatArray) {
        val norm = calculateNorm(embedding)
        if (norm > 0) {
            for (i in embedding.indices) {
                embedding[i] /= norm
            }
        }
    }

    /**
     * Calcule la norme L2 d'un vecteur
     */
    private fun calculateNorm(vector: FloatArray): Float {
        var sum = 0f
        for (value in vector) {
            sum += value * value
        }
        return kotlin.math.sqrt(sum)
    }

    /**
     * Libère les ressources
     */
    fun close() {
        try {
            ortSession?.close()
        } catch (_: Exception) {}
        ortSession = null
        ortEnv = null
        Log.d(TAG, "Ressources ONNX libérées")
    }
}
