package com.example.facerecognition.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
// GPU delegate import retiré pour compatibilité
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Extracteur d'embeddings faciaux utilisant MobileFaceNet
 * Génère des vecteurs 512D à partir d'images de visages
 */
class EmbeddingExtractor(private val context: Context) {

    private var interpreter: Interpreter? = null
    // Supprimé: délégation GPU pour éviter conflits de dépendances
    
    companion object {
        private const val TAG = "EmbeddingExtractor"
        private const val MODEL_FILE = "mobilefacenet.tflite"
        
        // Paramètres d'entrée du modèle
        private const val INPUT_SIZE = 112
        private const val PIXEL_SIZE = 3
        private const val IMAGE_MEAN = 127.5f
        private const val IMAGE_STD = 127.5f
        
        // Dimension de sortie
        private const val EMBEDDING_SIZE = 512
    }

    /**
     * Initialise le modèle TFLite
     */
    fun initialize(): Boolean {
        return try {
            Log.d(TAG, "[EmbeddingExtractor] Début initialisation...")
            Log.d(TAG, "[EmbeddingExtractor] Chargement de $MODEL_FILE...")
            
            // Vérifier que le fichier existe
            val assetList = context.assets.list("")
            Log.d(TAG, "[EmbeddingExtractor] Assets disponibles: ${assetList?.joinToString() ?: "aucun"}")
            
            if (assetList?.contains(MODEL_FILE) != true) {
                Log.e(TAG, "[EmbeddingExtractor] ✗ Fichier $MODEL_FILE introuvable dans assets")
                Log.e(TAG, "[EmbeddingExtractor] Fichiers .tflite présents: ${assetList?.filter { it.endsWith(".tflite") }?.joinToString()}")
                return false
            }
            
            Log.d(TAG, "[EmbeddingExtractor] ✓ Fichier $MODEL_FILE trouvé")
            
            // Charger le modèle
            Log.d(TAG, "[EmbeddingExtractor] Chargement du fichier modèle...")
            val model = loadModelFile()
            Log.d(TAG, "[EmbeddingExtractor] ✓ Fichier modèle chargé: ${model.capacity()} bytes")
            
            // Configurer les options TFLite
            Log.d(TAG, "[EmbeddingExtractor] Configuration options TFLite...")
            val options = Interpreter.Options().apply {
                // Utiliser plusieurs threads CPU et NNAPI pour accélération
                setNumThreads(4)
                setUseNNAPI(true)
            }
            Log.d(TAG, "[EmbeddingExtractor] ✓ Options configurées")
            
            // Créer l'interpréteur
            Log.d(TAG, "[EmbeddingExtractor] Création Interpreter...")
            interpreter = Interpreter(model, options)
            Log.d(TAG, "[EmbeddingExtractor] ✓ Interpreter créé")
            
            // Vérifier les dimensions d'entrée/sortie
            val inputShape = interpreter!!.getInputTensor(0).shape()
            val outputShape = interpreter!!.getOutputTensor(0).shape()
            
            Log.d(TAG, "[EmbeddingExtractor] ✓✓✓ Modèle chargé avec succès ✓✓✓")
            Log.d(TAG, "[EmbeddingExtractor]   Input shape: ${inputShape.contentToString()}")
            Log.d(TAG, "[EmbeddingExtractor]   Output shape: ${outputShape.contentToString()}")
            
            true
        } catch (e: Exception) {
            Log.e(TAG, "[EmbeddingExtractor] ✗✗✗ Erreur lors du chargement: ${e.message}", e)
            e.printStackTrace()
            false
        }
    }

    /**
     * Charge le fichier du modèle depuis les assets
     */
    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(MODEL_FILE)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Extrait un embedding 512D d'une image de visage
     * 
     * @param faceBitmap Image du visage (sera redimensionnée à 112x112)
     * @return FloatArray de 512 éléments (embedding normalisé L2)
     */
    fun extract(faceBitmap: Bitmap): FloatArray? {
        if (interpreter == null) {
            Log.e(TAG, "Modèle non initialisé")
            return null
        }

        return try {
            // 1. Prétraiter l'image
            val inputBuffer = preprocessImage(faceBitmap)
            
            // 2. Préparer le buffer de sortie
            val outputBuffer = Array(1) { FloatArray(EMBEDDING_SIZE) }
            
            // 3. Exécuter l'inférence
            interpreter!!.run(inputBuffer, outputBuffer)
            
            // 4. Extraire et normaliser l'embedding
            val embedding = outputBuffer[0]
            normalizeL2(embedding)
            
            Log.d(TAG, "Embedding extrait: ${embedding.size}D, norm=${calculateNorm(embedding)}")
            
            embedding
        } catch (e: Exception) {
            Log.e(TAG, "Erreur lors de l'extraction de l'embedding", e)
            null
        }
    }

    /**
     * Prétraite l'image pour l'entrée du modèle
     * - Redimensionne à 112x112
     * - Normalise les pixels: (pixel - 127.5) / 127.5
     * - Format: float32, ordre NCHW ou NHWC selon le modèle
     */
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // Redimensionner si nécessaire
        val scaledBitmap = if (bitmap.width != INPUT_SIZE || bitmap.height != INPUT_SIZE) {
            Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        } else {
            bitmap
        }

        // Créer le buffer d'entrée
        val inputBuffer = ByteBuffer.allocateDirect(
            4 * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE
        ).apply {
            order(ByteOrder.nativeOrder())
        }

        // Extraire les pixels et normaliser
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaledBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            // Extraire RGB
            val r = ((pixel shr 16) and 0xFF)
            val g = ((pixel shr 8) and 0xFF)
            val b = (pixel and 0xFF)

            // Normaliser: (pixel - mean) / std
            inputBuffer.putFloat((r - IMAGE_MEAN) / IMAGE_STD)
            inputBuffer.putFloat((g - IMAGE_MEAN) / IMAGE_STD)
            inputBuffer.putFloat((b - IMAGE_MEAN) / IMAGE_STD)
        }

        return inputBuffer
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
        interpreter?.close()
        interpreter = null
        
        // Pas de délégation GPU
        
        Log.d(TAG, "Ressources libérées")
    }
}
