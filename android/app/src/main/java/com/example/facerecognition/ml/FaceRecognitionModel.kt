package com.example.facerecognition.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.gson.Gson
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * Classe pour gérer l'inférence du modèle TensorFlow Lite
 * Reconnaissance faciale hors ligne
 */
class FaceRecognitionModel(private val context: Context) {
    
    private var interpreter: Interpreter? = null
    private var embeddingExtractor: EmbeddingExtractor? = null
    private var labels: List<String> = emptyList()
    private var inputShape: IntArray = intArrayOf(1, 512)
    private var outputShape: IntArray = intArrayOf(1, 2)
    
    companion object {
        private const val TAG = "FaceRecognitionModel"
        private const val MODEL_FILE = "face_recognition_model.tflite"
        private const val METADATA_FILE = "face_recognition_metadata.json"
        private const val BATCH_SIZE = 1
        private const val EMBEDDING_SIZE = 512
    }
    
    data class ModelMetadata(
        val names: List<String>,
        val num_classes: Int,
        val input_shape: List<Int>,
        val model_type: String
    )
    
    data class RecognitionResult(
        val name: String,
        val confidence: Float,
        val classIndex: Int
    )
    
    init {
        try {
            loadModel()
            loadMetadata()
            
            // Initialiser l'extracteur d'embeddings réel
            embeddingExtractor = EmbeddingExtractor(context)
            if (!embeddingExtractor!!.initialize()) {
                Log.e(TAG, "Echec initialisation EmbeddingExtractor, extracteur indisponible")
                embeddingExtractor = null
            } else {
                Log.d(TAG, "EmbeddingExtractor (MobileFaceNet) initialisé avec succès")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Erreur critique lors de l'initialisation du modèle", e)
            throw RuntimeException("Impossible d'initialiser FaceRecognitionModel: ${e.message}", e)
        }
    }
    
    /**
     * Charge le modèle TensorFlow Lite
     */
    private fun loadModel() {
        try {
            Log.d(TAG, "Chargement du modèle: $MODEL_FILE")
            
            // Vérifier que le fichier existe
            val assetFiles = context.assets.list("") ?: emptyArray()
            Log.d(TAG, "Fichiers assets disponibles: ${assetFiles.joinToString()}")
            
            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseNNAPI(true) // Utiliser Neural Networks API si disponible
            }
            
            val modelBuffer = FileUtil.loadMappedFile(context, MODEL_FILE)
            Log.d(TAG, "Buffer modèle chargé: ${modelBuffer.capacity()} bytes")
            
            interpreter = Interpreter(modelBuffer, options)
            Log.d(TAG, "✓ Interpreter créé")
            
            // Récupérer les dimensions du modèle
            val inputTensor = interpreter?.getInputTensor(0)
            val outputTensor = interpreter?.getOutputTensor(0)
            
            inputShape = inputTensor?.shape() ?: inputShape
            outputShape = outputTensor?.shape() ?: outputShape
            
            Log.d(TAG, "✓ Modèle chargé avec succès")
            Log.d(TAG, "  Input shape: ${inputShape.contentToString()}")
            Log.d(TAG, "  Output shape: ${outputShape.contentToString()}")
            
        } catch (e: Exception) {
            Log.e(TAG, "Erreur lors du chargement du modèle: ${e.message}", e)
            e.printStackTrace()
            throw e // Propager l'erreur pour qu'elle soit visible
        }
    }
    
    /**
     * Charge les métadonnées (noms des classes)
     */
    private fun loadMetadata() {
        try {
            Log.d(TAG, "📋 Chargement métadonnées: $METADATA_FILE")
            
            val jsonString = context.assets.open(METADATA_FILE).use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).use { reader ->
                    reader.readText()
                }
            }
            
            Log.d(TAG, "JSON lu: $jsonString")
            
            val gson = Gson()
            val metadata = gson.fromJson(jsonString, ModelMetadata::class.java)
            
            labels = metadata.names
            
            Log.d(TAG, "✓ Métadonnées chargées")
            Log.d(TAG, "  Nombre de classes: ${labels.size}")
            Log.d(TAG, "  Labels: $labels")
            
        } catch (e: Exception) {
            Log.e(TAG, "Erreur lors du chargement des métadonnées: ${e.message}", e)
            e.printStackTrace()
            // Labels par défaut si échec
            labels = List(outputShape[1]) { "Personne $it" }
        }
    }
    
    /**
     * Extrait un embedding 512D depuis une image de visage via MobileFaceNet.
     * Retourne null si l'extracteur n'est pas disponible ou en cas d'erreur.
     */
    private fun extractEmbedding(bitmap: Bitmap): FloatArray? {
        val extractedEmbedding = embeddingExtractor?.extract(bitmap)
        return if (extractedEmbedding != null) {
            Log.d(TAG, "✓ Embedding extrait par MobileFaceNet")
            extractedEmbedding
        } else {
            Log.e(TAG, "EmbeddingExtractor indisponible ou modèle non chargé")
            null
        }
    }
    
    /**
     * Effectue la reconnaissance faciale à partir d'une image
     */
    fun recognize(faceBitmap: Bitmap): RecognitionResult? {
        if (interpreter == null) {
            Log.e(TAG, "Modèle non chargé")
            return null
        }
        
        try {
            // Extraire l'embedding de l'image
            val embedding = extractEmbedding(faceBitmap)
            if (embedding == null) {
                Log.e(TAG, "Reconnaissance impossible: embedding non disponible")
                return null
            }
            
            // Créer le buffer d'entrée
            val inputBuffer = ByteBuffer.allocateDirect(BATCH_SIZE * EMBEDDING_SIZE * 4)
            inputBuffer.order(ByteOrder.nativeOrder())
            for (value in embedding) {
                inputBuffer.putFloat(value)
            }
            
            // Préparer le buffer de sortie
            val numClasses = outputShape[1]
            val outputBuffer = Array(BATCH_SIZE) { FloatArray(numClasses) }
            
            // Exécuter l'inférence
            interpreter?.run(inputBuffer, outputBuffer)
            
            // Analyser les résultats
            val probabilities = outputBuffer[0]
            
            // Appliquer softmax si nécessaire
            val softmaxProbs = softmax(probabilities)
            
            // Trouver la classe avec la plus haute probabilité
            val maxIndex = softmaxProbs.indices.maxByOrNull { softmaxProbs[it] } ?: 0
            val confidence = softmaxProbs[maxIndex]
            
            val name = if (maxIndex < labels.size) {
                labels[maxIndex]
            } else {
                "Inconnu"
            }
            
            Log.d(TAG, "Reconnaissance: $name (confiance: ${confidence * 100}%)")
            
            return RecognitionResult(
                name = name,
                confidence = confidence,
                classIndex = maxIndex
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Erreur lors de la reconnaissance: ${e.message}", e)
            return null
        }
    }
    
    /**
     * Applique la fonction softmax pour obtenir des probabilités
     */
    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val expValues = logits.map { exp((it - maxLogit).toDouble()).toFloat() }
        val sumExp = expValues.sum()
        return expValues.map { it / sumExp }.toFloatArray()
    }
    
    /**
     * Libère les ressources
     */
    fun close() {
        interpreter?.close()
        interpreter = null
        
        embeddingExtractor?.close()
        embeddingExtractor = null
        
        Log.d(TAG, "✓ Modèle et extracteur fermés")
    }
}
