package com.example.facerecognition.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.gson.Gson
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.FloatBuffer
import kotlin.math.exp

/**
 * Classe pour gérer l'inférence du modèle ONNX
 * Reconnaissance faciale hors ligne
 */
class FaceRecognitionModel(private val context: Context) {
    
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var embeddingExtractor: EmbeddingExtractor? = null
    private var labels: List<String> = emptyList()
    private var inputShape: LongArray = longArrayOf(1, 512)
    private var outputShape: LongArray = longArrayOf(1, 2)
    
    companion object {
        private const val TAG = "FaceRecognitionModel"
        private const val MODEL_FILE = "face_recognition_model.onnx"
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
            Log.d(TAG, "Début initialisation FaceRecognitionModel...")
            loadModel()
            Log.d(TAG, "✓ Modèle classifier chargé")
            loadMetadata()
            Log.d(TAG, "✓ Métadonnées chargées")
            
            // Initialiser l'extracteur d'embeddings réel
            Log.d(TAG, "Création EmbeddingExtractor...")
            embeddingExtractor = EmbeddingExtractor(context)
            Log.d(TAG, "Appel initialize() sur EmbeddingExtractor...")
            if (!embeddingExtractor!!.initialize()) {
                Log.e(TAG, "✗ Echec initialisation EmbeddingExtractor, extracteur indisponible")
                embeddingExtractor = null
            } else {
                Log.d(TAG, "✓ EmbeddingExtractor (MobileFaceNet) initialisé avec succès")
            }
        } catch (e: Exception) {
            Log.e(TAG, "✗ Erreur critique lors de l'initialisation du modèle", e)
            e.printStackTrace()
            throw RuntimeException("Impossible d'initialiser FaceRecognitionModel: ${e.message}", e)
        }
    }
    
    /**
     * Charge le modèle ONNX
     */
    private fun loadModel() {
        try {
            Log.d(TAG, "Chargement du modèle ONNX: $MODEL_FILE")
            
            // Vérifier que le fichier existe
            val assetFiles = context.assets.list("") ?: emptyArray()
            Log.d(TAG, "Fichiers assets disponibles: ${assetFiles.joinToString()}")
            
            if (!assetFiles.contains(MODEL_FILE)) {
                throw IllegalStateException("Fichier $MODEL_FILE introuvable dans assets")
            }
            
            // Charger le modèle ONNX
            val modelBytes = context.assets.open(MODEL_FILE).use { it.readBytes() }
            Log.d(TAG, "Modèle ONNX chargé: ${modelBytes.size} bytes")
            
            // Créer l'environnement et la session ONNX Runtime
            ortEnv = OrtEnvironment.getEnvironment()
            ortSession = ortEnv!!.createSession(modelBytes)
            
            Log.d(TAG, "✓ Session ONNX créée")
            
            // Récupérer les informations du modèle
            val inputInfo = ortSession!!.inputInfo
            val outputInfo = ortSession!!.outputInfo
            
            Log.d(TAG, "✓ Modèle ONNX chargé avec succès")
            Log.d(TAG, "  Input names: ${inputInfo.keys}")
            Log.d(TAG, "  Output names: ${outputInfo.keys}")
            
        } catch (e: Exception) {
            Log.e(TAG, "Erreur lors du chargement du modèle ONNX: ${e.message}", e)
            e.printStackTrace()
            throw e
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
        if (ortSession == null || ortEnv == null) {
            Log.e(TAG, "Modèle ONNX non chargé")
            return null
        }
        
        try {
            // Extraire l'embedding de l'image
            val embedding = extractEmbedding(faceBitmap)
            if (embedding == null) {
                Log.e(TAG, "Reconnaissance impossible: embedding non disponible")
                return null
            }
            
            // Créer le tenseur d'entrée ONNX (shape: [1, 512])
            val inputName = ortSession!!.inputNames.first()
            val inputTensor = OnnxTensor.createTensor(
                ortEnv!!,
                FloatBuffer.wrap(embedding),
                longArrayOf(1, EMBEDDING_SIZE.toLong())
            )
            
            // Exécuter l'inférence
            val results = ortSession!!.run(mapOf(inputName to inputTensor))
            
            // Récupérer les résultats
            val outputTensor = results.first().value as OnnxTensor
            val outputArray = outputTensor.floatBuffer.array()
            
            // Appliquer softmax si nécessaire
            val softmaxProbs = softmax(outputArray)
            
            // Trouver la classe avec la plus haute probabilité
            val maxIndex = softmaxProbs.indices.maxByOrNull { softmaxProbs[it] } ?: 0
            val confidence = softmaxProbs[maxIndex]
            
            val name = if (maxIndex < labels.size) {
                labels[maxIndex]
            } else {
                "Inconnu"
            }
            
            Log.d(TAG, "Reconnaissance ONNX: $name (confiance: ${confidence * 100}%)")
            
            // Libérer les ressources
            inputTensor.close()
            results.close()
            
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
        ortSession?.close()
        ortSession = null
        ortEnv = null
        
        embeddingExtractor?.close()
        embeddingExtractor = null
        
        Log.d(TAG, "✓ Modèle ONNX et extracteur fermés")
    }
}
