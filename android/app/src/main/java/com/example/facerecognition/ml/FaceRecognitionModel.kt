package com.example.facerecognition.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.gson.Gson
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.example.facerecognition.data.EmbeddingsLoader
import com.example.facerecognition.data.EmbeddingsStore
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.FloatBuffer
import kotlin.math.exp

/**
 * Classe pour gérer inference du modèle ONNX + matching embeddings JSON
 * Reconnaissance faciale hors ligne
 */
class FaceRecognitionModel(private val context: Context) {

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var embeddingExtractor: EmbeddingExtractor? = null
    private var labels: List<String> = emptyList()
    private var inputShape: LongArray = longArrayOf(1, 512)
    private var outputShape: LongArray = longArrayOf(1, 2)
    private var embeddingsStore: EmbeddingsStore? = null

    companion object {
        private const val TAG = "FaceRecognitionModel"
        private const val MODEL_FILE = "face_recognition_model.onnx"
        private const val METADATA_FILE = "face_recognition_metadata.json"
        private const val BATCH_SIZE = 1
        private const val EMBEDDING_SIZE = 512
        private const val MATCH_THRESHOLD = 0.55f // seuil cosine (embeddings L2 normalisés)
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
        val classIndex: Int,
        val method: String
    )

    data class EmbeddingMatch(
        val id: String,
        val similarity: Float
    )

    init {
        try {
            Log.d(TAG, "Début initialisation FaceRecognitionModel...")
            loadModel()
            Log.d(TAG, "✓ Modèle classifier chargé")
            loadMetadata()
            Log.d(TAG, "✓ Métadonnées chargées")

            // Charger éventuel store JSON d'embeddings
            embeddingsStore = EmbeddingsLoader.load(context)
            if (embeddingsStore != null) {
                Log.d(TAG, "✓ EmbeddingsStore chargé: ${embeddingsStore!!.count} identités")
            } else {
                Log.d(TAG, "Aucun embeddings.json chargé (matching par classifier seulement)")
            }

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

    /** Charge le modèle ONNX */
    private fun loadModel() {
        try {
            Log.d(TAG, "Chargement du modèle ONNX: $MODEL_FILE")
            val assetFiles = context.assets.list("") ?: emptyArray()
            Log.d(TAG, "Fichiers assets disponibles: ${assetFiles.joinToString()}")
            if (!assetFiles.contains(MODEL_FILE)) {
                throw IllegalStateException("Fichier $MODEL_FILE introuvable dans assets")
            }
            val modelBytes = context.assets.open(MODEL_FILE).use { it.readBytes() }
            Log.d(TAG, "Modèle ONNX chargé: ${modelBytes.size} bytes")
            ortEnv = OrtEnvironment.getEnvironment()
            ortSession = ortEnv!!.createSession(modelBytes)
            Log.d(TAG, "✓ Session ONNX créée")
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

    /** Charge les métadonnées (noms des classes) */
    private fun loadMetadata() {
        try {
            Log.d(TAG, "📋 Chargement métadonnées: $METADATA_FILE")
            val jsonString = context.assets.open(METADATA_FILE).use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).use { reader -> reader.readText() }
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
            labels = List(outputShape[1].toInt()) { "Personne $it" }
        }
    }

    /** Extrait un embedding 512D depuis une image de visage */
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

    /** Calcul cosine (embeddings supposés L2 normalisés) */
    private fun cosine(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        val n = minOf(a.size, b.size)
        for (i in 0 until n) dot += a[i] * b[i]
        return dot
    }

    /** Trouve meilleur match dans embeddingsStore */
    private fun matchEmbedding(embedding: FloatArray): EmbeddingMatch? {
        val store = embeddingsStore ?: return null
        var best: EmbeddingMatch? = null
        for (entry in store.embeddings) {
            if (entry.vector.size != embedding.size) continue
            val sim = cosine(embedding, entry.vector)
            if (best == null || sim > best.similarity) {
                best = EmbeddingMatch(entry.id, sim)
            }
        }
        return if (best != null && best.similarity >= MATCH_THRESHOLD) best else null
    }

    /** Reconnaissance par classification ONNX */
    fun recognizeClassifier(faceBitmap: Bitmap): RecognitionResult? {
        if (ortSession == null || ortEnv == null) {
            Log.e(TAG, "Modèle ONNX non chargé")
            return null
        }
        try {
            val embedding = extractEmbedding(faceBitmap) ?: return null
            val inputName = ortSession!!.inputNames.first()
            val inputTensor = OnnxTensor.createTensor(
                ortEnv!!,
                FloatBuffer.wrap(embedding),
                longArrayOf(1L, EMBEDDING_SIZE.toLong())
            )
            val results = ortSession!!.run(mapOf(inputName to inputTensor))
            val outputTensor = results.first().value as OnnxTensor
            val outputArray = outputTensor.floatBuffer.array()
            val softmaxProbs = softmax(outputArray)
            val maxIndex = softmaxProbs.indices.maxByOrNull { softmaxProbs[it] } ?: 0
            val confidence = softmaxProbs[maxIndex]
            val name = if (maxIndex < labels.size) labels[maxIndex] else "Inconnu"
            Log.d(TAG, "Reconnaissance ONNX (classifier): $name (conf: ${confidence * 100}%)")
            inputTensor.close()
            results.close()
            return RecognitionResult(name, confidence, maxIndex, method = "classifier")
        } catch (e: Exception) {
            Log.e(TAG, "Erreur classifier: ${e.message}", e)
            return null
        }
    }

    /** Reconnaissance par matching embeddings JSON */
    fun recognizeByEmbeddings(faceBitmap: Bitmap): RecognitionResult? {
        val store = embeddingsStore ?: return null
        val embedding = extractEmbedding(faceBitmap) ?: return null
        val match = matchEmbedding(embedding)
        return if (match != null) {
            Log.d(TAG, "Match embeddings: ${match.id} (sim=${"%.3f".format(match.similarity)})")
            RecognitionResult(match.id, match.similarity, -1, method = "embeddings")
        } else {
            Log.d(TAG, "Aucun match embeddings >= seuil $MATCH_THRESHOLD")
            null
        }
    }

    /** Méthode combinée: tente embeddings JSON puis fallback classifier */
    fun recognize(faceBitmap: Bitmap): RecognitionResult? {
        // Prioriser matching direct si store disponible
        recognizeByEmbeddings(faceBitmap)?.let { return it }
        return recognizeClassifier(faceBitmap)
    }

    /** Softmax */
    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val expValues = logits.map { exp((it - maxLogit).toDouble()).toFloat() }
        val sumExp = expValues.sum()
        return expValues.map { it / sumExp }.toFloatArray()
    }

    /** Libère les ressources */
    fun close() {
        ortSession?.close(); ortSession = null
        ortEnv = null
        embeddingExtractor?.close(); embeddingExtractor = null
        Log.d(TAG, "✓ Modèle ONNX, extracteur et store fermés")
    }
}

