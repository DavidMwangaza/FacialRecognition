package com.example.facerecognition.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.tasks.await

/**
 * Détecteur de visages utilisant ML Kit de Google
 * Détection rapide et efficace sur l'appareil
 */
class FaceDetector(private val context: Context) {
    
    companion object {
        private const val TAG = "FaceDetector"
    }
    
    private val options = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
        .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
        .setMinFaceSize(0.15f) // Taille minimale du visage (15% de l'image)
        .enableTracking()
        .build()
    
    private val detector = FaceDetection.getClient(options)
    
    data class DetectedFace(
        val boundingBox: Rect,
        val bitmap: Bitmap,
        val trackingId: Int?
    )
    
    /**
     * Détecte les visages dans une image
     */
    suspend fun detectFaces(bitmap: Bitmap): List<DetectedFace> {
        return try {
            Log.d(TAG, "🔍 Début détection - Image: ${bitmap.width}x${bitmap.height}")
            val inputImage = InputImage.fromBitmap(bitmap, 0)
            val faces = detector.process(inputImage).await()
            
            Log.d(TAG, "✓ ${faces.size} visage(s) détecté(s)")
            
            if (faces.isEmpty()) {
                Log.w(TAG, "⚠ Aucun visage détecté! Vérifiez l'image et l'éclairage")
            }
            
            faces.mapNotNull { face ->
                Log.d(TAG, "Face bounds: ${face.boundingBox}")
                extractFace(bitmap, face)
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Erreur lors de la détection: ${e.message}", e)
            emptyList()
        }
    }
    
    /**
     * Extrait le visage de l'image
     */
    private fun extractFace(bitmap: Bitmap, face: Face): DetectedFace? {
        return try {
            val bounds = face.boundingBox
            
            // Vérifier que le rectangle est valide
            if (bounds.left < 0 || bounds.top < 0 ||
                bounds.right > bitmap.width || bounds.bottom > bitmap.height) {
                Log.w(TAG, "Rectangle de visage hors limites")
                return null
            }
            
            // Extraire la région du visage avec un padding
            val padding = 20
            val left = maxOf(0, bounds.left - padding)
            val top = maxOf(0, bounds.top - padding)
            val right = minOf(bitmap.width, bounds.right + padding)
            val bottom = minOf(bitmap.height, bounds.bottom + padding)
            
            val width = right - left
            val height = bottom - top
            
            if (width <= 0 || height <= 0) {
                Log.w(TAG, "Dimensions de visage invalides")
                return null
            }
            
            val faceBitmap = Bitmap.createBitmap(
                bitmap,
                left,
                top,
                width,
                height
            )
            
            DetectedFace(
                boundingBox = bounds,
                bitmap = faceBitmap,
                trackingId = face.trackingId
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Erreur lors de l'extraction du visage: ${e.message}", e)
            null
        }
    }
    
    /**
     * Détecte et reconnaît les visages
     */
    suspend fun detectAndRecognize(
        bitmap: Bitmap,
        recognitionModel: FaceRecognitionModel
    ): List<RecognitionResult> {
        val detectedFaces = detectFaces(bitmap)
        
        return detectedFaces.mapNotNull { detectedFace ->
            val result = recognitionModel.recognize(detectedFace.bitmap)
            
            result?.let {
                RecognitionResult(
                    name = it.name,
                    confidence = it.confidence,
                    boundingBox = detectedFace.boundingBox,
                    trackingId = detectedFace.trackingId
                )
            }
        }
    }
    
    data class RecognitionResult(
        val name: String,
        val confidence: Float,
        val boundingBox: Rect,
        val trackingId: Int?
    )
    
    /**
     * Libère les ressources
     */
    fun close() {
        detector.close()
        Log.d(TAG, "✓ Détecteur fermé")
    }
}
