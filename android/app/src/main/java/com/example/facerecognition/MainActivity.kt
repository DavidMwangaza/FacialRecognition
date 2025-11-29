package com.example.facerecognition

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.facerecognition.databinding.ActivityMainBinding
import com.example.facerecognition.ml.FaceDetector
import com.example.facerecognition.ml.FaceRecognitionModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private var imageCapture: ImageCapture? = null
    private var cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
    private lateinit var cameraExecutor: ExecutorService
    
    // Modèles ML
    private var faceRecognitionModel: FaceRecognitionModel? = null
    private var faceDetector: FaceDetector? = null
    
    private var isProcessing = false
    private var modelsInitialized = false
    
    companion object {
        private const val TAG = "MainActivity"
    }
    
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(this, "Permission caméra refusée", Toast.LENGTH_SHORT).show()
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d(TAG, "onCreate démarré")
        
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        Log.d(TAG, "View binding créé")
        
        cameraExecutor = Executors.newSingleThreadExecutor()
        
        // Initialiser les modèles ML
        Log.d(TAG, "Appel de initializeModels()...")
        initializeModels()
        Log.d(TAG, "initializeModels() terminé")
        
        setupAppBar()
        setupUI()
        checkPermissions()
        Log.d(TAG, "onCreate terminé")
    }
    
    private fun initializeModels() {
        try {
            Log.d(TAG, "Initialisation des modèles ML...")
            faceRecognitionModel = FaceRecognitionModel(this)
            Log.d(TAG, "FaceRecognitionModel créé")
            faceDetector = FaceDetector(this)
            Log.d(TAG, "FaceDetector créé")
            modelsInitialized = true
            
            // Activer le bouton de capture
            binding.btnCapture.post {
                binding.btnCapture.isEnabled = true
                binding.btnCapture.isClickable = true
                binding.btnCapture.alpha = 1.0f
                Log.d(TAG, "Bouton capture activé: enabled=${binding.btnCapture.isEnabled}, clickable=${binding.btnCapture.isClickable}")
            }
            
                showMessage("Modèles chargés")
                Log.d(TAG, "Modèles ML initialisés avec succès")
        } catch (e: Exception) {
                Log.e(TAG, "Erreur lors de l'initialisation des modèles: ${e.message}", e)
            e.printStackTrace()
            modelsInitialized = false
            faceRecognitionModel = null
            faceDetector = null
            
            binding.btnCapture.post {
                binding.btnCapture.isEnabled = true  // Activé quand même pour tester
                binding.btnCapture.isClickable = true
                binding.btnCapture.alpha = 0.5f
            }
            
            showMessage("Erreur: ${e.message}")
        }
    }

    private fun setupAppBar() {
        setSupportActionBar(binding.topAppBar)
        binding.topAppBar.setNavigationOnClickListener {
            flipCamera()
        }
        binding.topAppBar.setOnMenuItemClickListener { item ->
            when (item.itemId) {
                R.id.action_settings -> {
                    showMessage("Paramètres")
                    true
                }
                R.id.action_about -> {
                    showMessage("À propos")
                    true
                }
                else -> false
            }
        }
    }
    
    private fun setupUI() {
        Log.d(TAG, "setupUI - Configuration des boutons")
        
        // S'assurer que le bouton est cliquable
        binding.btnCapture.isClickable = true
        binding.btnCapture.isFocusable = true
        
        binding.btnCapture.setOnClickListener {
            Log.d(TAG, "Bouton Capture cliqué! modelsInitialized=$modelsInitialized, isEnabled=${binding.btnCapture.isEnabled}")
            showMessage("Bouton cliqué!")
            if (modelsInitialized) {
                Log.d(TAG, "Appel de capturePhoto()")
                capturePhoto()
            } else {
                Log.w(TAG, "Modèles non initialisés")
                showMessage("Modèles non chargés")
            }
        }
        
        binding.btnFlip.setOnClickListener {
            Log.d(TAG, "Bouton Flip cliqué")
            flipCamera()
        }
        
        binding.btnRetake.setOnClickListener {
            Log.d(TAG, "Bouton Retake cliqué")
            resetCapture()
        }
        
        Log.d(TAG, "Boutons configurés")
    }
    
    private fun checkPermissions() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                startCamera()
            }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.previewView.surfaceProvider)
                }
            
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()
            
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageCapture
                )
            } catch (e: Exception) {
                Log.e(TAG, "Erreur de démarrage de la caméra", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun capturePhoto() {
        Log.d(TAG, "capturePhoto() appelé")
        if (isProcessing) {
            Log.w(TAG, "Déjà en traitement")
            showMessage("Traitement en cours...")
            return
        }
        
        val imageCapture = imageCapture ?: run {
            Log.e(TAG, "imageCapture est null!")
            showMessage("Caméra non prête")
            return
        }
        
        Log.d(TAG, "Prise de photo...")
        
        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    val bitmap = imageProxyToBitmap(image)
                    image.close()
                    
                    if (bitmap != null) {
                        binding.capturedImageView.setImageBitmap(bitmap)
                        binding.previewView.visibility = View.GONE
                        binding.capturedImageView.visibility = View.VISIBLE
                        binding.btnCapture.visibility = View.GONE
                        binding.btnFlip.visibility = View.GONE
                        binding.btnRetake.visibility = View.VISIBLE
                        
                        // Lancer la reconnaissance
                        recognizeFaces(bitmap)
                    }
                }
                
                override fun onError(exception: ImageCaptureException) {
                    Log.e(TAG, "Erreur capture", exception)
                    showMessage("Erreur lors de la capture")
                }
            }
        )
    }
    
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        val buffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        
        var bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        
        // Corriger la rotation
        if (bitmap != null && image.imageInfo.rotationDegrees != 0) {
            val matrix = Matrix()
            matrix.postRotate(image.imageInfo.rotationDegrees.toFloat())
            bitmap = Bitmap.createBitmap(
                bitmap, 0, 0,
                bitmap.width, bitmap.height,
                matrix, true
            )
        }
        
        return bitmap
    }
    
    private fun recognizeFaces(bitmap: Bitmap) {
        // Vérifier que les modèles sont initialisés
        if (!modelsInitialized || faceDetector == null || faceRecognitionModel == null) {
            binding.resultsText.text = "Erreur: Modèles non chargés. Redémarrez l'application."
            binding.resultsCard.visibility = View.VISIBLE
            showMessage("Modèles non chargés")
            return
        }
        
        isProcessing = true
        binding.progressBar.visibility = View.VISIBLE
        binding.loadingText.visibility = View.VISIBLE
        binding.resultsCard.visibility = View.GONE
        
        Log.d(TAG, "Début reconnaissance - Image: ${bitmap.width}x${bitmap.height}")
        
        lifecycleScope.launch {
            try {
                val results = withContext(Dispatchers.Default) {
                    Log.d(TAG, "Lancement détection...")
                    faceDetector!!.detectAndRecognize(bitmap, faceRecognitionModel!!)
                }
                
                Log.d(TAG, "Résultats obtenus: ${results.size} visage(s)")
                
                // Afficher les résultats
                displayResults(results, bitmap)
                
            } catch (e: Exception) {
                Log.e(TAG, "Erreur reconnaissance", e)
                binding.resultsText.text = "Erreur: ${e.message}\n${e.stackTraceToString()}"
                binding.resultsCard.visibility = View.VISIBLE
            } finally {
                isProcessing = false
                binding.progressBar.visibility = View.GONE
                binding.loadingText.visibility = View.GONE
            }
        }
    }
    
    private fun displayResults(results: List<FaceDetector.RecognitionResult>, originalBitmap: Bitmap) {
        binding.resultsCard.visibility = View.VISIBLE
        if (results.isEmpty()) {
            binding.resultsText.text = "Aucun visage détecté"
            showMessage("Aucun visage trouvé")
            // Do not show the annotated image if no faces are detected.
            binding.capturedImageView.setImageBitmap(originalBitmap)
            return
        }
        
        // Dessiner les rectangles sur l'image
        val annotatedBitmap = drawFaceBoxes(originalBitmap, results)
        binding.capturedImageView.setImageBitmap(annotatedBitmap)
        
        // Afficher les résultats textuels
        val resultText = buildString {
            append("${results.size} visage(s) détecté(s)\n\n")
            results.forEachIndexed { index, result ->
                append("Visage ${index + 1}:\n")
                append("   Nom: ${result.name}\n")
                append("   Confiance: ${String.format(Locale.US, "%.1f", result.confidence * 100)}%\n\n")
            }
        }
        
        binding.resultsText.text = resultText
        
        // Message toast
        val names = results.joinToString(", ") { it.name }
        showMessage("Reconnu: $names")
    }
    
    private fun drawFaceBoxes(
        bitmap: Bitmap,
        results: List<FaceDetector.RecognitionResult>
    ): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        
        val paint = Paint().apply {
            color = Color.GREEN
            style = Paint.Style.STROKE
            strokeWidth = 5f
        }
        
        val textPaint = Paint().apply {
            color = Color.GREEN
            textSize = 40f
            style = Paint.Style.FILL
        }
        
        val backgroundPaint = Paint().apply {
            color = Color.BLACK
            alpha = 150
            style = Paint.Style.FILL
        }
        
        results.forEach { result ->
            val box = result.boundingBox
            
            // Dessiner le rectangle
            canvas.drawRect(box, paint)
            
            // Dessiner le texte avec fond
            val text = "${result.name} (${String.format(Locale.US, "%.0f", result.confidence * 100)}%)"
            val textBounds = Rect()
            textPaint.getTextBounds(text, 0, text.length, textBounds)
            
            val textX = box.left.toFloat()
            val textY = (box.top - 10).toFloat()
            
            canvas.drawRect(
                textX,
                textY - textBounds.height() - 10,
                textX + textBounds.width() + 10,
                textY + 5,
                backgroundPaint
            )
            
            canvas.drawText(text, textX + 5, textY, textPaint)
        }
        
        return mutableBitmap
    }
    
    private fun flipCamera() {
        cameraSelector = if (cameraSelector == CameraSelector.DEFAULT_FRONT_CAMERA) {
            CameraSelector.DEFAULT_BACK_CAMERA
        } else {
            CameraSelector.DEFAULT_FRONT_CAMERA
        }
        startCamera()
    }
    
    private fun resetCapture() {
        binding.previewView.visibility = View.VISIBLE
        binding.capturedImageView.visibility = View.GONE
        binding.resultsCard.visibility = View.GONE
        binding.btnCapture.visibility = View.VISIBLE
        binding.btnFlip.visibility = View.VISIBLE
        binding.btnRetake.visibility = View.GONE
        binding.resultsText.text = ""
    }
    
    private fun showMessage(message: String) {
        runOnUiThread {
            Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        
        // Libérer les ressources ML
        try {
            faceRecognitionModel?.close()
            faceDetector?.close()
            Log.d(TAG, "Ressources ML libérées")
        } catch (e: Exception) {
            Log.e(TAG, "Erreur libération ressources", e)
        }
    }

    // Menus
    override fun onCreateOptionsMenu(menu: android.view.Menu?): Boolean {
        menuInflater.inflate(R.menu.top_app_bar_menu, menu)
        return true
    }
    
    // Animations simples pour moderniser les transitions
    private fun fadeIn(view: View) {
        view.apply {
            alpha = 0f
            visibility = View.VISIBLE
            animate().alpha(1f).setDuration(200).start()
        }
    }
    private fun fadeOut(view: View) {
        view.animate().alpha(0f).setDuration(200).withEndAction {
            view.visibility = View.GONE
            view.alpha = 1f
        }.start()
    }
}
