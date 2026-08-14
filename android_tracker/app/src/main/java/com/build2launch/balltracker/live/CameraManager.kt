package com.build2launch.balltracker.live

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.Executors

/**
 * Owns the CameraX use-cases and hands frames (as display-oriented Bitmaps with
 * a monotonic timestamp) to a callback on a dedicated analysis thread. With
 * `STRATEGY_KEEP_ONLY_LATEST` CameraX drops frames that arrive while inference
 * is busy, so the tracker always sees the freshest frame — the same contract as
 * the iOS `alwaysDiscardsLateVideoFrames` path.
 *
 * Android counterpart of ios_tracker CameraManager.
 */
class CameraManager(private val context: Context) {

    /** Called on the analysis thread for every frame that isn't dropped. */
    var onFrame: ((Bitmap, Double) -> Unit)? = null

    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private var provider: ProcessCameraProvider? = null

    fun start(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            val cameraProvider = future.get()
            provider = cameraProvider

            val preview = Preview.Builder().build().also {
                it.surfaceProvider = previewView.surfaceProvider
            }

            val resolution = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                .build()

            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolution)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            analysis.setAnalyzer(analysisExecutor) { proxy -> handleFrame(proxy) }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(context))
    }

    private fun handleFrame(proxy: ImageProxy) {
        try {
            val bmp = proxy.toBitmap()
            val rotation = proxy.imageInfo.rotationDegrees
            val oriented = if (rotation != 0) {
                val m = Matrix().apply { postRotate(rotation.toFloat()) }
                Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
            } else bmp
            val tSec = proxy.imageInfo.timestamp / 1e9
            onFrame?.invoke(oriented, tSec)
        } finally {
            proxy.close()
        }
    }

    fun stop() {
        provider?.unbindAll()
    }

    fun release() {
        provider?.unbindAll()
        analysisExecutor.shutdown()
    }
}
