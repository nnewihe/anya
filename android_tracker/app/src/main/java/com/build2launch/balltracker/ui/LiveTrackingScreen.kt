package com.build2launch.balltracker.ui

import android.Manifest
import android.app.Application
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewmodel.compose.viewModel
import com.build2launch.balltracker.detection.BallDetector
import com.build2launch.balltracker.live.CameraManager
import com.build2launch.balltracker.tracking.FrameResult
import com.build2launch.balltracker.tracking.TrackState
import com.build2launch.balltracker.tracking.TrackerEngine
import kotlin.math.roundToInt

class LiveTrackerViewModel(app: Application) : AndroidViewModel(app) {
    var result by mutableStateOf<FrameResult?>(null)
        private set
    var processedFps by mutableStateOf(0.0)
        private set
    var errorMessage by mutableStateOf<String?>(null)
        private set
    var backend by mutableStateOf("")
        private set

    val camera = CameraManager(app)
    private var engine: TrackerEngine? = null
    private var detector: BallDetector? = null
    private val frameTimes = ArrayList<Long>()

    fun startInference() {
        if (engine != null) return
        try {
            val det = BallDetector.fromAsset(getApplication())
            detector = det
            val eng = TrackerEngine(det, fps = 60.0)
            engine = eng
            backend = det.backend
            camera.onFrame = onFrame@{ bitmap, t ->
                val eng2 = engine ?: return@onFrame
                try {
                    val r = eng2.process(bitmap, t)
                    ingest(r)
                } catch (_: Throwable) {
                    // Drop a bad frame rather than crash the pipeline.
                } finally {
                    bitmap.recycle()
                }
            }
        } catch (t: Throwable) {
            errorMessage = "Model load failed: ${t.message}. " +
                "Run spikes/export_tflite.py and put ball_best.tflite in app/src/main/assets/."
        }
    }

    private fun ingest(r: FrameResult) {
        result = r
        val now = System.nanoTime()
        synchronized(frameTimes) {
            frameTimes.add(now)
            frameTimes.removeAll { now - it > 1_000_000_000L }
            processedFps = frameTimes.size.toDouble()
        }
    }

    fun stop() {
        camera.stop()
    }

    override fun onCleared() {
        camera.release()
        detector?.close()
    }
}

@Composable
fun LiveTrackingScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val model: LiveTrackerViewModel = viewModel()

    var hasPermission by remember {
        mutableStateOf(
            context.checkSelfPermission(Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED)
    }
    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()) { granted -> hasPermission = granted }

    LaunchedEffect(Unit) {
        if (!hasPermission) launcher.launch(Manifest.permission.CAMERA)
    }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        if (hasPermission) {
            val previewView = remember {
                PreviewView(context).apply {
                    scaleType = PreviewView.ScaleType.FIT_CENTER
                }
            }
            AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())

            DisposableEffect(Unit) {
                model.startInference()
                model.camera.start(lifecycleOwner, previewView)
                onDispose { model.stop() }
            }

            val result = model.result
            if (result != null) {
                BallOverlay(
                    frameWidth = result.frameWidth,
                    frameHeight = result.frameHeight,
                    trace = result.trace,
                    ballPosition = if (result.status.state != TrackState.NONE) result.ballPosition else null,
                    detections = result.detections.map {
                        OverlayBox(it.box.left, it.box.top, it.box.width(), it.box.height())
                    },
                    modifier = Modifier.fillMaxSize())
            }

            Column(Modifier.fillMaxSize().padding(12.dp)) {
                model.errorMessage?.let { ErrorBanner(it) }
                Box(Modifier.weight(1f))
                if (result != null) {
                    LiveHud(result, model.processedFps, model.backend,
                        modifier = Modifier.align(Alignment.CenterHorizontally))
                }
            }
        } else {
            Text(
                "Camera access is required.",
                color = Color.White,
                modifier = Modifier.align(Alignment.Center))
        }
    }
}

@Composable
private fun ErrorBanner(message: String) {
    Text(
        message,
        color = Color.White,
        modifier = Modifier
            .background(Color(0xCCCC3333), RoundedCornerShape(10.dp))
            .padding(12.dp))
}

@Composable
private fun LiveHud(
    result: FrameResult,
    processedFps: Double,
    backend: String,
    modifier: Modifier = Modifier,
) {
    val stateColor = when (result.status.state) {
        TrackState.MOVING -> Color(0xFF34C759)
        TrackState.COASTING -> Color(0xFFFFCC00)
        TrackState.FADING, TrackState.STOPPED -> Color(0xFFFF9500)
        TrackState.NONE, TrackState.LOST -> Color.Gray
    }
    Row(
        modifier
            .background(Color(0x99000000), CircleShape)
            .padding(horizontal = 16.dp, vertical = 10.dp),
        horizontalArrangement = Arrangement.spacedBy(14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(Modifier.background(stateColor, CircleShape).padding(5.dp))
            Mono(result.status.state.raw.replaceFirstChar { it.uppercase() })
        }
        Mono("${result.speedPxS.roundToInt()} px/s")
        Mono("${result.inferenceMs.roundToInt()} ms")
        Mono("${processedFps.roundToInt()} fps")
        if (backend.isNotEmpty()) Mono(backend)
    }
}

@Composable
private fun Mono(text: String) {
    Text(text, color = Color.White, fontFamily = FontFamily.Monospace)
}
