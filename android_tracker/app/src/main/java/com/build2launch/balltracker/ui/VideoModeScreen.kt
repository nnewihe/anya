package com.build2launch.balltracker.ui

import android.app.Application
import android.content.Intent
import android.net.Uri
import android.widget.VideoView
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.FileProvider
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.build2launch.balltracker.detection.BallDetector
import com.build2launch.balltracker.video.HighlightsExporter
import com.build2launch.balltracker.video.VideoAnalysis
import com.build2launch.balltracker.video.VideoProcessor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

sealed interface VideoPhase {
    data object Idle : VideoPhase
    data class Processing(val progress: Double) : VideoPhase
    data class Done(val analysis: VideoAnalysis) : VideoPhase
    data class Failed(val message: String) : VideoPhase
}

class VideoModeViewModel(app: Application) : AndroidViewModel(app) {
    var phase by mutableStateOf<VideoPhase>(VideoPhase.Idle)
        private set
    var exportMessage by mutableStateOf<String?>(null)
        private set
    var exportedReel by mutableStateOf<File?>(null)
        private set

    private var detector: BallDetector? = null
    private var job: Job? = null

    private fun detector(): BallDetector =
        detector ?: BallDetector.fromAsset(getApplication()).also { detector = it }

    fun process(uri: Uri) {
        job?.cancel()
        exportedReel = null
        exportMessage = null
        phase = VideoPhase.Processing(0.0)
        job = viewModelScope.launch {
            try {
                val analysis = withContext(Dispatchers.Default) {
                    VideoProcessor(getApplication(), detector()).process(uri) { p ->
                        phase = VideoPhase.Processing(p)
                    }
                }
                phase = VideoPhase.Done(analysis)
            } catch (t: kotlinx.coroutines.CancellationException) {
                phase = VideoPhase.Idle
            } catch (t: Throwable) {
                phase = VideoPhase.Failed(t.message ?: "Processing failed")
            }
        }
    }

    fun cancel() {
        job?.cancel()
        phase = VideoPhase.Idle
    }

    fun exportHighlights(analysis: VideoAnalysis) {
        exportMessage = "Exporting…"
        viewModelScope.launch {
            try {
                val file = withContext(Dispatchers.Default) {
                    HighlightsExporter.export(getApplication(), analysis)
                }
                exportedReel = file
                exportMessage = null
            } catch (t: Throwable) {
                exportMessage = t.message ?: "Export failed"
            }
        }
    }

    override fun onCleared() {
        job?.cancel()
        detector?.close()
    }
}

@Composable
fun VideoModeScreen() {
    val model: VideoModeViewModel = viewModel()
    val context = LocalContext.current

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()) { uri -> uri?.let { model.process(it) } }

    fun pick() = picker.launch(
        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly))

    Column(Modifier.fillMaxSize().padding(top = 8.dp)) {
        when (val phase = model.phase) {
            is VideoPhase.Idle -> PickerPrompt(::pick)
            is VideoPhase.Processing -> ProcessingView(phase.progress, model::cancel)
            is VideoPhase.Done -> {
                AnalysisPlayer(phase.analysis, model, Modifier.weight(1f))
                PickAnotherBar(::pick)
            }
            is VideoPhase.Failed -> {
                Box(Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                    Text(phase.message, color = Color.White,
                        textAlign = TextAlign.Center, modifier = Modifier.padding(24.dp))
                }
                PickAnotherBar(::pick)
            }
        }
    }
}

@Composable
private fun PickerPrompt(onPick: () -> Unit) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Text("Pick a match video to track the ball offline.",
                color = Color.White, textAlign = TextAlign.Center,
                modifier = Modifier.padding(horizontal = 32.dp))
            Button(onClick = onPick) { Text("Choose Video") }
        }
    }
}

@Composable
private fun ProcessingView(progress: Double, onCancel: () -> Unit) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Text("Tracking ball… ${(progress * 100).toInt()}%", color = Color.White)
            LinearProgressIndicator(
                progress = { progress.toFloat() },
                modifier = Modifier.fillMaxWidth().padding(horizontal = 32.dp))
            Button(onClick = onCancel) { Text("Cancel") }
        }
    }
}

@Composable
private fun PickAnotherBar(onPick: () -> Unit) {
    Box(Modifier.fillMaxWidth().padding(vertical = 10.dp), contentAlignment = Alignment.Center) {
        Button(onClick = onPick) { Text("Choose Another Video") }
    }
}

@Composable
private fun AnalysisPlayer(
    analysis: VideoAnalysis,
    model: VideoModeViewModel,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    var positionSec by remember { mutableStateOf(0.0) }
    val videoView = remember {
        VideoView(context).apply {
            setVideoURI(analysis.uri)
            setOnPreparedListener { it.isLooping = true; start() }
        }
    }

    // Drive the overlay from the player's clock.
    LaunchedEffect(videoView) {
        while (true) {
            positionSec = videoView.currentPosition / 1000.0
            kotlinx.coroutines.delay(16)
        }
    }

    Column(modifier) {
        StatsBar(analysis)
        Box(Modifier.fillMaxWidth().weight(1f)) {
            AndroidView(factory = { videoView }, modifier = Modifier.fillMaxSize())
            BallOverlay(
                frameWidth = analysis.width,
                frameHeight = analysis.height,
                trace = analysis.trail(positionSec),
                ballPosition = analysis.sample(positionSec)?.pos,
                modifier = Modifier.fillMaxSize())
        }
        ExportBar(analysis, model, context)
    }
}

@Composable
private fun StatsBar(analysis: VideoAnalysis) {
    Row(
        Modifier.fillMaxWidth().padding(vertical = 6.dp),
        horizontalArrangement = Arrangement.SpaceEvenly,
    ) {
        Text("${(analysis.liveTraceRate * 100).toInt()}% live", color = Color.White)
        Text("${analysis.maxSpeedPxS.toInt()} px/s max", color = Color.White)
        Text("${analysis.avgInferenceMs.toInt()} ms/frame · ${analysis.backend}", color = Color.White)
    }
}

@Composable
private fun ExportBar(analysis: VideoAnalysis, model: VideoModeViewModel, context: android.content.Context) {
    val segs = remember(analysis) { HighlightsExporter.segments(analysis) }
    val reelSec = segs.sumOf { it.duration }
    val reel = model.exportedReel
    val msg = model.exportMessage

    Box(Modifier.fillMaxWidth().padding(8.dp), contentAlignment = Alignment.Center) {
        when {
            msg != null -> Text(msg, color = Color.White)
            reel != null -> Button(onClick = { shareReel(context, reel) }) {
                Text("Share Highlights Reel")
            }
            else -> Button(
                onClick = { model.exportHighlights(analysis) },
                enabled = segs.isNotEmpty(),
            ) {
                Text(if (segs.isEmpty()) "No rallies to export"
                     else "Export Highlights · ${segs.size} clips · ${reelSec.toInt()}s")
            }
        }
    }
}

private fun shareReel(context: android.content.Context, file: File) {
    val uri = FileProvider.getUriForFile(
        context, "${context.packageName}.fileprovider", file)
    val intent = Intent(Intent.ACTION_SEND).apply {
        type = "video/mp4"
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(Intent.createChooser(intent, "Share highlights").apply {
        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    })
}
