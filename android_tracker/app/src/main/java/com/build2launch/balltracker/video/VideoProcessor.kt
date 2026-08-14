package com.build2launch.balltracker.video

import android.content.Context
import android.net.Uri
import com.build2launch.balltracker.detection.BallDetector
import com.build2launch.balltracker.detection.ExclusionZoneScanner
import com.build2launch.balltracker.tracking.TrackState
import com.build2launch.balltracker.tracking.TrackerEngine
import com.build2launch.balltracker.tracking.ViterbiBallTracker
import com.build2launch.balltracker.tracking.ViterbiConfig
import com.build2launch.balltracker.tracking.ViterbiDetection
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.ensureActive

/** Result of a full offline pass over a video: one sample per decoded frame. */
class VideoAnalysis(
    val uri: Uri,
    val width: Int,
    val height: Int,
    val fps: Double,
    val duration: Double,
    val samples: List<Sample>,
    val avgInferenceMs: Double,
    val backend: String,
) {
    class Sample(
        val t: Double,
        val pos: Pair<Float, Float>?,   // tracked ball, source px (null when no track)
        val state: TrackState,
        val speedPxS: Double,
    )

    /** Fraction of frames with a live moving/coasting trace. */
    val liveTraceRate: Double
        get() = if (samples.isEmpty()) 0.0 else
            samples.count { it.state == TrackState.MOVING || it.state == TrackState.COASTING }
                .toDouble() / samples.size

    val maxSpeedPxS: Double get() = samples.maxOfOrNull { it.speedPxS } ?: 0.0

    /** Live trail points in `(t - window, t]`, for drawing at playback time t. */
    fun trail(t: Double, window: Double = 0.5): List<Pair<Float, Float>> =
        samples.filter { it.t > t - window && it.t <= t }
            .mapNotNull { s ->
                if (s.state == TrackState.MOVING || s.state == TrackState.COASTING) s.pos else null
            }

    /** Last sample at or before t (binary search; samples are time-ordered). */
    fun sample(t: Double): Sample? {
        var lo = 0; var hi = samples.size - 1; var best = -1
        while (lo <= hi) {
            val mid = (lo + hi) / 2
            if (samples[mid].t <= t) { best = mid; lo = mid + 1 } else hi = mid - 1
        }
        return if (best >= 0) samples[best] else null
    }
}

/**
 * Decodes a video and runs every frame through the same detector + tracker
 * core as the live camera path, then solves the whole clip's trajectory with
 * the offline Viterbi solver. Port of ios_tracker VideoProcessor.
 */
class VideoProcessor(
    private val context: Context,
    private val detector: BallDetector,
    private val viterbiConfig: ViterbiConfig = ViterbiConfig(),
) {
    suspend fun process(
        uri: Uri,
        conf: Float = BallDetector.solverConf,
        progress: (Double) -> Unit,
    ): VideoAnalysis {
        val decoder = VideoDecoder(context, uri)
        val fps = if (decoder.fps > 0) decoder.fps else 30.0
        val duration = decoder.durationSec
        val engine = TrackerEngine(detector, fps = fps, confThreshold = conf)

        // Fence off stationary ball-like clutter first (baskets, balls at rest),
        // the way anya_base.py does at startup. Reuses this engine's detector.
        val scanScale = TrackerEngine.ANALYSIS_WIDTH / decoder.outWidth
        engine.exclusionZones = ExclusionZoneScanner.scan(
            context, uri, detector, analysisScale = scanScale)

        // Pass 1 — detect every frame; commit to no associations.
        val perFrame = ArrayList<List<ViterbiDetection>>()
        val times = ArrayList<Double>()
        var inferenceTotal = 0.0
        var lastProgressT = -1.0

        try {
            while (true) {
                coroutineContext.ensureActive()
                val frame = decoder.nextFrame() ?: break
                val (dets, ms) = engine.analysisDetections(frame.bitmap)
                frame.bitmap.recycle()
                inferenceTotal += ms
                perFrame.add(dets)
                times.add(frame.tSec)
                if (duration > 0 && frame.tSec - lastProgressT > 0.25) {
                    lastProgressT = frame.tSec
                    progress((frame.tSec / duration).coerceAtMost(1.0) * 0.95)
                }
            }
        } finally {
            decoder.close()
        }

        // Pass 2 — solve the trajectory over the whole clip.
        coroutineContext.ensureActive()
        val solved = ViterbiBallTracker(viterbiConfig).solve(perFrame, times)

        val toSource = if (engine.analysisScale > 0) 1.0 / engine.analysisScale else 1.0
        val samples = solved.map { s ->
            VideoAnalysis.Sample(
                t = s.t,
                pos = s.pos?.let {
                    Pair((it.first * toSource).toFloat(), (it.second * toSource).toFloat())
                },
                state = s.state,
                speedPxS = s.speedPxS * toSource)
        }
        progress(1.0)

        return VideoAnalysis(
            uri = uri,
            width = decoder.outWidth,
            height = decoder.outHeight,
            fps = fps,
            duration = duration,
            samples = samples,
            avgInferenceMs = if (times.isEmpty()) 0.0 else inferenceTotal / times.size,
            backend = detector.backend)
    }
}
