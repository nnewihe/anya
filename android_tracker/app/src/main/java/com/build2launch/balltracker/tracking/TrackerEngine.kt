package com.build2launch.balltracker.tracking

import android.graphics.Bitmap
import com.build2launch.balltracker.detection.BallDetection
import com.build2launch.balltracker.detection.BallDetector
import com.build2launch.balltracker.detection.ExclusionZones

/**
 * One frame's combined result: raw detections plus the tracker's verdict.
 * Detections and derived positions are in source-frame pixel space.
 * Port of ios_tracker FrameResult.
 */
class FrameResult(
    val t: Double,
    val frameWidth: Int,
    val frameHeight: Int,
    val detections: List<BallDetection>,
    val status: TrackStatus,          // in analysis space (960-wide)
    val analysisScale: Double,        // source px * scale = analysis px
    val inferenceMs: Double,
) {
    /** Tracked ball position in source pixels, if tracked. */
    val ballPosition: Pair<Float, Float>?
        get() = status.position?.let {
            Pair((it.first / analysisScale).toFloat(), (it.second / analysisScale).toFloat())
        }

    /** Smoothed trajectory over the tracker's motion window, in source pixels. */
    val trace: List<Pair<Float, Float>>
        get() = status.trace.map {
            Pair((it.first / analysisScale).toFloat(), (it.second / analysisScale).toFloat())
        }

    /** Ball speed in source pixels/second. */
    val speedPxS: Double get() = status.speedPxS / analysisScale
}

/**
 * Detector + Kalman tracker glued together. The tracker's pixel-tuned gates
 * were validated in 960-wide analysis space, so detections are normalized into
 * that space before tracking regardless of source frame size.
 * Port of ios_tracker TrackerEngine.
 */
class TrackerEngine(
    private val detector: BallDetector,
    private val fps: Double,
    val confThreshold: Float = BallDetector.defaultConf,
) {
    private var manager: BallTrackManager? = null
    var analysisScale: Double = 1.0
        private set

    /** Stationary ball-like clutter to drop before tracking, in analysis space. */
    var exclusionZones: ExclusionZones = ExclusionZones.NONE

    val backend: String get() = detector.backend

    fun reset() {
        manager = null
    }

    /**
     * Detect and map into analysis space, dropping exclusion-zone clutter — no
     * online tracking. The offline video path uses this to gather every frame's
     * candidates before solving the trajectory globally.
     */
    fun analysisDetections(bitmap: Bitmap): Pair<List<ViterbiDetection>, Double> {
        val w = bitmap.width
        analysisScale = ANALYSIS_WIDTH / w

        val t0 = System.nanoTime()
        val detections = detector.detect(bitmap, conf = confThreshold)
        val inferenceMs = (System.nanoTime() - t0) / 1e6

        val dets = detections.mapNotNull { d ->
            val x = d.centerX * analysisScale
            val y = d.centerY * analysisScale
            if (exclusionZones.contains(x, y)) null else ViterbiDetection(x, y, d.conf.toDouble())
        }
        return Pair(dets, inferenceMs)
    }

    fun process(bitmap: Bitmap, t: Double): FrameResult {
        val w = bitmap.width
        val h = bitmap.height

        val t0 = System.nanoTime()
        val detections = detector.detect(bitmap, conf = confThreshold)
        val inferenceMs = (System.nanoTime() - t0) / 1e6

        var mgr = manager
        if (mgr == null) {
            analysisScale = ANALYSIS_WIDTH / w
            mgr = BallTrackManager(
                fps = fps,
                perspectiveScale = makeImageRowPerspective(frameHeight = h * analysisScale))
            manager = mgr
        }
        // Drop stationary clutter before it reaches the tracker.
        val tracked = detections.mapNotNull { d ->
            val x = d.centerX * analysisScale
            val y = d.centerY * analysisScale
            if (exclusionZones.contains(x, y)) null else TrackerDetection(x, y, d.conf.toDouble())
        }
        val status = mgr.update(tracked, t)

        return FrameResult(
            t = t,
            frameWidth = w,
            frameHeight = h,
            detections = detections,
            status = status,
            analysisScale = analysisScale,
            inferenceMs = inferenceMs)
    }

    companion object {
        const val ANALYSIS_WIDTH = 960.0
    }
}
