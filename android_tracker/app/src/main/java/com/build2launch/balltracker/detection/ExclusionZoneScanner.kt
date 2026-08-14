package com.build2launch.balltracker.detection

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.content.Context
import android.util.Log

/**
 * Sample frames across a clip, detect balls, and cluster the centres that keep
 * landing in the same place into exclusion rects. Defaults mirror the
 * production call in anya_base.py (num_frames=50, conf=0.04, eps=12,
 * padding=0). Android counterpart of ios_tracker ExclusionZoneScanner.
 *
 * Deviation from the pipeline (shared with iOS): it scans at native size rather
 * than BALL_IMGSZ=1920, so very faint clutter the 1920 scan would catch can
 * slip through; a basket still detects at ~0.8 even at 960.
 */
object ExclusionZoneScanner {
    private const val TAG = "ExclusionZoneScanner"

    fun scan(
        context: Context,
        uri: Uri,
        detector: BallDetector,
        analysisScale: Double,       // source px * scale = analysis px
        sampleCount: Int = 50,
        conf: Float = 0.04f,
        eps: Double = 12.0,
        minSamples: Int = 15,
        padding: Double = 0.0,
    ): ExclusionZones {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, uri)
            val durationMs = retriever.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: return ExclusionZones.NONE
            if (durationMs <= 0) return ExclusionZones.NONE

            val centres = ArrayList<Pair<Double, Double>>()
            for (i in 0 until sampleCount) {
                val f = (i + 0.5) / sampleCount
                val usec = (durationMs * 1000 * f).toLong()
                val frame: Bitmap = retriever.getFrameAtTime(
                    usec, MediaMetadataRetriever.OPTION_CLOSEST_SYNC) ?: continue
                val dets = try {
                    detector.detect(frame, conf = conf)
                } catch (t: Throwable) {
                    Log.w(TAG, "detect failed on sample $i: ${t.message}"); continue
                }
                for (d in dets) {
                    centres.add(Pair(d.centerX * analysisScale, d.centerY * analysisScale))
                }
            }
            val zones = clusterZones(centres, eps, minSamples, padding)
            Log.i(TAG, "${zones.rects.size} exclusion zone(s)")
            return zones
        } catch (t: Throwable) {
            Log.w(TAG, "scan failed: ${t.message}")
            return ExclusionZones.NONE
        } finally {
            retriever.release()
        }
    }
}
