package com.build2launch.balltracker.video

import android.content.Context
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import android.net.Uri
import com.build2launch.balltracker.tracking.TrackState
import java.io.File
import java.nio.ByteBuffer

/** A contiguous keep-range of the source video, in seconds. */
class HighlightSegment(val start: Double, val end: Double) {
    val duration: Double get() = end - start
}

/**
 * Turns a [VideoAnalysis] trace into a stitched highlights reel.
 *
 * The segment logic is the exact port of ios_tracker/make_highlights.py
 * (== ios_tracker HighlightsExporter.segments):
 *   * a "live span" is a contiguous run of moving/coasting frames,
 *   * keep spans at least `minLen` seconds long,
 *   * merge kept spans whose gap is <= `mergeGap` seconds into one cut,
 *   * pad each cut by `pad` seconds for watchability.
 *
 * The stitch uses MediaExtractor + MediaMuxer (stream copy, no re-encode of the
 * picture) rather than ffmpeg; the kept time ranges are identical to the
 * command-line tool's.
 */
object HighlightsExporter {
    class Params(
        val minLen: Double = 1.5,
        val mergeGap: Double = 3.0,
        val pad: Double = 0.5,
    )

    class ExportException(message: String) : Exception(message)

    /** Compute keep-segments from the analysis. Pure and cheap. */
    fun segments(analysis: VideoAnalysis, params: Params = Params()): List<HighlightSegment> {
        // 1. Contiguous live spans.
        val spans = ArrayList<Pair<Double, Double>>()
        var runStart: Double? = null
        var prevT = 0.0
        for (s in analysis.samples) {
            val live = s.state == TrackState.MOVING || s.state == TrackState.COASTING
            if (live && runStart == null) {
                runStart = s.t
            } else if (!live && runStart != null) {
                spans.add(Pair(runStart!!, prevT))
                runStart = null
            }
            prevT = s.t
        }
        runStart?.let { spans.add(Pair(it, prevT)) }

        // 2. Keep the long-enough spans.
        val kept = spans.filter { it.second - it.first >= params.minLen }.sortedBy { it.first }
        if (kept.isEmpty()) return emptyList()

        // 3. Merge within mergeGap.
        val merged = ArrayList<DoubleArray>()
        merged.add(doubleArrayOf(kept[0].first, kept[0].second))
        for (i in 1 until kept.size) {
            val (s, e) = kept[i]
            val last = merged.last()
            if (s - last[1] <= params.mergeGap) {
                last[1] = maxOf(last[1], e)
            } else {
                merged.add(doubleArrayOf(s, e))
            }
        }

        // 4. Pad, clamp to the clip, and re-merge any overlap the pad created.
        val out = ArrayList<HighlightSegment>()
        for (m in merged) {
            val s = maxOf(0.0, m[0] - params.pad)
            val e = minOf(analysis.duration, m[1] + params.pad)
            val last = out.lastOrNull()
            if (last != null && s <= last.end) {
                out[out.size - 1] = HighlightSegment(last.start, maxOf(last.end, e))
            } else {
                out.add(HighlightSegment(s, e))
            }
        }
        return out
    }

    /** Build the reel and return the output file. Stream-copies video + audio. */
    fun export(
        context: Context,
        analysis: VideoAnalysis,
        params: Params = Params(),
    ): File {
        val segs = segments(analysis, params)
        if (segs.isEmpty()) throw ExportException(
            "No rallies long enough to include (need a tracked span ≥ ${params.minLen}s).")

        val extractor = MediaExtractor()
        extractor.setDataSource(context, analysis.uri, null)

        // Map source tracks -> muxer tracks (video required, audio if present).
        val trackMap = HashMap<Int, Int>()
        var maxInputSize = 0
        val out = File(context.cacheDir,
            "highlights_${System.currentTimeMillis()}.mp4")
        val muxer = MediaMuxer(out.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)

        var rotation = 0
        for (i in 0 until extractor.trackCount) {
            val fmt = extractor.getTrackFormat(i)
            val mime = fmt.getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("video/") || mime.startsWith("audio/")) {
                trackMap[i] = muxer.addTrack(fmt)
                if (fmt.containsKey(MediaFormat.KEY_MAX_INPUT_SIZE)) {
                    maxInputSize = maxOf(maxInputSize, fmt.getInteger(MediaFormat.KEY_MAX_INPUT_SIZE))
                }
                if (mime.startsWith("video/") && fmt.containsKey(MediaFormat.KEY_ROTATION)) {
                    rotation = fmt.getInteger(MediaFormat.KEY_ROTATION)
                }
            }
        }
        if (trackMap.isEmpty()) { extractor.release(); muxer.release()
            throw ExportException("The source video has no usable track.") }
        if (maxInputSize <= 0) maxInputSize = 1 shl 21 // 2 MB fallback

        muxer.setOrientationHint(rotation)
        muxer.start()

        val buffer = ByteBuffer.allocate(maxInputSize)
        val info = android.media.MediaCodec.BufferInfo()
        var ptsOffsetUs = 0L

        try {
            for (seg in segs) {
                val startUs = (seg.start * 1_000_000).toLong()
                val endUs = (seg.end * 1_000_000).toLong()
                var segFirstPtsUs = -1L
                var segLastPtsUs = startUs

                for ((srcTrack, dstTrack) in trackMap) {
                    extractor.selectTrack(srcTrack)
                    extractor.seekTo(startUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
                    while (true) {
                        buffer.clear()
                        val size = extractor.readSampleData(buffer, 0)
                        if (size < 0) break
                        val pts = extractor.sampleTime
                        if (pts > endUs) break
                        if (pts >= startUs) {
                            if (segFirstPtsUs < 0) segFirstPtsUs = pts
                            info.offset = 0
                            info.size = size
                            info.presentationTimeUs = pts - startUs + ptsOffsetUs
                            info.flags = extractor.sampleFlags
                            muxer.writeSampleData(dstTrack, buffer, info)
                            if (info.presentationTimeUs > segLastPtsUs) segLastPtsUs = info.presentationTimeUs
                        }
                        if (!extractor.advance()) break
                    }
                    extractor.unselectTrack(srcTrack)
                }
                // Advance the running offset by this segment's covered duration.
                ptsOffsetUs += (endUs - startUs)
            }
        } finally {
            try { muxer.stop() } catch (_: Throwable) {}
            muxer.release()
            extractor.release()
        }
        return out
    }
}
