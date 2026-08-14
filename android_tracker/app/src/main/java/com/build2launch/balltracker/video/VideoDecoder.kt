package com.build2launch.balltracker.video

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.nio.ByteBuffer

/**
 * Decodes every frame of a video to an ARGB [Bitmap] with its presentation
 * timestamp, headlessly (no output Surface), so the whole clip can be run
 * through the detector offline. Android counterpart of the AVAssetReader pass
 * in ios_tracker VideoProcessor.
 *
 * MediaCodec is asked for a flexible YUV_420_888 output and each frame is
 * converted to ARGB in software; frames are rotated by the track's rotation
 * metadata so overlay math matches what a player displays.
 */
class VideoDecoder(context: Context, uri: Uri) : AutoCloseable {
    private val extractor = MediaExtractor()
    private val codec: MediaCodec
    private val rotationDegrees: Int
    val fps: Double
    val durationSec: Double
    /** Display-oriented output size (after rotation). */
    val outWidth: Int
    val outHeight: Int

    private val bufferInfo = MediaCodec.BufferInfo()
    private var inputDone = false
    private var outputDone = false

    init {
        extractor.setDataSource(context, uri, null)
        var trackIndex = -1
        var format: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val f = extractor.getTrackFormat(i)
            if (f.getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true) {
                trackIndex = i; format = f; break
            }
        }
        require(trackIndex >= 0 && format != null) { "no video track" }
        extractor.selectTrack(trackIndex)

        val srcW = format.getInteger(MediaFormat.KEY_WIDTH)
        val srcH = format.getInteger(MediaFormat.KEY_HEIGHT)
        rotationDegrees = if (format.containsKey(MediaFormat.KEY_ROTATION))
            format.getInteger(MediaFormat.KEY_ROTATION) else 0
        if (rotationDegrees == 90 || rotationDegrees == 270) {
            outWidth = srcH; outHeight = srcW
        } else {
            outWidth = srcW; outHeight = srcH
        }
        fps = if (format.containsKey(MediaFormat.KEY_FRAME_RATE))
            format.getInteger(MediaFormat.KEY_FRAME_RATE).toDouble() else 30.0
        durationSec = if (format.containsKey(MediaFormat.KEY_DURATION))
            format.getLong(MediaFormat.KEY_DURATION) / 1e6 else 0.0

        format.setInteger(MediaFormat.KEY_COLOR_FORMAT,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
        codec = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
        codec.configure(format, null, null, 0)
        codec.start()
    }

    /**
     * Pull the next decoded frame. Returns null at end of stream. `onFrame` is
     * NOT retained — copy the bitmap if you need it past the next call.
     */
    fun nextFrame(): Frame? {
        while (!outputDone) {
            if (!inputDone) {
                val inIndex = codec.dequeueInputBuffer(TIMEOUT_US)
                if (inIndex >= 0) {
                    val inBuf = codec.getInputBuffer(inIndex)!!
                    val size = extractor.readSampleData(inBuf, 0)
                    if (size < 0) {
                        codec.queueInputBuffer(inIndex, 0, 0, 0,
                            MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inputDone = true
                    } else {
                        codec.queueInputBuffer(inIndex, 0, size, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
            }

            val outIndex = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US)
            when {
                outIndex >= 0 -> {
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        outputDone = true
                    }
                    if (bufferInfo.size > 0) {
                        val image = codec.getOutputImage(outIndex)
                        val tSec = bufferInfo.presentationTimeUs / 1e6
                        if (image != null) {
                            val bmp = yuvToBitmap(image)
                            image.close()
                            codec.releaseOutputBuffer(outIndex, false)
                            val rotated = if (rotationDegrees != 0) rotate(bmp, rotationDegrees) else bmp
                            return Frame(rotated, tSec)
                        }
                    }
                    codec.releaseOutputBuffer(outIndex, false)
                }
                outIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (inputDone) { /* keep draining */ }
                }
            }
        }
        return null
    }

    class Frame(val bitmap: Bitmap, val tSec: Double)

    override fun close() {
        try { codec.stop() } catch (_: Throwable) {}
        codec.release()
        extractor.release()
    }

    private fun rotate(src: Bitmap, degrees: Int): Bitmap {
        val m = Matrix().apply { postRotate(degrees.toFloat()) }
        val out = Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
        if (out != src) src.recycle()
        return out
    }

    /** Software YUV_420_888 -> ARGB_8888. Handles arbitrary pixel/row strides. */
    private fun yuvToBitmap(image: android.media.Image): Bitmap {
        val w = image.width
        val h = image.height
        val yP = image.planes[0]
        val uP = image.planes[1]
        val vP = image.planes[2]
        val yBuf = yP.buffer
        val uBuf = uP.buffer
        val vBuf = vP.buffer
        val yRow = yP.rowStride
        val uRow = uP.rowStride
        val vRow = vP.rowStride
        val uPix = uP.pixelStride
        val vPix = vP.pixelStride

        val out = IntArray(w * h)
        var idx = 0
        for (row in 0 until h) {
            val yBase = row * yRow
            val cBase = (row shr 1)
            val uBase = cBase * uRow
            val vBase = cBase * vRow
            for (col in 0 until w) {
                val yv = (yBuf.get(yBase + col).toInt() and 0xFF)
                val cCol = col shr 1
                val u = (uBuf.get(uBase + cCol * uPix).toInt() and 0xFF) - 128
                val v = (vBuf.get(vBase + cCol * vPix).toInt() and 0xFF) - 128
                // BT.601 full-range-ish conversion.
                var r = yv + ((91881 * v) shr 16)
                var g = yv - ((22554 * u + 46802 * v) shr 16)
                var b = yv + ((116130 * u) shr 16)
                if (r < 0) r = 0 else if (r > 255) r = 255
                if (g < 0) g = 0 else if (g > 255) g = 255
                if (b < 0) b = 0 else if (b > 255) b = 255
                out[idx++] = -0x1000000 or (r shl 16) or (g shl 8) or b
            }
        }
        return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
    }

    companion object {
        private const val TIMEOUT_US = 10_000L
    }
}
