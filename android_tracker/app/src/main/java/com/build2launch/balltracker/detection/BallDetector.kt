package com.build2launch.balltracker.detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Build
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/** A decoded ball detection in source-frame pixel space. */
data class BallDetection(val box: RectF, val conf: Float) {
    val centerX: Float get() = box.centerX()
    val centerY: Float get() = box.centerY()
}

/**
 * Runs the NMS-free ball_best model on Android's built-in ML acceleration —
 * TensorFlow Lite (LiteRT) with the NNAPI delegate (NPU/DSP), falling back to
 * the GPU delegate, then multithreaded CPU (XNNPACK). Decode + single-class
 * NMS run on the CPU afterwards, exactly like the iOS BallDetector: keeping
 * NMS out of the graph keeps the whole conv net on the accelerator and the
 * ultra-low conf thresholds the tennis pipeline relies on tunable per call.
 *
 * Port of ios_tracker BallTracker/Detection/BallDetector.swift.
 */
class BallDetector private constructor(
    private val interpreter: Interpreter,
    private val nnApiDelegate: NnApiDelegate?,
    private val gpuDelegate: GpuDelegate?,
    /** The accelerator that actually backed this interpreter, for the HUD/logs. */
    val backend: String,
) : AutoCloseable {

    private val letterbox = Letterbox(inputWidth, inputHeight)

    // Output tensor geometry, read from the model at load time so the decode
    // adapts to whichever layout the exporter produced.
    private val outShape: IntArray = interpreter.getOutputTensor(0).shape()
    private val channelMajor: Boolean   // true: [1,5,N]; false: [1,N,5]
    private val numAnchors: Int
    private val outputBuffer: ByteBuffer

    /**
     * Ultralytics' TFLite export emits box coords normalized to [0,1]; the .pt
     * / CoreML paths emit pixels. Auto-detected on the first frame that carries
     * a real box and then cached, so parity holds whichever export is bundled.
     */
    private var coordsArePixels: Boolean? = null

    init {
        // Expect [1, 5, N] or [1, N, 5]; channel count is whichever axis is 5.
        require(outShape.size == 3 && outShape[0] == 1) {
            "unexpected output shape ${outShape.joinToString()}"
        }
        if (outShape[1] == 5) {
            channelMajor = true
            numAnchors = outShape[2]
        } else {
            channelMajor = false
            numAnchors = outShape[1]
        }
        outputBuffer = ByteBuffer
            .allocateDirect(outShape[1] * outShape[2] * 4)
            .order(ByteOrder.nativeOrder())
    }

    /**
     * Detect balls in a bitmap of any size; boxes come back in that bitmap's
     * pixel space. Thread-confined: call from a single inference thread.
     */
    fun detect(
        bitmap: Bitmap,
        conf: Float = defaultConf,
        iouThreshold: Float = defaultIoU,
        maxBoxPx: Float = defaultMaxBoxPx,
    ): List<BallDetection> {
        val transform = letterbox.apply(bitmap)
        outputBuffer.rewind()
        interpreter.run(letterbox.inputBuffer, outputBuffer)
        outputBuffer.rewind()
        return decode(outputBuffer.asFloatBuffer(), conf, iouThreshold, maxBoxPx, transform)
    }

    /** Value of channel `ch` for anchor `i` in the flat output buffer. */
    private fun get(fb: java.nio.FloatBuffer, ch: Int, i: Int): Float =
        if (channelMajor) fb.get(ch * numAnchors + i) else fb.get(i * 5 + ch)

    private fun decode(
        fb: java.nio.FloatBuffer,
        conf: Float,
        iouThreshold: Float,
        maxBoxPx: Float,
        transform: LetterboxTransform,
    ): List<BallDetection> {
        // Resolve pixel-vs-normalized coords once, from the strongest anchor.
        if (coordsArePixels == null) {
            var maxCoord = 0f
            var i = 0
            while (i < numAnchors) {
                val cx = get(fb, 0, i); val cy = get(fb, 1, i)
                if (cx > maxCoord) maxCoord = cx
                if (cy > maxCoord) maxCoord = cy
                i++
            }
            // Normalized coords live in [0,1]; a few px of slop → threshold 4.
            coordsArePixels = maxCoord > 4f
        }
        val sx = if (coordsArePixels == true) 1f else inputWidth.toFloat()
        val sy = if (coordsArePixels == true) 1f else inputHeight.toFloat()

        data class Cand(val box: RectF, val conf: Float)
        val candidates = ArrayList<Cand>()
        var i = 0
        while (i < numAnchors) {
            val c = get(fb, 4, i)
            if (c >= conf) {
                val cx = get(fb, 0, i) * sx
                val cy = get(fb, 1, i) * sy
                val w = get(fb, 2, i) * sx
                val h = get(fb, 3, i) * sy
                if (w <= maxBoxPx && h <= maxBoxPx) {
                    candidates.add(Cand(RectF(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), c))
                }
            }
            i++
        }

        val kept = nms(candidates.map { it.box to it.conf }, iouThreshold)
        return kept.map { (box, c) ->
            val ox = transform.unmapX(box.left)
            val oy = transform.unmapY(box.top)
            val bw = box.width() / transform.scale
            val bh = box.height() / transform.scale
            BallDetection(RectF(ox, oy, ox + bw, oy + bh), c)
        }
    }

    /** Greedy single-class non-maximum suppression, highest confidence first. */
    private fun nms(cands: List<Pair<RectF, Float>>, iouThreshold: Float): List<Pair<RectF, Float>> {
        if (cands.size <= 1) return cands
        val sorted = cands.sortedByDescending { it.second }
        val kept = ArrayList<Pair<RectF, Float>>()
        for (cand in sorted) {
            var suppressed = false
            for (k in kept) {
                if (iou(cand.first, k.first) > iouThreshold) { suppressed = true; break }
            }
            if (!suppressed) kept.add(cand)
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val ix = maxOf(a.left, b.left)
        val iy = maxOf(a.top, b.top)
        val ax = minOf(a.right, b.right)
        val ay = minOf(a.bottom, b.bottom)
        val iw = ax - ix
        val ih = ay - iy
        if (iw <= 0 || ih <= 0) return 0f
        val inter = iw * ih
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union > 0) inter / union else 0f
    }

    override fun close() {
        interpreter.close()
        nnApiDelegate?.close()
        gpuDelegate?.close()
    }

    companion object {
        private const val TAG = "BallDetector"
        const val inputWidth = 960
        const val inputHeight = 544

        /** ACTIVE_BALL_CONF — the online tracker's operating threshold. */
        const val defaultConf = 0.10f
        /** Offline Viterbi solver threshold (recall over precision). */
        const val solverConf = 0.10f
        /** Ultralytics' default NMS IoU, matching the parity fixtures. */
        const val defaultIoU = 0.7f
        /** Upper bound on a ball box, in model-input px (see iOS notes). */
        const val defaultMaxBoxPx = 45f

        const val ASSET_NAME = "ball_best.tflite"

        /** Load the bundled model, choosing the best available accelerator. */
        fun fromAsset(context: Context, assetName: String = ASSET_NAME): BallDetector {
            val model = loadModelBuffer(context, assetName)
            return build(model)
        }

        private fun build(model: ByteBuffer): BallDetector {
            // 1. NNAPI — Android's built-in NPU/DSP path (API 27+).
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                try {
                    val delegate = NnApiDelegate()
                    val opts = Interpreter.Options().addDelegate(delegate)
                    val interp = Interpreter(model, opts)
                    Log.i(TAG, "using NNAPI delegate")
                    return BallDetector(interp, delegate, null, "NNAPI")
                } catch (t: Throwable) {
                    Log.w(TAG, "NNAPI delegate unavailable: ${t.message}")
                }
            }
            // 2. GPU delegate.
            try {
                if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                    val delegate = GpuDelegate()
                    val opts = Interpreter.Options().addDelegate(delegate)
                    val interp = Interpreter(model, opts)
                    Log.i(TAG, "using GPU delegate")
                    return BallDetector(interp, null, delegate, "GPU")
                }
            } catch (t: Throwable) {
                Log.w(TAG, "GPU delegate unavailable: ${t.message}")
            }
            // 3. CPU (XNNPACK, multithreaded).
            val opts = Interpreter.Options().apply {
                numThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(4)
                useXNNPACK = true
            }
            Log.i(TAG, "using CPU (XNNPACK)")
            return BallDetector(Interpreter(model, opts), null, null, "CPU")
        }

        private fun loadModelBuffer(context: Context, assetName: String): ByteBuffer {
            context.assets.openFd(assetName).use { fd ->
                fd.createInputStream().channel.use { channel ->
                    return channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        fd.startOffset,
                        fd.declaredLength,
                    )
                }
            }
        }
    }
}
