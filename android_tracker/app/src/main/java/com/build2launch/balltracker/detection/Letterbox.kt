package com.build2launch.balltracker.detection

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * The uniform-scale + centered-pad transform mapping source-frame pixels into
 * the model's input, so detections can be mapped back out.
 * Port of ios_tracker LetterboxTransform.
 */
data class LetterboxTransform(val scale: Float, val padX: Float, val padY: Float) {
    /** Model-input px -> source px. */
    fun unmapX(x: Float): Float = (x - padX) / scale
    fun unmapY(y: Float): Float = (y - padY) / scale
}

/**
 * Aspect-fit letterbox of any Bitmap into a fixed-size buffer with the YOLO
 * gray-114 padding, then packed into a float32 NHWC tensor normalized to
 * [0,1] RGB — exactly what the Ultralytics TFLite export expects. Mirrors the
 * Ultralytics letterbox (uniform scale, centered pad) the parity fixtures were
 * generated with, matching ios_tracker Letterbox.swift.
 *
 * The intermediate ARGB bitmap and the direct float buffer are reused across
 * calls so the steady-state live path allocates nothing.
 */
class Letterbox(val width: Int, val height: Int) {
    private val canvasBitmap: Bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(canvasBitmap)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val pixels = IntArray(width * height)
    // NHWC float32: 1 * H * W * 3 * 4 bytes.
    val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(width * height * 3 * 4).order(ByteOrder.nativeOrder())
    private val gray114 = Color.rgb(114, 114, 114)

    /**
     * Letterbox `src` into [inputBuffer] (rewound, ready to feed the
     * interpreter) and return the transform to map detections back.
     */
    fun apply(src: Bitmap): LetterboxTransform {
        val sw = src.width
        val sh = src.height
        val r = min(width.toFloat() / sw, height.toFloat() / sh)
        val newW = (sw * r).roundToInt()
        val newH = (sh * r).roundToInt()
        val padX = (width - newW) / 2f
        val padY = (height - newH) / 2f

        canvas.drawColor(gray114)
        val dst = Rect(padX.roundToInt(), padY.roundToInt(),
            padX.roundToInt() + newW, padY.roundToInt() + newH)
        canvas.drawBitmap(src, null, dst, paint)

        canvasBitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val buf = inputBuffer
        buf.rewind()
        val fb = buf.asFloatBuffer()
        var i = 0
        val n = width * height
        while (i < n) {
            val p = pixels[i]
            fb.put(((p shr 16) and 0xFF) / 255f) // R
            fb.put(((p shr 8) and 0xFF) / 255f)  // G
            fb.put((p and 0xFF) / 255f)          // B
            i++
        }
        buf.rewind()
        return LetterboxTransform(r, padX, padY)
    }
}
