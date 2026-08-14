package com.build2launch.balltracker.ui

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.StrokeCap
import kotlin.math.min

/** A raw detection box in source-frame pixels. */
class OverlayBox(val left: Float, val top: Float, val width: Float, val height: Float)

/**
 * Draws the smoothed trajectory, the current ball marker, and any raw detection
 * boxes over an aspect-fit video rect. Shared by the live and video screens —
 * the Compose counterpart of ios_tracker BallOverlay / PlaybackOverlay.
 *
 * All geometry is given in source-frame pixels; this maps it onto the composable
 * the same way an aspect-fit (resizeAspect) video surface lays the picture out.
 */
@Composable
fun BallOverlay(
    frameWidth: Int,
    frameHeight: Int,
    trace: List<Pair<Float, Float>>,
    ballPosition: Pair<Float, Float>?,
    modifier: Modifier = Modifier,
    detections: List<OverlayBox> = emptyList(),
) {
    Canvas(modifier = modifier) {
        if (frameWidth <= 0 || frameHeight <= 0) return@Canvas
        val fit = aspectFit(Size(frameWidth.toFloat(), frameHeight.toFloat()), size)
        val s = fit.width / frameWidth
        fun map(px: Float, py: Float) = Offset(fit.x + px * s, fit.y + py * s)

        // Trajectory trail, fading toward the oldest point.
        if (trace.size >= 2) {
            for (i in 1 until trace.size) {
                val alpha = 0.15f + 0.85f * (i.toFloat() / (trace.size - 1))
                drawLine(
                    color = BallYellow.copy(alpha = alpha),
                    start = map(trace[i - 1].first, trace[i - 1].second),
                    end = map(trace[i].first, trace[i].second),
                    strokeWidth = 3f * density,
                    cap = StrokeCap.Round)
            }
        }

        // Raw detections (thin white boxes).
        for (d in detections) {
            val tl = map(d.left, d.top)
            drawRect(
                color = Color.White.copy(alpha = 0.6f),
                topLeft = tl,
                size = Size(d.width * s, d.height * s),
                style = Stroke(width = 1f * density))
        }

        // Current tracked position.
        if (ballPosition != null) {
            val p = map(ballPosition.first, ballPosition.second)
            drawCircle(
                color = BallYellow,
                radius = 10f * density,
                center = p,
                style = Stroke(width = 2.5f * density))
        }
    }
}

private class FitRect(val x: Float, val y: Float, val width: Float, val height: Float)

/** Largest rect of `content` aspect ratio that fits centered inside `bounds`. */
private fun aspectFit(content: Size, bounds: Size): FitRect {
    val r = min(bounds.width / content.width, bounds.height / content.height)
    val w = content.width * r
    val h = content.height * r
    return FitRect((bounds.width - w) / 2f, (bounds.height - h) / 2f, w, h)
}
