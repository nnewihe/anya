package com.build2launch.balltracker.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

/** The tennis-ball accent used across the app (matches iOS BallOverlay.ballYellow). */
val BallYellow = Color(0xFFDEFF29)

private val DarkColors = darkColorScheme(
    primary = BallYellow,
    onPrimary = Color.Black,
    background = Color.Black,
    surface = Color(0xFF101010),
)

@Composable
fun BallTrackerTheme(content: @Composable () -> Unit) {
    // The app is intentionally always dark (iOS `.preferredColorScheme(.dark)`).
    @Suppress("UNUSED_EXPRESSION")
    isSystemInDarkTheme()
    MaterialTheme(colorScheme = DarkColors, content = content)
}
