package com.build2launch.balltracker

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CenterFocusWeak
import androidx.compose.material.icons.filled.Movie
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import com.build2launch.balltracker.ui.BallTrackerTheme
import com.build2launch.balltracker.ui.LiveTrackingScreen
import com.build2launch.balltracker.ui.VideoModeScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            BallTrackerTheme {
                RootScaffold()
            }
        }
    }
}

@Composable
private fun RootScaffold() {
    var tab by remember { mutableIntStateOf(0) }
    Scaffold(
        bottomBar = {
            NavigationBar {
                NavigationBarItem(
                    selected = tab == 0,
                    onClick = { tab = 0 },
                    icon = { Icon(Icons.Filled.CenterFocusWeak, contentDescription = "Live") },
                    label = { Text("Live") })
                NavigationBarItem(
                    selected = tab == 1,
                    onClick = { tab = 1 },
                    icon = { Icon(Icons.Filled.Movie, contentDescription = "Video") },
                    label = { Text("Video") })
            }
        }
    ) { padding ->
        val content = Modifier.fillMaxSize().padding(padding)
        when (tab) {
            0 -> androidx.compose.foundation.layout.Box(content) { LiveTrackingScreen() }
            else -> androidx.compose.foundation.layout.Box(content) { VideoModeScreen() }
        }
    }
}
