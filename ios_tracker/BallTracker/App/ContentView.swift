import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            LiveTrackingView()
                .tabItem {
                    Label("Live", systemImage: "dot.viewfinder")
                }
            VideoModeView()
                .tabItem {
                    Label("Video", systemImage: "film")
                }
        }
        .tint(BallOverlay.ballYellow)
    }
}
