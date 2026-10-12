// RecordView.swift
//
// While recording: the camera picture with the tapped court on it, and what
// live analysis is doing -- how far behind the camera it is, whether it sees
// the players, and anything that needs attention (heat, storage).
import SwiftUI

struct RecordView: View {
    @Environment(AppModel.self) private var model
    @State private var preview: PreviewView?
    @State private var confirmStop = false

    var body: some View {
        ZStack {
            CameraPreview(session: model.camera.session) { preview = $0 }.ignoresSafeArea()
            CourtOutline(corners: model.current?.manifest.courtCorners ?? [], preview: preview)
            VStack {
                TimelineView(.periodic(from: .now, by: 1)) { _ in statusBar }
                Spacer()
                HStack {
                    Spacer()
                    Button(role: .destructive) { confirmStop = true } label: {
                        Label("Stop", systemImage: "stop.circle.fill").font(.title2.bold()).padding(.horizontal)
                    }
                    .buttonStyle(.borderedProminent).tint(.red)
                }
            }
            .padding()
        }
        .confirmationDialog("Stop recording and make the reel?", isPresented: $confirmStop, titleVisibility: .visible) {
            Button("Stop and make the reel", role: .destructive) { Task { await model.stop() } }
            Button("Keep recording", role: .cancel) {}
        }
    }

    private var statusBar: some View {
        let controller = model.current
        let status = controller?.analysisStatus ?? .init()
        let recorded = controller?.secondsRecorded ?? 0
        let behind = max(0, recorded - status.secondsAnalysed)
        let freeGB = Double(MatchStore.freeBytes()) / 1e9
        return HStack(spacing: 16) {
            Label(Self.clock(recorded), systemImage: "record.circle").foregroundStyle(.red).monospacedDigit()
            Label("analysis \(Int(behind)) s behind", systemImage: "waveform.path.ecg").monospacedDigit()
            Label("near \(status.nearPlayers) · far \(status.farPlayers)", systemImage: "figure.tennis")
            if status.pausedForHeat { Label("phone hot: analysis paused", systemImage: "thermometer.sun").foregroundStyle(.orange) }
            if freeGB < 15 { Label("\(Int(freeGB)) GB left", systemImage: "externaldrive").foregroundStyle(.orange) }
        }
        .font(.callout)
        .padding(10)
        .background(.black.opacity(0.6), in: Capsule())
    }

    static func clock(_ seconds: Double) -> String {
        let s = Int(seconds)
        return String(format: "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60)
    }
}

/// The tapped court drawn over the preview, so a moved camera is obvious.
struct CourtOutline: View {
    let corners: [MatchManifest.Point]
    let preview: PreviewView?

    var body: some View {
        Canvas { context, _ in
            guard let preview, corners.count == 4 else { return }
            var path = Path()
            path.addLines(corners.map { preview.screenPoint(from: CGPoint(x: $0.x, y: $0.y)) })
            path.closeSubpath()
            context.stroke(path, with: .color(.yellow.opacity(0.6)), lineWidth: 1.5)
        }
        .allowsHitTesting(false)
        .ignoresSafeArea()
    }
}

/// After a phone call or leaving the app: the part so far is safe.
struct PausedView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "pause.circle").font(.system(size: 60))
            Text("Recording paused").font(.title.bold())
            Text("Everything up to the interruption is saved. Check the camera still points at the court, then carry on.")
                .multilineTextAlignment(.center).foregroundStyle(.secondary).frame(maxWidth: 500)
            HStack(spacing: 16) {
                Button("Resume recording") {
                    if let manifest = model.current?.manifest { Task { await model.resume(manifest) } }
                }
                .buttonStyle(.borderedProminent)
                Button("Finish the match") {
                    if let manifest = model.current?.manifest { Task { await model.finish(manifest) } }
                }
                .buttonStyle(.bordered)
            }
        }
        .padding()
    }
}
