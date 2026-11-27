// RecordView.swift
//
// While recording: the camera picture with the tapped court on it, and what
// live analysis is doing -- how far behind the camera it is, whether it sees
// the players, and anything that needs attention (heat, storage).
import SwiftUI

struct RecordView: View {
    @EnvironmentObject private var model: AppModel
    @State private var preview: PreviewView?
    @State private var confirmStop = false

    var body: some View {
        ZStack {
            CameraPreview(session: model.camera.session) { preview = $0 }.ignoresSafeArea()
            CourtOutline(corners: model.current?.manifest.courtCorners ?? [], preview: preview)
            HStack {
                VStack(alignment: .leading) {
                    TimelineView(.periodic(from: .now, by: 1)) { _ in statusBar }
                    Spacer()
                }
                Spacer()
                stopButton
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
        return HStack(spacing: 14) {
            HStack(spacing: 8) {
                PulsingDot(color: .white)
                Text(Self.clock(recorded)).font(Theme.display(20, .semibold)).monospacedDigit()
            }
            .padding(.horizontal, 14).padding(.vertical, 9)
            .background(Theme.record.opacity(0.85), in: Capsule())
            HStack(spacing: 14) {
                Label("analysis \(Int(behind)) s behind", systemImage: "waveform.path.ecg").monospacedDigit()
                Label("near \(status.nearPlayers) · far \(status.farPlayers)", systemImage: "figure.tennis")
                if status.pausedForHeat {
                    Label("phone hot: analysis paused", systemImage: "thermometer.sun").foregroundStyle(Theme.warning)
                }
                if freeGB < 15 { Label("\(Int(freeGB)) GB left", systemImage: "externaldrive").foregroundStyle(Theme.warning) }
            }
            .font(.callout.weight(.medium))
            .padding(.horizontal, 14).padding(.vertical, 10)
            .glass(Capsule())
        }
    }

    /// The camera app's shutter, as Stop.
    private var stopButton: some View {
        Button { confirmStop = true } label: {
            ZStack {
                Circle().stroke(Color.white, lineWidth: 4).frame(width: 76, height: 76)
                RoundedRectangle(cornerRadius: 8, style: .continuous).fill(Theme.record).frame(width: 32, height: 32)
            }
            .frame(width: 84, height: 84)
            .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Stop")
    }

    /// "1 point", "3 points".
    static func points(_ count: Int) -> String { count == 1 ? "1 point" : "\(count) points" }

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
            context.stroke(path, with: .color(Theme.court.opacity(0.6)), lineWidth: 1.5)
        }
        .allowsHitTesting(false)
        .ignoresSafeArea()
    }
}

/// After a phone call or leaving the app: the part so far is safe.
struct PausedView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "pause.fill")
                .font(.system(size: 36, weight: .bold)).foregroundStyle(Theme.warning)
                .frame(width: 88, height: 88).background(Theme.warning.opacity(0.16), in: Circle())
            Text("Recording paused").font(Theme.display(28, .heavy))
            Text("Everything up to the interruption is saved. Check the camera still points at the court, then carry on.")
                .multilineTextAlignment(.center).foregroundStyle(.secondary).frame(maxWidth: 500)
            HStack(spacing: 16) {
                Button {
                    if let manifest = model.current?.manifest { Task { await model.resume(manifest) } }
                } label: { Label("Resume recording", systemImage: "record.circle") }
                .buttonStyle(.primary(Theme.record, large: true))
                Button("Finish the match") {
                    if let manifest = model.current?.manifest { Task { await model.finish(manifest) } }
                }
                .buttonStyle(.secondary)
            }
        }
        .card(padding: 32)
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.background.ignoresSafeArea())
    }
}
