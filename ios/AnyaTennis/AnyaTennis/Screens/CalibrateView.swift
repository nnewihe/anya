// CalibrateView.swift
//
// Before recording: aim the camera, then tap the four corners of the SINGLES
// court in order -- near-left, near-right, far-right, far-left. Drag a corner
// to fine-tune it. The tapped corners define the court for the whole match
// (court metres = homography of these four points).
import Anya2Engine
import CoreMedia
import CoreVideo
import SwiftUI

struct CalibrateView: View {
    @Environment(AppModel.self) private var model
    @State private var corners: [CGPoint] = []          // analysis px
    @State private var preview: PreviewView?
    @State private var latestFrame = LatestFrame()
    @State private var dragging: Int?
    @State private var working = false

    private static let names = ["near-left", "near-right", "far-right", "far-left"]

    var body: some View {
        ZStack {
            CameraPreview(session: model.camera.session) { preview = $0 }
                .ignoresSafeArea()
                .gesture(DragGesture(minimumDistance: 0)
                    .onChanged { value in drag(value) }
                    .onEnded { value in endDrag(value) })
            courtOverlay
            VStack {
                HStack {
                    Button("Cancel") { model.home() }.buttonStyle(.bordered)
                    Spacer()
                    Text(instruction).font(.headline).padding(10)
                        .background(.black.opacity(0.6), in: Capsule())
                    Spacer()
                    Button("Clear") { corners.removeAll() }.buttonStyle(.bordered).disabled(corners.isEmpty)
                }
                Spacer()
                Button {
                    Task { await confirm() }
                } label: {
                    Label("Start recording", systemImage: "record.circle").font(.title3.bold()).padding(.horizontal)
                }
                .buttonStyle(.borderedProminent).tint(.red)
                .disabled(corners.count < 4 || working)
            }
            .padding()
        }
        .onAppear { model.camera.setHandlers(video: { [latestFrame] in latestFrame.keep($0) }, audio: nil) }
        .onDisappear { if model.current?.isRecording != true { model.camera.setHandlers(video: nil, audio: nil) } }
    }

    private var instruction: String {
        corners.count < 4 ? "Tap the \(Self.names[corners.count]) corner of the singles court"
                          : "Drag a corner to adjust, then start recording"
    }

    /// The tapped quad, its corner handles, and where the next tap goes.
    private var courtOverlay: some View {
        Canvas { context, _ in
            guard let preview, !corners.isEmpty else { return }
            let points = corners.map { preview.screenPoint(from: $0) }
            var path = Path()
            path.addLines(points)
            if points.count == 4 { path.closeSubpath() }
            context.stroke(path, with: .color(.yellow), lineWidth: 2)
            for (i, point) in points.enumerated() {
                let handle = CGRect(x: point.x - 9, y: point.y - 9, width: 18, height: 18)
                context.stroke(Path(ellipseIn: handle), with: .color(.yellow), lineWidth: 2)
                context.draw(Text(Self.names[i]).font(.caption2).foregroundColor(.yellow),
                             at: CGPoint(x: point.x, y: point.y - 18))
            }
        }
        .allowsHitTesting(false)
        .ignoresSafeArea()
    }

    private func drag(_ value: DragGesture.Value) {
        guard let preview else { return }
        let point = preview.analysisPoint(from: value.location)
        if dragging == nil {
            // Start dragging the nearest existing corner if the touch is on it.
            let screen = corners.map { preview.screenPoint(from: $0) }
            dragging = screen.firstIndex { hypot($0.x - value.startLocation.x, $0.y - value.startLocation.y) < 30 } ?? -1
        }
        if let index = dragging, index >= 0 { corners[index] = clamp(point) }
    }

    private func endDrag(_ value: DragGesture.Value) {
        defer { dragging = nil }
        guard let preview, dragging == -1, corners.count < 4 else { return }
        corners.append(clamp(preview.analysisPoint(from: value.location)))
    }

    private func clamp(_ p: CGPoint) -> CGPoint { CGPoint(x: min(max(p.x, 0), 959), y: min(max(p.y, 0), 539)) }

    private func confirm() async {
        working = true
        defer { working = false }
        guard let frame = latestFrame.take(), let near = FrameScaler(band: .init(x: 0, y: 0, width: 2, height: 2)).near(frame)
        else { model.error = "No picture from the camera yet."; return }
        do {
            await model.calibrated(try MatchController.create(corners: corners, calibrationFrame: near, camera: model.camera))
        } catch {
            model.error = error.localizedDescription
        }
    }
}

/// The most recent camera frame (one only, so the capture pool never starves).
final class LatestFrame: @unchecked Sendable {
    private let lock = NSLock()
    private var frame: CVPixelBuffer?
    func keep(_ sample: CMSampleBuffer) {
        lock.withLock { frame = CMSampleBufferGetImageBuffer(sample) }
    }
    func take() -> CVPixelBuffer? { lock.withLock { frame } }
}
