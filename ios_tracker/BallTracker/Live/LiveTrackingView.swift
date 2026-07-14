import AVFoundation
import SwiftUI

@MainActor
final class LiveTrackerViewModel: ObservableObject {
    @Published var result: FrameResult?
    @Published var processedFps: Double = 0
    @Published var errorMessage: String?
    @Published var running = false

    let camera = CameraManager()
    private var engine: TrackerEngine?
    private var frameTimes: [Double] = []

    func start() async {
        guard await CameraManager.requestPermission() else {
            errorMessage = "Camera access is required. Enable it in Settings."
            return
        }
        do {
            let fps = try camera.configure()
            let engine = try TrackerEngine(fps: fps)
            self.engine = engine
            camera.onFrame = { [weak self] pixelBuffer, t in
                guard let self, let r = try? engine.process(pixelBuffer, at: t) else { return }
                Task { @MainActor in self.ingest(r) }
            }
            camera.start()
            running = true
        } catch {
            errorMessage = "Camera setup failed: \(error.localizedDescription)"
        }
    }

    func stop() {
        camera.stop()
        running = false
    }

    private func ingest(_ r: FrameResult) {
        result = r
        let now = CFAbsoluteTimeGetCurrent()
        frameTimes.append(now)
        frameTimes.removeAll { now - $0 > 1.0 }
        processedFps = Double(frameTimes.count)
    }
}

struct LiveTrackingView: View {
    @StateObject private var model = LiveTrackerViewModel()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            CameraPreview(manager: model.camera)
                .ignoresSafeArea()
            if let result = model.result {
                BallOverlay(result: result)
                    .ignoresSafeArea()
                    .allowsHitTesting(false)
            }
            VStack {
                if let error = model.errorMessage {
                    Text(error)
                        .font(.callout)
                        .foregroundStyle(.white)
                        .padding(12)
                        .background(.red.opacity(0.8), in: RoundedRectangle(cornerRadius: 10))
                        .padding(.top, 8)
                }
                Spacer()
                if let result = model.result {
                    LiveHUD(result: result, processedFps: model.processedFps)
                        .padding(.bottom, 12)
                }
            }
        }
        .task { await model.start() }
        .onDisappear { model.stop() }
    }
}

/// UIKit host for the AVCaptureVideoPreviewLayer.
struct CameraPreview: UIViewRepresentable {
    let manager: CameraManager

    final class PreviewHostView: UIView {
        var previewLayer: AVCaptureVideoPreviewLayer? {
            didSet {
                oldValue?.removeFromSuperlayer()
                if let previewLayer {
                    layer.addSublayer(previewLayer)
                    setNeedsLayout()
                }
            }
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            previewLayer?.frame = bounds
        }
    }

    func makeUIView(context: Context) -> PreviewHostView {
        let view = PreviewHostView()
        view.backgroundColor = .black
        view.previewLayer = manager.previewLayer
        return view
    }

    func updateUIView(_ uiView: PreviewHostView, context: Context) {}
}

/// Draws the smoothed trajectory, the current ball marker, and the raw
/// detection boxes over the aspect-fit video rect.
struct BallOverlay: View {
    let result: FrameResult

    static let ballYellow = Color(red: 0.87, green: 1.0, blue: 0.16)

    var body: some View {
        GeometryReader { geo in
            Canvas { ctx, size in
                let fit = AVMakeRect(aspectRatio: result.frameSize,
                                     insideRect: CGRect(origin: .zero, size: size))
                let s = fit.width / result.frameSize.width
                func map(_ p: CGPoint) -> CGPoint {
                    CGPoint(x: fit.minX + p.x * s, y: fit.minY + p.y * s)
                }

                // Trajectory trail, fading toward the oldest point.
                let trace = result.trace
                if trace.count >= 2 {
                    for i in 1..<trace.count {
                        let alpha = 0.15 + 0.85 * Double(i) / Double(trace.count - 1)
                        var seg = Path()
                        seg.move(to: map(trace[i - 1]))
                        seg.addLine(to: map(trace[i]))
                        ctx.stroke(seg, with: .color(Self.ballYellow.opacity(alpha)),
                                   style: StrokeStyle(lineWidth: 3, lineCap: .round))
                    }
                }

                // Raw detections (thin white boxes).
                for det in result.detections {
                    let r = CGRect(x: fit.minX + det.box.minX * s,
                                   y: fit.minY + det.box.minY * s,
                                   width: det.box.width * s,
                                   height: det.box.height * s)
                    ctx.stroke(Path(roundedRect: r, cornerRadius: 2),
                               with: .color(.white.opacity(0.6)), lineWidth: 1)
                }

                // Current tracked position.
                if let pos = result.ballPosition, result.status.state != .none {
                    let p = map(pos)
                    let ring = Path(ellipseIn: CGRect(x: p.x - 10, y: p.y - 10,
                                                      width: 20, height: 20))
                    ctx.stroke(ring, with: .color(Self.ballYellow), lineWidth: 2.5)
                }
            }
        }
    }
}

struct LiveHUD: View {
    let result: FrameResult
    let processedFps: Double

    private var stateColor: Color {
        switch result.status.state {
        case .moving: return .green
        case .coasting: return .yellow
        case .fading, .stopped: return .orange
        case .none, .lost: return .gray
        }
    }

    var body: some View {
        HStack(spacing: 16) {
            HStack(spacing: 6) {
                Circle().fill(stateColor).frame(width: 10, height: 10)
                Text(result.status.state.rawValue.capitalized)
            }
            Text("\(Int(result.speedPxS)) px/s")
            Text("\(String(format: "%.0f", result.inferenceMs)) ms")
            Text("\(String(format: "%.0f", processedFps)) fps")
        }
        .font(.system(.footnote, design: .monospaced))
        .foregroundStyle(.white)
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.black.opacity(0.6), in: Capsule())
    }
}
