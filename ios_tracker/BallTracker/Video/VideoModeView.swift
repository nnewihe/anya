import AVFoundation
import AVKit
import PhotosUI
import SwiftUI

/// A picked video copied into a temp file we can read from.
struct PickedMovie: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension(received.file.pathExtension.isEmpty
                                        ? "mov" : received.file.pathExtension)
            try FileManager.default.copyItem(at: received.file, to: dest)
            return PickedMovie(url: dest)
        }
    }
}

struct VideoModeView: View {
    enum Phase {
        case idle
        case loading
        case processing(Double)
        case done(VideoAnalysis)
        case failed(String)
    }

    @State private var selection: PhotosPickerItem?
    @State private var phase: Phase = .idle
    @State private var task: Task<Void, Never>?

    var body: some View {
        VStack(spacing: 0) {
            switch phase {
            case .idle:
                pickerPrompt
            case .loading:
                ProgressView("Loading video…")
                    .frame(maxHeight: .infinity)
            case .processing(let p):
                VStack(spacing: 16) {
                    ProgressView(value: p) {
                        Text("Tracking ball… \(Int(p * 100))%")
                    }
                    .padding(.horizontal, 32)
                    Button("Cancel", role: .cancel) {
                        task?.cancel()
                        phase = .idle
                        selection = nil
                    }
                }
                .frame(maxHeight: .infinity)
            case .done(let analysis):
                AnalysisPlayerView(analysis: analysis)
                pickAnotherBar
            case .failed(let message):
                VStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.largeTitle)
                    Text(message).font(.callout)
                }
                .frame(maxHeight: .infinity)
                pickAnotherBar
            }
        }
        .onChange(of: selection) { _, item in
            guard let item else { return }
            phase = .loading
            task = Task { await load(item) }
        }
    }

    private var pickerPrompt: some View {
        VStack(spacing: 16) {
            Image(systemName: "video.badge.waveform")
                .font(.system(size: 44))
                .foregroundStyle(BallOverlay.ballYellow)
            Text("Pick a match video to track the ball offline.")
                .multilineTextAlignment(.center)
            PhotosPicker(selection: $selection, matching: .videos) {
                Label("Choose Video", systemImage: "photo.on.rectangle")
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(BallOverlay.ballYellow, in: Capsule())
                    .foregroundStyle(.black)
            }
        }
        .padding()
        .frame(maxHeight: .infinity)
    }

    private var pickAnotherBar: some View {
        PhotosPicker(selection: $selection, matching: .videos) {
            Label("Choose Another Video", systemImage: "photo.on.rectangle")
        }
        .padding(.vertical, 10)
    }

    private func load(_ item: PhotosPickerItem) async {
        do {
            guard let movie = try await item.loadTransferable(type: PickedMovie.self) else {
                phase = .failed("Could not load that video.")
                return
            }
            phase = .processing(0)
            let analysis = try await Task.detached(priority: .userInitiated) {
                try await VideoProcessor().process(url: movie.url) { p in
                    Task { @MainActor in
                        if case .processing = phase { phase = .processing(p) }
                    }
                }
            }.value
            phase = .done(analysis)
        } catch is CancellationError {
            // User cancelled; phase already reset.
        } catch {
            phase = .failed("Processing failed: \(error.localizedDescription)")
        }
    }
}

/// Plays the analyzed video with the tracked trajectory drawn on top,
/// synchronized to the player's clock (scrubbing included).
struct AnalysisPlayerView: View {
    let analysis: VideoAnalysis
    @State private var player: AVPlayer

    init(analysis: VideoAnalysis) {
        self.analysis = analysis
        _player = State(initialValue: AVPlayer(url: analysis.url))
    }

    var body: some View {
        VStack(spacing: 8) {
            statsBar
            VideoPlayer(player: player) {
                TimelineView(.animation) { _ in
                    PlaybackOverlay(analysis: analysis,
                                    t: player.currentTime().seconds)
                        .allowsHitTesting(false)
                }
            }
            .onDisappear { player.pause() }
        }
    }

    private var statsBar: some View {
        HStack(spacing: 16) {
            Label(String(format: "%.0f%% live trace", analysis.liveTraceRate * 100),
                  systemImage: "scope")
            Label(String(format: "%.0f px/s max", analysis.maxSpeedPxS),
                  systemImage: "speedometer")
            Label(String(format: "%.0f ms/frame", analysis.avgInferenceMs),
                  systemImage: "bolt")
        }
        .font(.caption.monospacedDigit())
        .padding(.vertical, 6)
    }
}

struct PlaybackOverlay: View {
    let analysis: VideoAnalysis
    let t: Double

    var body: some View {
        GeometryReader { geo in
            Canvas { ctx, size in
                let fit = AVMakeRect(aspectRatio: analysis.size,
                                     insideRect: CGRect(origin: .zero, size: size))
                let s = fit.width / analysis.size.width
                func map(_ p: CGPoint) -> CGPoint {
                    CGPoint(x: fit.minX + p.x * s, y: fit.minY + p.y * s)
                }

                let trail = analysis.trail(at: t)
                if trail.count >= 2 {
                    for i in 1..<trail.count {
                        let alpha = 0.15 + 0.85 * Double(i) / Double(trail.count - 1)
                        var seg = Path()
                        seg.move(to: map(trail[i - 1]))
                        seg.addLine(to: map(trail[i]))
                        ctx.stroke(seg, with: .color(BallOverlay.ballYellow.opacity(alpha)),
                                   style: StrokeStyle(lineWidth: 3, lineCap: .round))
                    }
                }
                if let sample = analysis.sample(at: t), let pos = sample.pos {
                    let p = map(pos)
                    let ring = Path(ellipseIn: CGRect(x: p.x - 10, y: p.y - 10,
                                                      width: 20, height: 20))
                    ctx.stroke(ring, with: .color(BallOverlay.ballYellow), lineWidth: 2.5)
                }
            }
        }
    }
}
