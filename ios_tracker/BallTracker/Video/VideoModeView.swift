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
        case resumable(ResumeInfo)
        case loading
        case processing(Double)
        case done(VideoAnalysis)
        case failed(String)
    }

    @State private var selection: PhotosPickerItem?
    @State private var phase: Phase = .idle
    @State private var task: Task<Void, Never>?
    private let store = CheckpointStore()

    var body: some View {
        VStack(spacing: 0) {
            switch phase {
            case .idle:
                pickerPrompt
            case .resumable(let info):
                resumePrompt(info)
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
        .task {
            // Reclaim old storage, then offer the last interrupted analysis (if
            // its video and detector inputs still match) once, on entry.
            store.prune()
            if case .idle = phase,
               let info = store.resumable(
                    detectorFingerprint: VideoProcessor.detectorFingerprint(
                        conf: BallDetector.defaultConf,
                        roi: BallDetector.bundledRoiModelURL != nil)) {
                phase = .resumable(info)
            }
        }
    }

    private func resumePrompt(_ info: ResumeInfo) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "arrow.clockwise.circle")
                .font(.system(size: 44))
                .foregroundStyle(Theme.ballYellow)
            Text("Resume tracking “\(info.displayName)”?")
                .multilineTextAlignment(.center)
            Text("\(Int(info.progress * 100))% processed")
                .font(.callout.monospacedDigit())
                .foregroundStyle(.secondary)
            HStack(spacing: 12) {
                Button {
                    resume(info)
                } label: {
                    Label("Resume", systemImage: "play.fill")
                        .padding(.horizontal, 20).padding(.vertical, 10)
                        .background(Theme.ballYellow, in: Capsule())
                        .foregroundStyle(.black)
                }
                Button(role: .destructive) {
                    store.discard(key: info.key)
                    phase = .idle
                } label: {
                    Label("Discard", systemImage: "trash")
                        .padding(.horizontal, 20).padding(.vertical, 10)
                }
            }
        }
        .padding()
        .frame(maxHeight: .infinity)
    }

    private var pickerPrompt: some View {
        VStack(spacing: 16) {
            Image(systemName: "video.badge.waveform")
                .font(.system(size: 44))
                .foregroundStyle(Theme.ballYellow)
            Text("Pick a match video to track the ball offline.")
                .multilineTextAlignment(.center)
            PhotosPicker(selection: $selection, matching: .videos) {
                Label("Choose Video", systemImage: "photo.on.rectangle")
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(Theme.ballYellow, in: Capsule())
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
            // Give the picked video a stable identity and a persistent home, so
            // an interrupted analysis can be resumed after the app is killed
            // (the temp import would otherwise be purged / renamed).
            let key = try store.fingerprint(of: movie.url)
            let url = try store.adoptWorkingCopy(
                tempURL: movie.url, key: key, ext: movie.url.pathExtension)
            // PhotosPicker gives no human-readable title; label it by a short
            // hash prefix so distinct pending videos are distinguishable.
            let name = "Video \(key.prefix(6))"
            await run(url: url, key: key, name: name)
        } catch is CancellationError {
            // User cancelled; phase already reset.
        } catch {
            phase = .failed("Processing failed: \(error.localizedDescription)")
        }
    }

    private func resume(_ info: ResumeInfo) {
        task = Task { await run(url: info.workingCopyURL, key: info.key,
                                name: info.displayName) }
    }

    /// Run (or resume) the offline analysis for a persistent working copy,
    /// checkpointing Pass 1 under `key` so a kill mid-run can be resumed.
    private func run(url: URL, key: String, name: String) async {
        phase = .processing(0)
        do {
            let store = store
            let analysis = try await Task.detached(priority: .userInitiated) {
                try await VideoProcessor(roiModelURL: BallDetector.bundledRoiModelURL).process(
                    url: url, checkpoint: (store, key, name)) { p in
                    Task { @MainActor in
                        if case .processing = phase { phase = .processing(p) }
                    }
                }
            }.value
            phase = .done(analysis)
        } catch is CancellationError {
            // User cancelled; progress is checkpointed and offered again later.
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

    private enum Export: Equatable {
        case idle, running
        case done(URL)
        case failed(String)
    }
    @State private var export: Export = .idle
    @State private var shareItem: ShareItem?

    init(analysis: VideoAnalysis) {
        self.analysis = analysis
        _player = State(initialValue: AVPlayer(url: analysis.url))
    }

    /// Precompute the keep-segments so the button can show the reel length and
    /// disable itself when there's nothing worth exporting.
    private var segments: [HighlightSegment] {
        HighlightsExporter.segments(for: analysis)
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
            exportBar
        }
        .sheet(item: $shareItem) { item in ShareSheet(items: [item.url]) }
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

    @ViewBuilder
    private var exportBar: some View {
        let segs = segments
        let reelSec = segs.reduce(0) { $0 + $1.duration }
        switch export {
        case .running:
            HStack(spacing: 8) {
                ProgressView()
                Text("Exporting highlights…")
            }
            .font(.footnote)
            .padding(.bottom, 8)
        case .failed(let message):
            Text(message)
                .font(.footnote)
                .foregroundStyle(.red)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 16)
                .padding(.bottom, 8)
        default:
            Button {
                runExport()
            } label: {
                Label(segs.isEmpty
                        ? "No rallies to export"
                        : String(format: "Export Highlights · %d clips · %.0fs",
                                 segs.count, reelSec),
                      systemImage: "film.stack")
                    .font(.callout.weight(.medium))
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(segs.isEmpty ? Color.gray.opacity(0.3) : Theme.ballYellow,
                                in: Capsule())
                    .foregroundStyle(segs.isEmpty ? Color.secondary : Color.black)
            }
            .disabled(segs.isEmpty)
            .padding(.bottom, 8)
        }
    }

    private func runExport() {
        export = .running
        Task {
            do {
                let url = try await HighlightsExporter.export(analysis: analysis)
                export = .done(url)
                shareItem = ShareItem(url: url)
            } catch {
                export = .failed(error.localizedDescription)
            }
        }
    }
}

/// Wrapper so the reel URL can drive a `.sheet(item:)` without retroactively
/// conforming URL to Identifiable (which can clash with SDK conformances).
private struct ShareItem: Identifiable {
    let url: URL
    var id: String { url.absoluteString }
}

/// UIKit share sheet, so the reel can be saved to Photos or shared.
struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
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

                // Fenced stationary-clutter zones (ball baskets, resting balls).
                for zone in analysis.exclusionZones {
                    let o = map(CGPoint(x: zone.minX, y: zone.minY))
                    let rect = CGRect(x: o.x, y: o.y,
                                      width: zone.width * s, height: zone.height * s)
                    ctx.stroke(Path(rect), with: .color(.red.opacity(0.9)),
                               style: StrokeStyle(lineWidth: 1.5))
                }

                let trail = analysis.trail(at: t)
                if trail.count >= 2 {
                    for i in 1..<trail.count {
                        let alpha = 0.15 + 0.85 * Double(i) / Double(trail.count - 1)
                        var seg = Path()
                        seg.move(to: map(trail[i - 1]))
                        seg.addLine(to: map(trail[i]))
                        ctx.stroke(seg, with: .color(Theme.ballYellow.opacity(alpha)),
                                   style: StrokeStyle(lineWidth: 3, lineCap: .round))
                    }
                }
                if let sample = analysis.sample(at: t), let pos = sample.pos {
                    let p = map(pos)
                    let ring = Path(ellipseIn: CGRect(x: p.x - 10, y: p.y - 10,
                                                      width: 20, height: 20))
                    ctx.stroke(ring, with: .color(Theme.ballYellow), lineWidth: 2.5)
                }
            }
        }
    }
}
