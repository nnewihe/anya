// macOS end-to-end check of the video path: real mp4 -> AVAssetReader ->
// TrackerEngine (Letterbox + CoreML on the ANE + decode/NMS + Kalman/IMM).
// Exercises the exact VideoProcessor the iOS app ships.
//
// Run:  ios_tracker/run_video_check.sh [/path/to/video.mp4]

import CoreML
import Foundation

@main
struct VideoCheck {
    static func main() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let repoPath = env["ANYA_REPO"], let videoPath = env["VIDEO"] else {
            print("FATAL: set ANYA_REPO and VIDEO")
            exit(1)
        }
        let repo = URL(fileURLWithPath: repoPath)
        let mlpackage = repo.appendingPathComponent("spikes/models/ball_best.mlpackage")

        let compiled = try await MLModel.compileModel(at: mlpackage)
        let processor = VideoProcessor(modelURL: compiled)

        let t0 = CFAbsoluteTimeGetCurrent()
        let analysis = try await processor.process(url: URL(fileURLWithPath: videoPath)) { p in
            print(String(format: "\rprogress: %3.0f%%", p * 100), terminator: "")
        }
        let wall = CFAbsoluteTimeGetCurrent() - t0

        print()
        print(String(format: "video     : %.1fs @ %.0f fps, %dx%d",
                     analysis.duration, analysis.fps,
                     Int(analysis.size.width), Int(analysis.size.height)))
        print("frames    : \(analysis.samples.count)")
        print(String(format: "wall time : %.1fs  (%.2fx realtime, %.1f ms/frame incl. decode)",
                     wall, analysis.duration / wall,
                     wall * 1000 / Double(max(analysis.samples.count, 1))))
        print(String(format: "inference : %.1f ms/frame avg", analysis.avgInferenceMs))
        print(String(format: "live trace: %.0f%% of frames", analysis.liveTraceRate * 100))
        print(String(format: "max speed : %.0f px/s", analysis.maxSpeedPxS))

        let states = Dictionary(grouping: analysis.samples, by: \.state.rawValue)
            .mapValues(\.count)
            .sorted { $0.value > $1.value }
        print("states    : \(states.map { "\($0.key)=\($0.value)" }.joined(separator: " "))")

        exit(analysis.samples.isEmpty ? 1 : 0)
    }
}
