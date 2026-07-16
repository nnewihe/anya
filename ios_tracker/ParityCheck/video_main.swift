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
        // VITERBI_* env vars override solver weights, so the tuning sweep can
        // explore the space without rebuilding.
        let processor = VideoProcessor(modelURL: compiled,
                                       viterbiConfig: .fromEnvironment(env))

        // The detector threshold is the other half of the tuning space: it sets
        // how much evidence the solver has to work with, and trades recall
        // against ghosts. Sweepable for the same reason the weights are.
        let conf = env["VITERBI_SOLVER_CONF"].flatMap { Float($0) } ?? BallDetector.solverConf

        let t0 = CFAbsoluteTimeGetCurrent()
        let analysis = try await processor.process(
            url: URL(fileURLWithPath: videoPath), conf: conf
        ) { p in
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

        // When is the trace actually on screen? Contiguous live spans, so a
        // "nothing is drawn" report can be checked against playback position.
        var spans: [(Double, Double)] = []
        for s in analysis.samples {
            let live = s.state == .moving || s.state == .coasting
            if live {
                if var last = spans.last, last.1 >= s.t - 0.2 {
                    last.1 = s.t; spans[spans.count - 1] = last
                } else {
                    spans.append((s.t, s.t))
                }
            }
        }
        let shown = spans.filter { $0.1 - $0.0 >= 0.15 }
        print("live spans: \(shown.count) (>=0.15s)")
        for (a, b) in shown.prefix(25) {
            print(String(format: "   %6.2fs - %6.2fs  (%.2fs)", a, b, b - a))
        }
        if shown.count > 25 { print("   … \(shown.count - 25) more") }
        if let first = shown.first {
            print(String(format: "first trace appears at %.2fs", first.0))
        }

        if let out = env["DUMP_CSV"] {
            var s = "t,x,y,state,speed\n"
            for smp in analysis.samples {
                s += String(format: "%.4f,%.2f,%.2f,%@,%.1f\n", smp.t,
                            smp.pos?.x ?? -1, smp.pos?.y ?? -1,
                            smp.state.rawValue, smp.speedPxS)
            }
            try? s.write(toFile: out, atomically: true, encoding: .utf8)
            print("wrote \(out)")
        }

        exit(analysis.samples.isEmpty ? 1 : 0)
    }
}
