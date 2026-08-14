// macOS end-to-end check of the video path: real mp4 -> AVAssetReader ->
// TrackerEngine (Letterbox + CoreML on the ANE + decode/NMS + Kalman/IMM) +
// YOLO player boxes + carry-suppression -> rally detector (rally_detector.py port).
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
        // MODEL overrides the default package, so a sweep can compare exports
        // (e.g. input resolutions) without touching the bundled model.
        let mlpackage = env["MODEL"].map { URL(fileURLWithPath: $0) }
            ?? repo.appendingPathComponent("spikes/models/ball_best.mlpackage")

        // Player model for carry-suppression; PLAYER_MODEL overrides the default.
        let playerPkg = env["PLAYER_MODEL"].map { URL(fileURLWithPath: $0) }
            ?? repo.appendingPathComponent("spikes/models/yolo26n.mlpackage")

        // ROI_MODEL (path to the small 480x288 mlpackage, or "1" for the
        // default export) switches detection to tracked-ROI + tiered scan.
        let roiPkg: URL? = env["ROI_MODEL"].flatMap {
            if $0.isEmpty { return nil }
            return $0 == "1"
                ? repo.appendingPathComponent("spikes/models/ball_best_roi.mlpackage")
                : URL(fileURLWithPath: $0)
        }

        let compiled = try await MLModel.compileModel(at: mlpackage)
        let compiledPlayer = try await MLModel.compileModel(at: playerPkg)
        var compiledRoi: URL?
        if let roiPkg { compiledRoi = try await MLModel.compileModel(at: roiPkg) }
        let processor = VideoProcessor(modelURL: compiled, playerModelURL: compiledPlayer,
                                       roiModelURL: compiledRoi)

        // The detector threshold trades recall against ghosts; sweepable so the
        // harness can explore it without rebuilding.
        let conf = env["BALL_CONF"].flatMap { Float($0) } ?? BallDetector.defaultConf

        let t0 = CFAbsoluteTimeGetCurrent()
        let analysis = try await processor.process(
            url: URL(fileURLWithPath: videoPath), conf: conf
        ) { p in
            print(String(format: "\rprogress: %3.0f%%", p * 100), terminator: "")
        }
        let wall = CFAbsoluteTimeGetCurrent() - t0

        print()
        let probe = try BallDetector(modelURL: compiled)
        print("model     : \(mlpackage.lastPathComponent) "
              + "input=\(probe.inputWidth)x\(probe.inputHeight) "
              + String(format: "maxBoxPx=%.0f conf=%.3f", probe.maxBoxPx, conf))
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

        // The rally detector's verdict — the segments the app would export.
        let segs = analysis.rallySegments
        let reelSec = segs.reduce(0.0) { $0 + ($1.end - $1.start) }
        print(String(format: "rallies   : %d segment(s), %.0fs reel", segs.count, reelSec))
        for seg in segs.prefix(50) {
            print(String(format: "   %7.2fs - %7.2fs  (%5.2fs)  %@",
                         seg.start, seg.end, seg.end - seg.start, seg.origin.rawValue))
        }
        if segs.count > 50 { print("   … \(segs.count - 50) more") }

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
