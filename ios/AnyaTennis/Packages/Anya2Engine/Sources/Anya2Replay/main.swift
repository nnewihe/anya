// anya2-replay: runs the Swift pipeline (stages 5-8) over clips exported by
// tools/export_replay.py and prints, per clip, what Swift found next to what
// main's Python found, plus how long Swift took.
//
//     swift run -c release anya2-replay EXPORT_DIR [CLIP ...]
import Anya2Engine
import Foundation

let arguments = CommandLine.arguments.dropFirst()
guard let root = arguments.first else {
    print("usage: anya2-replay EXPORT_DIR [CLIP ...]")
    exit(2)
}
let only = Set(arguments.dropFirst())
let model = try WalkingModel.bundled()
let clips = try FileManager.default.contentsOfDirectory(atPath: root)
    .filter { only.isEmpty || only.contains($0) }
    .filter { FileManager.default.fileExists(atPath: "\(root)/\($0)/meta.json") }
    .sorted()

print("clip  minutes | near serves  far serves  point ends  segments | swift / python      | seconds")
for name in clips {
    let clip = try ReplayClip(directory: URL(fileURLWithPath: "\(root)/\(name)"))
    let started = Date()
    let result = runBatch(near: clip.poseDetections("near"), far: clip.poseDetections("far"),
                          geometry: clip.geometry, walkingModel: model)
    let seconds = Date().timeIntervalSince(started)
    let reference = try clip.referenceJSON()
    func count(_ key: String) -> Int { (reference[key] as! [Any]).count }
    let pythonSegments = ((reference["reel"] as! [String: Any])["segments"] as! [Any]).count
    let swift = "\(result.nearServes.count)/\(result.farServes.count)/\(result.pointEnds.count)/\(result.reel.segments.count)"
    let python = "\(count("near_serve"))/\(count("far_serve"))/\(count("point_end"))/\(pythonSegments)"
    print(String(format: "%4@  %7.1f | %11d  %10d  %10d  %8d | %@ %@ | %6.1f",
                 name, result.tracks.durationSeconds / 60, result.nearServes.count, result.farServes.count,
                 result.pointEnds.count, result.reel.segments.count,
                 swift == python ? "same          " : "DIFFERS       ", python, seconds))
}
