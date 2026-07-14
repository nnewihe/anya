// macOS parity harness for the Swift port — runs the REAL app code
// (Letterbox, BallDetector, Kalman/IMM, BallTrackManager) against the Python
// golden fixtures. Not part of the iOS app target.
//
// Checks:
//  1. Detection parity: fixture frame -> Swift letterbox -> Core ML (ANE) ->
//     Swift decode+NMS, compared to the Ultralytics .pt golden boxes.
//  2. Tracker parity: the 10 self-test scenarios from
//     pipeline/ball_tracker.py / mobile/test/ball_tracker_test.dart.
//
// Run:  ios_tracker/run_parity_check.sh

import CoreML
import CoreVideo
import Foundation
import ImageIO

// MARK: - Helpers

func fail(_ msg: String) -> Never {
    print("FATAL: \(msg)")
    exit(1)
}

var failures = 0
func check(_ name: String, _ ok: Bool, _ detail: String = "") {
    print("\(ok ? "PASS" : "FAIL")  \(name)\(detail.isEmpty ? "" : "  — \(detail)")")
    if !ok { failures += 1 }
}

func loadPixelBuffer(png url: URL) -> CVPixelBuffer {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        fail("cannot load \(url.path)")
    }
    let w = img.width, h = img.height
    var pb: CVPixelBuffer?
    let attrs: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]]
    guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA,
                              attrs as CFDictionary, &pb) == kCVReturnSuccess,
          let pb else { fail("CVPixelBufferCreate failed") }
    CVPixelBufferLockBaseAddress(pb, [])
    defer { CVPixelBufferUnlockBaseAddress(pb, []) }
    guard let ctx = CGContext(
        data: CVPixelBufferGetBaseAddress(pb), width: w, height: h,
        bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue |
            CGBitmapInfo.byteOrder32Little.rawValue) else {
        fail("CGContext failed")
    }
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    return pb
}

func iou(_ a: CGRect, _ b: CGRect) -> Double {
    let inter = a.intersection(b)
    if inter.isNull || inter.isEmpty { return 0 }
    let ia = inter.width * inter.height
    let ua = a.width * a.height + b.width * b.height - ia
    return ua > 0 ? Double(ia / ua) : 0
}

// MARK: - Locate repo fixtures

let env = ProcessInfo.processInfo.environment
guard let repoPath = env["ANYA_REPO"] else {
    fail("set ANYA_REPO to the repo root")
}
let repo = URL(fileURLWithPath: repoPath)
let fixtures = repo.appendingPathComponent("spikes/fixtures")
let mlpackage = repo.appendingPathComponent("spikes/models/ball_best.mlpackage")

// MARK: - 1. Detection parity

print("=== Detection parity (Swift letterbox + CoreML + decode/NMS vs .pt golden) ===")

let compiled = try MLModel.compileModel(at: mlpackage)
let detector = try BallDetector(modelURL: compiled, computeUnits: .all)

let frame = loadPixelBuffer(png: fixtures.appendingPathComponent("frame_960x540.png"))

struct GoldenBoxes: Decodable {
    struct Box: Decodable {
        let conf: Double
        let xyxy: [Double]
    }
    let boxes_pt: [Box]
}
let golden = try JSONDecoder().decode(
    GoldenBoxes.self,
    from: Data(contentsOf: fixtures.appendingPathComponent("coreml_ball_boxes.json")))

let dets = try detector.detect(in: frame, conf: 0.05)
print("swift  : \(dets.map { (String(format: "%.3f", $0.conf), $0.box) })")
print("golden : \(golden.boxes_pt.map { ($0.conf, $0.xyxy) })")

check("box count matches golden", dets.count == golden.boxes_pt.count,
      "swift=\(dets.count) golden=\(golden.boxes_pt.count)")
var worstIoU = 1.0
var worstConf = 0.0
for (det, gold) in zip(dets.sorted { $0.conf > $1.conf },
                       golden.boxes_pt.sorted { $0.conf > $1.conf }) {
    let g = CGRect(x: gold.xyxy[0], y: gold.xyxy[1],
                   width: gold.xyxy[2] - gold.xyxy[0],
                   height: gold.xyxy[3] - gold.xyxy[1])
    worstIoU = min(worstIoU, iou(det.box, g))
    worstConf = max(worstConf, abs(Double(det.conf) - gold.conf))
}
check("worst IoU >= 0.85", worstIoU >= 0.85, String(format: "%.4f", worstIoU))
check("worst conf delta <= 0.03", worstConf <= 0.03, String(format: "%.4f", worstConf))

// Timed inference (pure detect calls, model warm).
_ = try detector.detect(in: frame, conf: 0.05)
let t0 = CFAbsoluteTimeGetCurrent()
let iters = 50
for _ in 0..<iters { _ = try detector.detect(in: frame, conf: 0.05) }
let msPerFrame = (CFAbsoluteTimeGetCurrent() - t0) * 1000 / Double(iters)
print(String(format: "timing: %.1f ms/frame (letterbox+ANE inference+decode, macOS)", msPerFrame))

// MARK: - 2. Tracker scenarios (port of pipeline/ball_tracker.py self-test)

print("\n=== Tracker scenarios (10-scenario oracle) ===")

let fps = 30.0
let dt = 1.0 / fps

func run(_ stream: [[TrackerDetection]], persp: PerspectiveScale? = nil) -> [TrackStatus] {
    let mgr = BallTrackManager(fps: fps, perspectiveScale: persp)
    var out: [TrackStatus] = []
    var t = 0.0
    for dets in stream {
        out.append(mgr.update(detections: dets, now: t))
        t += dt
    }
    return out
}

func d1(_ x: Double, _ y: Double, _ c: Double) -> [TrackerDetection] {
    [TrackerDetection(x: x, y: y, conf: c)]
}

// S1: moving ball becomes and stays a live trace.
do {
    var stream: [[TrackerDetection]] = []
    var x = 100.0
    for _ in 0..<40 { x += 18.0; stream.append(d1(x, 300.0, 0.9)) }
    let res = run(stream)
    check("S1 moving ball live trace",
          res.contains { $0.hasMovingTrace } && res.last!.hasMovingTrace
              && res.last!.state == .moving)
}

// S2: ball that stops in view ends the trace.
do {
    var stream: [[TrackerDetection]] = []
    var x = 100.0
    for _ in 0..<25 { x += 18.0; stream.append(d1(x, 300.0, 0.9)) }
    for _ in 0..<25 { stream.append(d1(x, 300.0, 0.9)) }
    let res = run(stream)
    check("S2 stopped ball ends trace",
          !res.last!.hasMovingTrace && res.last!.state == .stopped)
}

// S3: disappearing ball ends within miss_timeout.
do {
    var stream: [[TrackerDetection]] = []
    var x = 100.0
    for _ in 0..<25 { x += 18.0; stream.append(d1(x, 300.0, 0.9)) }
    for _ in 0..<100 { stream.append([]) }
    let res = run(stream)
    let aliveIdx = res.indices.filter { res[$0].hasMovingTrace }
    let lastAlive = aliveIdx.last ?? 24
    let coastS = Double(lastAlive - 24) * dt
    check("S3 lost within timeout",
          !res.last!.hasMovingTrace && coastS <= 2.0 + dt + 1e-9 && lastAlive >= 24)
}

// S4: scattered false positives never form a live trace.
do {
    var seed: UInt64 = 0x9E3779B97F4A7C15
    func nextRand() -> Double {
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        return Double(seed >> 11) / Double(UInt64(1) << 53)
    }
    var stream: [[TrackerDetection]] = []
    for _ in 0..<40 {
        if nextRand() < 0.4 {
            stream.append(d1((nextRand() * 900).rounded(.down),
                             (nextRand() * 500).rounded(.down), 0.3))
        } else {
            stream.append([])
        }
    }
    let res = run(stream)
    check("S4 scattered FPs no trace", !res.contains { $0.hasMovingTrace })
}

// S5: a stationary ball never becomes a live trace.
do {
    let stream = [[TrackerDetection]](repeating: d1(500.0, 250.0, 0.8), count: 40)
    let res = run(stream)
    check("S5 stationary ball no trace", !res.contains { $0.hasMovingTrace })
}

// S6: trace survives a brief occlusion.
do {
    var stream: [[TrackerDetection]] = []
    var x = 100.0
    for _ in 0..<20 { x += 18.0; stream.append(d1(x, 300.0, 0.9)) }
    for _ in 0..<6 { x += 18.0; stream.append([]) }
    for _ in 0..<20 { x += 18.0; stream.append(d1(x, 300.0, 0.9)) }
    let res = run(stream)
    check("S6 survives occlusion",
          res.last!.hasMovingTrace && res[25...].allSatisfy { $0.hasMovingTrace })
}

// S7: ball stays alive through a 180-degree reversal (racket).
do {
    var stream: [[TrackerDetection]] = []
    var x = 200.0
    for _ in 0..<25 { x += 30.0; stream.append(d1(x, 300.0, 0.9)) }
    stream.append([])
    for _ in 0..<25 { x -= 30.0; stream.append(d1(x, 300.0, 0.9)) }
    let res = run(stream)
    let aliveAfter = res[26...].map(\.hasMovingTrace)
    var maxDead = 0, cur = 0
    for a in aliveAfter { cur = a ? 0 : cur + 1; maxDead = max(maxDead, cur) }
    let mp = res[24..<34].map(\.maneuverProb).max()!
    let rp = res[24..<34].map(\.racketProb).max()!
    let bp = res[24..<34].map(\.bounceProb).max()!
    check("S7 racket reversal",
          aliveAfter.suffix(10).allSatisfy { $0 } && maxDead <= 2
              && mp > 0.5 && rp > bp)
}

// S8: ball stays alive through a court bounce (vy flip).
do {
    var stream: [[TrackerDetection]] = []
    var x = 200.0, y = 100.0
    for _ in 0..<25 { x += 15.0; y += 20.0; stream.append(d1(x, y, 0.9)) }
    for _ in 0..<25 { x += 15.0; y -= 20.0; stream.append(d1(x, y, 0.9)) }
    let res = run(stream)
    let aliveAfter = res[25...].map(\.hasMovingTrace)
    var maxDead = 0, cur = 0
    for a in aliveAfter { cur = a ? 0 : cur + 1; maxDead = max(maxDead, cur) }
    let mp = res[23..<33].map(\.maneuverProb).max()!
    let rp = res[23..<40].map(\.racketProb).max()!
    let bp = res[23..<40].map(\.bounceProb).max()!
    check("S8 court bounce",
          aliveAfter.suffix(10).allSatisfy { $0 } && maxDead <= 2
              && mp > 0.5 && bp > rp)
}

// S9: re-acquire ball after fast serve contact.
do {
    var stream: [[TrackerDetection]] = []
    let xt = 400.0
    var yt = 400.0
    for _ in 0..<20 { yt -= 20.0; stream.append(d1(xt, yt, 0.9)) }
    stream.append([])
    var xs = xt
    let ys = yt
    for _ in 0..<40 { xs += 100.0; stream.append(d1(xs, ys, 0.9)) }
    let res = run(stream)
    let alivePost = res[22...].map(\.hasMovingTrace)
    check("S9 serve re-acquire",
          alivePost.contains(true) && res.last!.hasMovingTrace)
}

// S10: track follows ball across sparse near->far net crossing.
do {
    let persp = makeImageRowPerspective(frameHeight: 540.0)
    let mgr = BallTrackManager(fps: fps, perspectiveScale: persp)
    var stream: [[TrackerDetection]] = []
    var truth: [(Double, Double)] = []
    var xn = 150.0, yn = 460.0
    for _ in 0..<16 {
        xn += 26.0; yn -= 11.0
        stream.append(d1(xn, yn, 0.9)); truth.append((xn, yn))
    }
    for _ in 0..<7 {
        xn += 15.0; yn -= 6.0
        stream.append([]); truth.append((xn, yn))
    }
    for i in 0..<28 {
        xn += 15.0; yn -= 6.0
        stream.append(i % 4 == 0 ? d1(xn, yn, 0.8) : []); truth.append((xn, yn))
    }
    var errs: [Double] = []
    var lastStatus: TrackStatus?
    var tt = 0.0
    for i in 0..<stream.count {
        let s = mgr.update(detections: stream[i], now: tt)
        tt += dt
        lastStatus = s
        if let p = s.position {
            errs.append(((p.x - truth[i].0) * (p.x - truth[i].0) +
                         (p.y - truth[i].1) * (p.y - truth[i].1)).squareRoot())
        }
    }
    check("S10 sparse net crossing",
          errs.last! < 60.0 && errs[20...].max()! < 120.0
              && lastStatus!.hasMovingTrace,
          String(format: "lastErr=%.1f maxErr=%.1f", errs.last!, errs[20...].max()!))
}

print("\n\(failures == 0 ? "ALL CHECKS PASSED ✅" : "\(failures) CHECK(S) FAILED ⚠️")")
exit(failures == 0 ? 0 : 1)
