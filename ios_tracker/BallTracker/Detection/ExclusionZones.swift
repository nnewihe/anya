import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation

/// Rectangles (in 960-wide analysis space) fencing off stationary ball-like
/// clutter — ball baskets, balls sitting on the court. Detections whose centre
/// lands inside one are dropped before they reach the tracker.
///
/// Port of `_is_in_exclusion_zone` / `create_auto_exclusion_zones` from
/// pipeline/utilities.py. This matters more than it looks: a ball basket is
/// detected at ~0.8 conf on *every* frame, so a track that latches onto it is
/// never starved of detections, never times out, and — because the hijack
/// guard only reopens once a track goes stale — blocks the real ball from ever
/// being promoted. Fencing the basket off is what keeps the tracker honest.
struct ExclusionZones {
    /// Zones in analysis space.
    let rects: [CGRect]

    static let none = ExclusionZones(rects: [])

    var isEmpty: Bool { rects.isEmpty }

    /// Inclusive on all edges, matching the Python `x1 <= x <= x2` test.
    func contains(x: Double, y: Double) -> Bool {
        for r in rects where x >= r.minX && x <= r.maxX && y >= r.minY && y <= r.maxY {
            return true
        }
        return false
    }
}

/// DBSCAN over 2-D points, matching scikit-learn's semantics: a point's
/// eps-neighbourhood includes the point itself, so `minSamples` counts it.
/// Returns one label per point; -1 means noise.
func dbscan(_ pts: [CGPoint], eps: Double, minSamples: Int) -> [Int] {
    let unvisited = -2
    let noise = -1
    var labels = [Int](repeating: unvisited, count: pts.count)
    var cluster = 0
    let eps2 = eps * eps

    func neighbours(of i: Int) -> [Int] {
        var out: [Int] = []
        for j in 0..<pts.count {
            let dx = Double(pts[i].x - pts[j].x)
            let dy = Double(pts[i].y - pts[j].y)
            if dx * dx + dy * dy <= eps2 { out.append(j) }
        }
        return out
    }

    for i in 0..<pts.count {
        guard labels[i] == unvisited else { continue }
        let seeds = neighbours(of: i)
        if seeds.count < minSamples {
            labels[i] = noise      // may still be claimed later as a border point
            continue
        }
        labels[i] = cluster

        var queue = seeds.filter { $0 != i }
        var queued = [Bool](repeating: false, count: pts.count)
        queued[i] = true
        for s in queue { queued[s] = true }

        var k = 0
        while k < queue.count {
            let j = queue[k]
            k += 1
            if labels[j] == noise { labels[j] = cluster }   // border point
            guard labels[j] == unvisited else { continue }
            labels[j] = cluster
            let jn = neighbours(of: j)
            if jn.count >= minSamples {                     // core point — expand
                for m in jn where !queued[m] {
                    queued[m] = true
                    queue.append(m)
                }
            }
        }
        cluster += 1
    }
    return labels
}

enum ExclusionZoneScanner {
    /// Sample frames across the clip, detect balls, and cluster the centres
    /// that keep landing in the same place. Defaults mirror the production call
    /// in anya_base.py (num_frames=50, conf=0.04, eps=12) except `padding`,
    /// widened from the pipeline's 0: measured DBSCAN cores on real footage run
    /// a few px across (e.g. 5×1), tight enough that a solver conf down at 0.03
    /// still picks up clutter just outside the fenced rect.
    ///
    /// Deviation from the pipeline: it scans at `BALL_IMGSZ=1920` to squeeze
    /// faint basket balls out of a stationary cluster, but the Core ML model is
    /// exported at a fixed 960×544 so we scan at native size instead. That's
    /// fine here — a basket detects at ~0.8 even at 960 — but it means very
    /// faint clutter the 1920 scan would catch can slip through.
    static func scan(asset: AVAsset,
                     detector: BallDetector,
                     analysisScale: CGFloat,
                     sampleCount: Int = 200,
                     conf: Float = 0.02,
                     eps: Double = 18,
                     minSamples: Int = 15,
                     padding: Double = 10) async throws -> ExclusionZones {
        let duration = try await asset.load(.duration)
        guard duration.seconds > 0 else { return .none }

        // Spread samples evenly rather than randomly: deterministic runs, and
        // even coverage beats random sampling at the same count.
        let times: [NSValue] = (0..<sampleCount).map { i in
            let f = (Double(i) + 0.5) / Double(sampleCount)
            return NSValue(time: CMTime(seconds: duration.seconds * f, preferredTimescale: 600))
        }

        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true   // display-oriented, matches VideoProcessor
        gen.requestedTimeToleranceBefore = CMTime(seconds: 0.2, preferredTimescale: 600)
        gen.requestedTimeToleranceAfter = CMTime(seconds: 0.2, preferredTimescale: 600)

        var centres: [CGPoint] = []
        for await result in gen.images(for: times.map { $0.timeValue }) {
            guard let cg = try? result.image else { continue }
            guard let pb = pixelBuffer(from: cg) else { continue }
            guard let dets = try? detector.detect(in: pb, conf: conf) else { continue }
            for d in dets {
                centres.append(CGPoint(x: d.center.x * analysisScale,
                                       y: d.center.y * analysisScale))
            }
        }

        if centres.count < minSamples { return .none }

        let labels = dbscan(centres, eps: eps, minSamples: minSamples)
        var rects: [CGRect] = []
        for k in Set(labels) where k >= 0 {
            let pts = zip(centres, labels).filter { $0.1 == k }.map(\.0)
            guard let first = pts.first else { continue }
            var minX = first.x, maxX = first.x, minY = first.y, maxY = first.y
            for p in pts {
                minX = min(minX, p.x); maxX = max(maxX, p.x)
                minY = min(minY, p.y); maxY = max(maxY, p.y)
            }
            rects.append(CGRect(x: minX - padding, y: minY - padding,
                                width: (maxX - minX) + 2 * padding,
                                height: (maxY - minY) + 2 * padding))
        }
        return ExclusionZones(rects: rects)
    }

    private static func pixelBuffer(from cg: CGImage) -> CVPixelBuffer? {
        let w = cg.width, h = cg.height
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA,
                                  attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let pb else { return nil }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(pb), width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue |
                CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return pb
    }
}
