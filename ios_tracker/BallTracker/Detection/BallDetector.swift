import CoreML
import CoreVideo
import Foundation

/// A decoded ball detection in source-frame pixel space.
struct BallDetection {
    let box: CGRect
    let conf: Float

    var center: CGPoint { CGPoint(x: box.midX, y: box.midY) }
}

enum BallDetectorError: Error {
    case modelMissing
    case letterboxFailed
    case badOutput
}

/// Runs the NMS-free ball_best Core ML model (raw [1,5,10710] output) fully on
/// the Neural Engine, then decodes + NMS on the CPU. Keeping NMS out of the
/// graph is deliberate: baked NonMaxSuppression/TopK ops are not
/// ANE-supported and force graph partitioning (see spikes/FINDINGS.md), and it
/// keeps the ultra-low conf thresholds the tennis pipeline relies on tunable
/// per call-site.
final class BallDetector {
    static let inputWidth = 960
    static let inputHeight = 544

    /// ACTIVE_BALL_CONF from the pipeline — the operating threshold.
    static let defaultConf: Float = 0.10
    /// Ultralytics' default NMS IoU, matching the parity fixtures.
    static let defaultIoU: Float = 0.7

    private let model: MLModel
    private let letterbox: Letterbox
    private let inputName = "image"
    private let outputName = "detections"

    convenience init(computeUnits: MLComputeUnits = .all) throws {
        guard let url = Bundle.main.url(forResource: "ball_best", withExtension: "mlmodelc") else {
            throw BallDetectorError.modelMissing
        }
        try self.init(modelURL: url, computeUnits: computeUnits)
    }

    init(modelURL: URL, computeUnits: MLComputeUnits = .all) throws {
        guard let letterbox = Letterbox(width: Self.inputWidth, height: Self.inputHeight) else {
            throw BallDetectorError.letterboxFailed
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        self.model = try MLModel(contentsOf: modelURL, configuration: cfg)
        self.letterbox = letterbox
    }

    /// Detect balls in a frame of any size; boxes come back in that frame's
    /// pixel space.
    func detect(in pixelBuffer: CVPixelBuffer,
                conf: Float = BallDetector.defaultConf,
                iouThreshold: Float = BallDetector.defaultIoU) throws -> [BallDetection] {
        guard let (input, transform) = letterbox.apply(to: pixelBuffer) else {
            throw BallDetectorError.letterboxFailed
        }
        let provider = try MLDictionaryFeatureProvider(
            dictionary: [inputName: MLFeatureValue(pixelBuffer: input)])
        let output = try model.prediction(from: provider)
        guard let arr = output.featureValue(for: outputName)?.multiArrayValue else {
            throw BallDetectorError.badOutput
        }
        return decode(arr, conf: conf, iouThreshold: iouThreshold, transform: transform)
    }

    /// Decode the raw [1,5,N] head (cx,cy,w,h,conf per anchor, channel-major),
    /// threshold, NMS, and un-letterbox to source pixels.
    private func decode(_ arr: MLMultiArray, conf: Float, iouThreshold: Float,
                        transform: LetterboxTransform) -> [BallDetection] {
        guard arr.shape.count == 3, arr.shape[1].intValue == 5 else { return [] }
        let n = arr.shape[2].intValue
        // The backing buffer can be padded (e.g. channel stride 10720 for
        // n=10710), so index via the declared strides, never densely.
        let chStride = arr.strides[1].intValue
        let anchorStride = arr.strides[2].intValue

        var candidates: [(box: CGRect, conf: Float)] = []
        arr.withUnsafeBufferPointer(ofType: Float.self) { p in
            for i in 0..<n {
                let base = i * anchorStride
                let c = p[base + 4 * chStride]
                if c < conf { continue }
                let cx = CGFloat(p[base])
                let cy = CGFloat(p[base + chStride])
                let w = CGFloat(p[base + 2 * chStride])
                let h = CGFloat(p[base + 3 * chStride])
                candidates.append((CGRect(x: cx - w / 2, y: cy - h / 2, width: w, height: h), c))
            }
        }

        let kept = nms(candidates, iouThreshold: iouThreshold)
        return kept.map { cand in
            let o = transform.unmap(cand.box.origin)
            let box = CGRect(x: o.x, y: o.y,
                             width: cand.box.width / transform.scale,
                             height: cand.box.height / transform.scale)
            return BallDetection(box: box, conf: cand.conf)
        }
    }

    /// Greedy single-class non-maximum suppression, highest confidence first.
    private func nms(_ candidates: [(box: CGRect, conf: Float)],
                     iouThreshold: Float) -> [(box: CGRect, conf: Float)] {
        guard candidates.count > 1 else { return candidates }
        let sorted = candidates.sorted { $0.conf > $1.conf }
        var kept: [(box: CGRect, conf: Float)] = []
        for cand in sorted {
            var suppressed = false
            for k in kept where iou(cand.box, k.box) > iouThreshold {
                suppressed = true
                break
            }
            if !suppressed { kept.append(cand) }
        }
        return kept
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let inter = a.intersection(b)
        if inter.isNull || inter.isEmpty { return 0 }
        let interArea = inter.width * inter.height
        let union = a.width * a.height + b.width * b.height - interArea
        return union > 0 ? Float(interArea / union) : 0
    }
}
