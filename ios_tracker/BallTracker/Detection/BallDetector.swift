import CoreImage
import CoreML
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers

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

    /// ACTIVE_BALL_CONF from the pipeline — the operating threshold for the
    /// online tracker, which must commit to an association every frame and so
    /// needs precision over recall.
    static let defaultConf: Float = 0.10

    /// Threshold for the offline Viterbi solver, which wants the opposite
    /// trade: it discriminates by trajectory dynamics, so it would rather sift
    /// a pile of weak candidates than never see the ball. Measured on real
    /// footage, at 0.10 the ball is missed in 48% of frames; at 0.03, 4%.
    /// Only safe because exclusion zones already fence off stationary clutter.
    static let solverConf: Float = 0.10
    /// Ultralytics' default NMS IoU, matching the parity fixtures.
    static let defaultIoU: Float = 0.7

    /// Upper bound on a ball box, in model-input px (960-wide space).
    ///
    /// Deliberately loose. It is tempting to derive this from geometry — a
    /// 6.7 cm ball at 18–93 ft through this lens is only ~1–6 px — but the
    /// model boxes the *motion streak*, not the ball: at 30 fps a 60 mph ball
    /// smears ~26 px across a frame. Measured on real footage, real-ball boxes
    /// run 9–38 px wide and barely vary with distance (1.4× top-to-bottom,
    /// where geometry demands ~5×), so box size tracks ball *speed*, not range.
    /// A physical bound would reject every real ball. This only fences off
    /// grossly non-ball blobs (players, bags, chairs).
    static let defaultMaxBoxPx: Float = 45

    /// The Simulator has no ANE and its MPSGraph backend doesn't reliably
    /// support this model's ops, so `.all` there throws an internal Espresso
    /// exception (MpsGraph backend validation on incompatible OS) that can
    /// leave the prediction output torn and crash on decode. CPU-only is the
    /// only compute path the Simulator can run this model on.
    static let defaultComputeUnits: MLComputeUnits = {
        #if targetEnvironment(simulator)
        return .cpuOnly
        #else
        return .all
        #endif
    }()

    private let model: MLModel
    private let letterbox: Letterbox
    private let inputName = "image"
    private let outputName = "detections"

    /// When BALL_DEBUG_DUMP is set in the environment, print the top raw
    /// confidence per frame (pre-threshold) and save the first few letterboxed
    /// 960×544 model-input frames as PNGs, so the exact on-device input can be
    /// diffed against the Python/Ultralytics letterbox.
    private let debugDump = ProcessInfo.processInfo.environment["BALL_DEBUG_DUMP"] != nil
    private var dumpCount = 0
    private let dumpMax = 5
    private let debugCIContext = CIContext(options: [.cacheIntermediates: false])

    convenience init(computeUnits: MLComputeUnits = BallDetector.defaultComputeUnits) throws {
        guard let url = Bundle.main.url(forResource: "ball_best", withExtension: "mlmodelc") else {
            throw BallDetectorError.modelMissing
        }
        try self.init(modelURL: url, computeUnits: computeUnits)
    }

    init(modelURL: URL, computeUnits: MLComputeUnits = BallDetector.defaultComputeUnits) throws {
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
                iouThreshold: Float = BallDetector.defaultIoU,
                maxBoxPx: Float = BallDetector.defaultMaxBoxPx) throws -> [BallDetection] {
        guard let (input, transform) = letterbox.apply(to: pixelBuffer) else {
            throw BallDetectorError.letterboxFailed
        }
        if debugDump { dumpLetterboxedInput(input) }
        let provider = try MLDictionaryFeatureProvider(
            dictionary: [inputName: MLFeatureValue(pixelBuffer: input)])
        let output = try model.prediction(from: provider)
        guard let arr = output.featureValue(for: outputName)?.multiArrayValue else {
            throw BallDetectorError.badOutput
        }
        return decode(arr, conf: conf, iouThreshold: iouThreshold,
                      maxBoxPx: maxBoxPx, transform: transform)
    }

    /// Saves the letterboxed model input as a PNG and logs where, so it can be
    /// pulled off the device/simulator and re-run through Python.
    private func dumpLetterboxedInput(_ input: CVPixelBuffer) {
        guard dumpCount < dumpMax else { return }
        let idx = dumpCount
        dumpCount += 1
        let ci = CIImage(cvPixelBuffer: input)
        guard let cg = debugCIContext.createCGImage(ci, from: ci.extent) else { return }
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let url = dir.appendingPathComponent("letterbox_\(idx).png")
        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(dest, cg, nil)
        if CGImageDestinationFinalize(dest) {
            print("[BallDetector] dumped letterboxed input -> \(url.path)")
        }
    }

    /// Decode the raw [1,5,N] head (cx,cy,w,h,conf per anchor, channel-major),
    /// threshold, NMS, and un-letterbox to source pixels.
    private func decode(_ arr: MLMultiArray, conf: Float, iouThreshold: Float,
                        maxBoxPx: Float, transform: LetterboxTransform) -> [BallDetection] {
        guard arr.shape.count == 3, arr.shape[1].intValue == 5 else { return [] }
        let n = arr.shape[2].intValue
        // The backing buffer can be padded (e.g. channel stride 10720 for
        // n=10710), so index via the declared strides, never densely.
        let declaredChStride = arr.strides[1].intValue
        let declaredAnchorStride = arr.strides[2].intValue

        var candidates: [(box: CGRect, conf: Float)] = []
        var maxConf: Float = 0
        arr.withUnsafeBufferPointer(ofType: Float.self) { p in
            // On device (and macOS, .all/.cpuOnly) the backing buffer is
            // allocated at the *padded* size, so the declared strides read in
            // bounds. The iOS Simulator's CPU/MPS backend reports the same
            // padded strides but hands back a buffer only the *dense* size —
            // stride math then runs past the allocation and crashes with
            // EXC_BAD_ACCESS on decode. Only trust the declared strides while
            // the highest index they produce stays in bounds; otherwise fall
            // back to dense channel-major strides ([5n, n, 1]).
            let maxDeclaredIdx = (n - 1) * declaredAnchorStride + 4 * declaredChStride
            let stridesInBounds = maxDeclaredIdx < p.count
            let chStride = stridesInBounds ? declaredChStride : n
            let anchorStride = stridesInBounds ? declaredAnchorStride : 1
            for i in 0..<n {
                let base = i * anchorStride
                let c = p[base + 4 * chStride]
                if c > maxConf { maxConf = c }
                if c < conf { continue }
                let cx = CGFloat(p[base])
                let cy = CGFloat(p[base + chStride])
                let w = CGFloat(p[base + 2 * chStride])
                let h = CGFloat(p[base + 3 * chStride])
                if Float(w) > maxBoxPx || Float(h) > maxBoxPx { continue }
                candidates.append((CGRect(x: cx - w / 2, y: cy - h / 2, width: w, height: h), c))
            }
        }

        if debugDump {
            print(String(format: "[BallDetector] maxRawConf=%.4f  above %.2f=%d",
                         maxConf, conf, candidates.count))
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
