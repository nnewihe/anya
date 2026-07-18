import CoreML
import CoreVideo
import Foundation

enum PlayerDetectorError: Error {
    case modelMissing
    case letterboxFailed
    case badOutput
}

/// Detects every player on court with the Ultralytics YOLO person model
/// (pipeline/models/yolo26n.pt, exported to Core ML) and tracks each box across
/// frames so its centroid velocity is available to the ball tracker's
/// carry-suppression. Replaces the earlier Apple Vision human-rectangle path.
///
/// YOLO26 is an end-to-end (NMS-free) detector: the model emits a fixed
/// `[1, 300, 6]` head of `[x1, y1, x2, y2, conf, class]` rows in model-input
/// pixels, so decoding is a threshold + person-class filter with no Swift NMS.
///
/// Boxes are returned in the tracker's analysis space (top-left origin) to match
/// the ball detections. The pipeline classifies near/far by homography world
/// coordinates; with no homography on-device the feet row (box bottom edge)
/// stands in for court depth — near = lowest feet in the bottom half, far =
/// lowest feet among the rest of the top half (keeps higher-footed back-fence
/// spectators from winning the far slot).
final class PlayerDetector {
    /// ACTIVE_PLAYER_STRIDE — boxes change slowly and only steer carry
    /// suppression and origin labelling, so detection runs every `stride`
    /// frames with the last result held in between.
    static let stride = 4

    /// Person confidence. Lower than the pipeline's 0.5 because the far player
    /// is small at this input resolution (measured ~0.37); the boxes only steer
    /// suppression/origin, so admitting a few weak ones is cheap.
    static let defaultConf: Float = 0.25

    private let analysisWidth: Double
    private let conf: Float
    private let model: MLModel
    private let letterbox: Letterbox
    private static let inputName = "image"
    private static let outputName = "detections"

    /// Reference analysis width the pixel gates below are quoted at; they scale
    /// with the actual analysis width so 960- and 1920-wide callers behave alike.
    private static let referenceWidth = 1920.0
    /// Greedy centroid match gate for associating a box to a track (players move
    /// slowly, so this is generous).
    private let matchGatePx: Double
    /// Drop a track unseen for this long (a player left / detection dropped out).
    private static let trackTimeoutSec = 1.0

    private var frameCounter = 0
    private var cached = PlayerBoxes.none
    private var tracks: [PlayerTrack] = []
    private var lastKnownFar: PlayerBox?

    /// One tracked player: latest box plus a smoothed centroid velocity.
    private final class PlayerTrack {
        var box: PlayerBox
        var lastCenter: (x: Double, y: Double)
        var lastSeenT: Double
        let vel: SmoothedVelocity

        init(box: PlayerBox, t: Double, windowSec: Double) {
            self.box = box
            self.lastCenter = box.center
            self.lastSeenT = t
            self.vel = SmoothedVelocity(windowSec: windowSec)
            let c = box.center
            vel.add(t: t, x: c.x, y: c.y)
        }
    }

    convenience init(analysisWidth: Double,
                     computeUnits: MLComputeUnits = BallDetector.defaultComputeUnits) throws {
        guard let url = Bundle.main.url(forResource: "yolo26n", withExtension: "mlmodelc") else {
            throw PlayerDetectorError.modelMissing
        }
        try self.init(analysisWidth: analysisWidth, modelURL: url, computeUnits: computeUnits)
    }

    init(analysisWidth: Double, modelURL: URL,
         conf: Float = PlayerDetector.defaultConf,
         computeUnits: MLComputeUnits = BallDetector.defaultComputeUnits) throws {
        self.analysisWidth = analysisWidth
        self.conf = conf
        self.matchGatePx = 150.0 * analysisWidth / Self.referenceWidth

        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        let model = try MLModel(contentsOf: modelURL, configuration: cfg)
        guard let constraint = model.modelDescription
            .inputDescriptionsByName[Self.inputName]?.imageConstraint else {
            throw PlayerDetectorError.badOutput
        }
        guard let letterbox = Letterbox(width: constraint.pixelsWide,
                                        height: constraint.pixelsHigh) else {
            throw PlayerDetectorError.letterboxFailed
        }
        self.model = model
        self.letterbox = letterbox
    }

    /// Detect (or serve cached) player boxes for one frame at time `t`, in
    /// analysis space (`analysisWidth`-wide, top-left origin).
    func update(pixelBuffer: CVPixelBuffer, t: Double) -> PlayerBoxes {
        frameCounter += 1
        if frameCounter % Self.stride != 0, !cached.all.isEmpty {
            return cached
        }
        cached = detect(pixelBuffer: pixelBuffer, t: t)
        return cached
    }

    private func detect(pixelBuffer: CVPixelBuffer, t: Double) -> PlayerBoxes {
        let sourceW = Double(CVPixelBufferGetWidth(pixelBuffer))
        let sourceH = Double(CVPixelBufferGetHeight(pixelBuffer))
        guard sourceW > 0 else { return .none }
        let scale = analysisWidth / sourceW

        guard let (input, transform) = letterbox.apply(to: pixelBuffer),
              let output = try? model.prediction(from: MLDictionaryFeatureProvider(
                  dictionary: [Self.inputName: MLFeatureValue(pixelBuffer: input)])),
              let arr = output.featureValue(for: Self.outputName)?.multiArrayValue else {
            return .none
        }

        // Decode person boxes → analysis space.
        var boxes: [PlayerBox] = []
        decodeRows(arr) { x1, y1, x2, y2 in
            let a = transform.unmap(CGPoint(x: x1, y: y1))
            let b = transform.unmap(CGPoint(x: x2, y: y2))
            boxes.append(PlayerBox(x1: Double(a.x) * scale, y1: Double(a.y) * scale,
                                   x2: Double(b.x) * scale, y2: Double(b.y) * scale,
                                   vx: nil, vy: nil))
        }

        // Attach a smoothed velocity to each box by matching it to a track.
        let tracked = updateTracks(boxes, t: t)
        var obs = classify(tracked, sourceH: sourceH, scale: scale)
        if obs.far != nil {
            lastKnownFar = obs.far
        } else {
            obs.far = lastKnownFar
        }
        return obs
    }

    /// Walk the `[1, 300, 6]` head, calling `emit` for every person-class row
    /// above threshold. Indexes via the reported strides with the same
    /// torn-buffer fallback the ball decoder uses (the Simulator's CPU backend
    /// can report padded strides over a dense buffer).
    private func decodeRows(_ arr: MLMultiArray,
                            _ emit: (Double, Double, Double, Double) -> Void) {
        guard arr.shape.count == 3, arr.shape[2].intValue >= 6 else { return }
        let rows = arr.shape[1].intValue
        let cols = arr.shape[2].intValue
        guard rows > 0 else { return }
        let rowStrideRep = arr.strides[1].intValue
        let colStrideRep = arr.strides[2].intValue

        arr.withUnsafeBufferPointer(ofType: Float.self) { p in
            var rowS = rowStrideRep
            var colS = colStrideRep
            if rowS < 1 || colS < 1 || (rows - 1) * rowS + (cols - 1) * colS >= p.count {
                rowS = cols
                colS = 1
            }
            guard (rows - 1) * rowS + (cols - 1) * colS < p.count else { return }
            for i in 0..<rows {
                let base = i * rowS
                let c = p[base + 4 * colS]
                if c < conf { continue }
                // Class id lives in the last column; keep persons only.
                if Int(p[base + 5 * colS].rounded()) != 0 { continue }
                emit(Double(p[base]), Double(p[base + colS]),
                     Double(p[base + 2 * colS]), Double(p[base + 3 * colS]))
            }
        }
    }

    /// Greedy nearest-centroid association of this frame's boxes to persistent
    /// tracks, returning each box with its track's smoothed velocity attached.
    private func updateTracks(_ boxes: [PlayerBox], t: Double) -> [PlayerBox] {
        tracks.removeAll { t - $0.lastSeenT > Self.trackTimeoutSec }
        var claimed = [Bool](repeating: false, count: tracks.count)
        var out: [PlayerBox] = []
        out.reserveCapacity(boxes.count)

        for box in boxes {
            let c = box.center
            var bestJ = -1
            var bestD = matchGatePx
            for (j, tr) in tracks.enumerated() where !claimed[j] {
                let d = (( c.x - tr.lastCenter.x) * (c.x - tr.lastCenter.x)
                       + (c.y - tr.lastCenter.y) * (c.y - tr.lastCenter.y)).squareRoot()
                if d <= bestD { bestD = d; bestJ = j }
            }
            let tr: PlayerTrack
            if bestJ >= 0 {
                claimed[bestJ] = true
                tr = tracks[bestJ]
                tr.vel.add(t: t, x: c.x, y: c.y)
                tr.lastCenter = c
                tr.lastSeenT = t
            } else {
                tr = PlayerTrack(box: box, t: t,
                                 windowSec: RallyConstants.couplingWindowSec)
                tracks.append(tr)
                claimed.append(true)
            }
            var b = box
            if let v = tr.vel.velocity() {
                b.vx = v.vx
                b.vy = v.vy
            }
            tr.box = b
            out.append(b)
        }
        return out
    }

    /// Derive the near/far pair from all detected boxes via the feet-row rule.
    private func classify(_ boxes: [PlayerBox], sourceH: Double,
                          scale: Double) -> PlayerBoxes {
        guard !boxes.isEmpty else { return .none }
        let midY = sourceH * scale / 2.0
        let near = boxes.filter { $0.y2 >= midY }.max { $0.y2 < $1.y2 }
        let far = boxes.filter { $0.y2 < midY }.max { $0.y2 < $1.y2 }
        return PlayerBoxes(all: boxes, near: near, far: far)
    }
}
