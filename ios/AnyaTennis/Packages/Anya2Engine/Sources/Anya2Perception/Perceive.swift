// Perceive.swift  (perceive.near / perceive.far / _people_in_result / _to_analysis_frame)
//
// One pose sample = two model calls:
//   near pass  the whole frame at 960x540 (analysis px), letterboxed to 640
//   far pass   the far band at native resolution, letterboxed to 960, then
//              mapped back to analysis px
// Each pass keeps its MAX_PERSONS most confident people; the far pass then
// drops anyone whose box bottom touches the crop's lower edge (the crop cut
// them off, so the box bottom is not their feet). The order matters and is
// main's: keep the top 8 FIRST, then drop.
import Anya2Engine
import CoreML
import CoreVideo
import Foundation

public enum PerceiveRules {
    public static let N_KEYPOINTS = 17
    public static let MAX_PERSONS = 8      // people kept per frame, highest confidence first
    public static let POSE_FPS = 15.0      // pose samples per second, both passes
    public static let FAR_IMGSZ = 960      // the far player is at the model's limit; 768 lost recall
    public static let NEAR_IMGSZ = 640     // near players are large; 640 is as accurate and cheaper

    /// perceive._stride_for: (integer stride >= 1, the rate it gives).
    public static func stride(sourceFPS: Double, poseFPS: Double = POSE_FPS) -> (stride: Int, fps: Double) {
        let stride = max(1, pythonRound(sourceFPS / poseFPS))
        return (stride, sourceFPS / Double(stride))
    }
}

/// One pass's people for one sample, in the PoseDetections row layout:
/// MAX_PERSONS rows, NaN where nobody is.
public struct PassRow {
    public var keypoints = [Float](repeating: .nan, count: PerceiveRules.MAX_PERSONS * 51)
    public var boxes = [Float](repeating: .nan, count: PerceiveRules.MAX_PERSONS * 4)
    public var confidences = [Float](repeating: .nan, count: PerceiveRules.MAX_PERSONS)

    public init() {}

    /// `people` in the pass's own image pixels, most confident first.
    /// `bottomEdge`: the far crop's height (drop boxes touching it), or nil.
    /// `toAnalysis`: (scale, offset) per axis, applied in float32 as numpy does.
    public init(_ people: [DetectedPerson], bottomEdge: Double? = nil,
                toAnalysis: (scaleX: Float, offsetX: Float, scaleY: Float, offsetY: Float) = (1, 0, 1, 0)) {
        self.init()
        var kept = Array(people.prefix(PerceiveRules.MAX_PERSONS))
        if let bottomEdge {
            kept = kept.filter { Float($0.box.y2) < Float(bottomEdge - FarBandRules.FAR_BAND_EDGE_PX) }
        }
        let m = toAnalysis
        for (row, person) in kept.enumerated() {
            boxes[row * 4] = Float(person.box.x1) * m.scaleX + m.offsetX
            boxes[row * 4 + 1] = Float(person.box.y1) * m.scaleY + m.offsetY
            boxes[row * 4 + 2] = Float(person.box.x2) * m.scaleX + m.offsetX
            boxes[row * 4 + 3] = Float(person.box.y2) * m.scaleY + m.offsetY
            confidences[row] = Float(person.confidence)
            for k in 0..<PerceiveRules.N_KEYPOINTS {
                let base = row * 51 + k * 3
                keypoints[base] = Float(person.keypoints[k * 3]) * m.scaleX + m.offsetX
                keypoints[base + 1] = Float(person.keypoints[k * 3 + 1]) * m.scaleY + m.offsetY
                keypoints[base + 2] = Float(person.keypoints[k * 3 + 2])
            }
        }
    }
}

/// Both passes for one sample.
public struct PerceivedSample {
    public var near: PassRow
    public var far: PassRow

    public init(near: PassRow, far: PassRow) { self.near = near; self.far = far }

    /// The tracker's input: near people first, then far, as tracks.build concatenates them.
    public var pooled: PooledSample {
        PooledSample(people: 2 * PerceiveRules.MAX_PERSONS,
                     keypoints: ArraySlice(near.keypoints + far.keypoints),
                     boxes: ArraySlice(near.boxes + far.boxes),
                     confidences: ArraySlice(near.confidences + far.confidences))
    }
}

/// Runs the two passes on frames from the camera (or a file).
///
/// One model INSTANCE per pass: Core ML re-plans whenever an instance's input
/// shape changes, and alternating 640x384 / 960xH on one instance cost 60 ms
/// a sample on the Neural Engine instead of 6.6 ms (M-series Mac, measured).
public final class Perceiver {
    public let nearModel: PoseModel
    public let farModel: PoseModel
    public let band: FarBand
    private let nearGeometry: LetterboxGeometry
    private let farGeometry: LetterboxGeometry
    private let farToAnalysis: (scaleX: Float, offsetX: Float, scaleY: Float, offsetY: Float)

    /// `compiledModelURL`: PoseModel.mlmodelc.
    public convenience init(compiledModelURL: URL, computeUnits: MLComputeUnits = .all, band: FarBand) throws {
        self.init(nearModel: try PoseModel(compiledModelURL: compiledModelURL, computeUnits: computeUnits),
                  farModel: try PoseModel(compiledModelURL: compiledModelURL, computeUnits: computeUnits), band: band)
    }

    public init(nearModel: PoseModel, farModel: PoseModel, band: FarBand) {
        self.nearModel = nearModel; self.farModel = farModel
        self.band = band
        nearGeometry = LetterboxGeometry(sourceWidth: Court.ANALYSIS_WIDTH, sourceHeight: Court.ANALYSIS_HEIGHT,
                                         imageSize: PerceiveRules.NEAR_IMGSZ)
        farGeometry = LetterboxGeometry(sourceWidth: band.encodedWidth, sourceHeight: band.encodedHeight,
                                        imageSize: PerceiveRules.FAR_IMGSZ)
        // perceive.far: offset = crop / scale, scale = 1 / scale (Python floats,
        // cast to float32 when they meet the float32 detections).
        farToAnalysis = (Float(1 / band.scaleX), Float(Double(band.x1) / band.scaleX),
                         Float(1 / band.scaleY), Float(Double(band.y1) / band.scaleY))
    }

    /// `nearFrame`: the 960x540 BGRA frame. `farBandFrame`: the far band at
    /// native resolution (encodedWidth x encodedHeight), BGRA.
    public func perceive(nearFrame: CVPixelBuffer, farBandFrame: CVPixelBuffer) throws -> PerceivedSample {
        guard let nearInput = letterbox(nearFrame, geometry: nearGeometry),
              let farInput = letterbox(farBandFrame, geometry: farGeometry)
        else { throw PerceiveError.couldNotAllocateBuffer }
        let near = try nearModel.people(in: nearInput, geometry: nearGeometry)
        let far = try farModel.people(in: farInput, geometry: farGeometry)
        return PerceivedSample(near: PassRow(near),
                               far: PassRow(far, bottomEdge: Double(band.height), toAnalysis: farToAnalysis))
    }
}

public enum PerceiveError: Error { case couldNotAllocateBuffer }
