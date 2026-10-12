// PoseModel.swift
//
// yolov8n-pose on Core ML (Resources/PoseModel.mlpackage, exported by
// tools/export_pose_coreml.py) and the decode Ultralytics would have done.
//
// The model ends at the head's three RAW maps so that one model serves every
// input shape; decoding them here is the same arithmetic as Ultralytics':
//   box     4 sides x 16-bin distribution ("DFL"): softmax, expected bin,
//           then (cell centre -+ distance) x stride
//   score   sigmoid of the person logit
//   keypoint (offset x 2 + cell centre - 0.5) x stride; visibility = sigmoid
// then NMS exactly as Ultralytics: score strictly above POSE_CONF, greedy,
// suppress IoU above POSE_IOU, at most 300 people.
import Anya2Engine
import CoreML
import CoreVideo
import Foundation

public enum PoseModelRules {
    public static let POSE_CONF = 0.20     // perceive.py POSE_CONF
    public static let POSE_IOU = 0.70      // Ultralytics' default NMS IoU (perceive.py never overrides it)
    public static let MAX_DETECTIONS = 300 // Ultralytics' max_det
    public static let STRIDES = [8, 16, 32]
    static let DFL_BINS = 16
    static let CHANNELS = 4 * 16 + 1 + 17 * 3   // 116
}

/// One person the model found, in the coordinates of the image it was given.
public struct DetectedPerson {
    public var box: (x1: Double, y1: Double, x2: Double, y2: Double)
    public var confidence: Double
    public var keypoints: [Double]   // 17 x (x, y, visibility)
}

public final class PoseModel {
    private let model: MLModel

    /// `computeUnits`: .all lets Core ML use the Neural Engine (fp16);
    /// .cpuOnly is the deterministic reference used by parity tests.
    public init(compiledModelURL: URL, computeUnits: MLComputeUnits = .all) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        #if targetEnvironment(simulator)
        configuration.computeUnits = .cpuOnly   // the simulator has no Neural Engine
        #endif
        model = try MLModel(contentsOf: compiledModelURL, configuration: configuration)
    }

    /// Compile an .mlpackage at runtime (tests and tools; the app bundles a compiled model).
    public convenience init(packageURL: URL, computeUnits: MLComputeUnits = .all) throws {
        try self.init(compiledModelURL: try MLModel.compileModel(at: packageURL), computeUnits: computeUnits)
    }

    /// People in a letterboxed frame, mapped back to the SOURCE frame's pixels.
    public func people(in letterboxed: CVPixelBuffer, geometry: LetterboxGeometry) throws -> [DetectedPerson] {
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: letterboxed)])
        let output = try model.prediction(from: input)
        var candidates: [DetectedPerson] = []
        for stride in PoseModelRules.STRIDES {
            guard let map = output.featureValue(for: "p\(stride)")?.multiArrayValue else { continue }
            decode(map, stride: stride, into: &candidates)
        }
        return nonMaximumSuppression(candidates).map { person in
            var mapped = person
            let topLeft = geometry.toSource(x: person.box.x1, y: person.box.y1)
            let bottomRight = geometry.toSource(x: person.box.x2, y: person.box.y2)
            mapped.box = (topLeft.x, topLeft.y, bottomRight.x, bottomRight.y)
            for k in 0..<17 {
                let p = geometry.toSource(x: person.keypoints[k * 3], y: person.keypoints[k * 3 + 1])
                mapped.keypoints[k * 3] = p.x; mapped.keypoints[k * 3 + 1] = p.y
            }
            return mapped
        }
    }

    /// One raw map [1, 116, h, w] -> people with score > POSE_CONF.
    /// Read through the array's STRIDES: Core ML may pad rows (see memory
    /// "iOS tracker Swift port"), so dense indexing would read garbage.
    private func decode(_ map: MLMultiArray, stride: Int, into people: inout [DetectedPerson]) {
        let height = map.shape[2].intValue, width = map.shape[3].intValue
        let channelStride = map.strides[1].intValue, rowStride = map.strides[2].intValue
        let columnStride = map.strides[3].intValue
        let read: (Int) -> Double
        switch map.dataType {
        case .float16:
            let pointer = map.dataPointer.assumingMemoryBound(to: Float16.self)
            read = { Double(pointer[$0]) }
        case .float32:
            let pointer = map.dataPointer.assumingMemoryBound(to: Float.self)
            read = { Double(pointer[$0]) }
        default:
            let pointer = map.dataPointer.assumingMemoryBound(to: Double.self)
            read = { pointer[$0] }
        }
        let s = Double(stride)
        for row in 0..<height {
            for column in 0..<width {
                let cell = row * rowStride + column * columnStride
                func value(_ channel: Int) -> Double { read(cell + channel * channelStride) }
                let score = 1 / (1 + exp(-value(64)))
                guard score > PoseModelRules.POSE_CONF else { continue }
                let cx = Double(column) + 0.5, cy = Double(row) + 0.5
                var sides = [Double](repeating: 0, count: 4)   // left, top, right, bottom
                for side in 0..<4 {
                    var bins = (0..<PoseModelRules.DFL_BINS).map { value(side * PoseModelRules.DFL_BINS + $0) }
                    let peak = bins.max()!
                    bins = bins.map { exp($0 - peak) }
                    let total = bins.reduce(0, +)
                    sides[side] = bins.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element } / total
                }
                var keypoints = [Double](repeating: 0, count: 51)
                for k in 0..<17 {
                    keypoints[k * 3] = (value(65 + k * 3) * 2 + (cx - 0.5)) * s
                    keypoints[k * 3 + 1] = (value(66 + k * 3) * 2 + (cy - 0.5)) * s
                    keypoints[k * 3 + 2] = 1 / (1 + exp(-value(67 + k * 3)))
                }
                people.append(DetectedPerson(box: ((cx - sides[0]) * s, (cy - sides[1]) * s,
                                                   (cx + sides[2]) * s, (cy + sides[3]) * s),
                                             confidence: score, keypoints: keypoints))
            }
        }
    }

    /// Greedy NMS, strongest first; suppress overlaps above POSE_IOU.
    private func nonMaximumSuppression(_ candidates: [DetectedPerson]) -> [DetectedPerson] {
        let ordered = stableSorted(candidates) { $0.confidence > $1.confidence }
        var kept: [DetectedPerson] = []
        for candidate in ordered where kept.count < PoseModelRules.MAX_DETECTIONS {
            if kept.allSatisfy({ intersectionOverUnion($0.box, candidate.box) <= PoseModelRules.POSE_IOU }) {
                kept.append(candidate)
            }
        }
        return kept
    }
}

func intersectionOverUnion(_ a: (x1: Double, y1: Double, x2: Double, y2: Double),
                           _ b: (x1: Double, y1: Double, x2: Double, y2: Double)) -> Double {
    let width = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)), height = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    let intersection = width * height
    let union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersection
    return union > 0 ? intersection / union : 0
}
