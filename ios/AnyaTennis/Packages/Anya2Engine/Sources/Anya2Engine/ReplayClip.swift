// ReplayClip.swift
//
// Reads one clip written by ios/AnyaTennis/tools/export_replay.py: the
// corpus clip's cached pose passes (the INPUT) and main's own stage outputs
// (the REFERENCE the Swift pipeline is checked against). Used by the parity
// tests and the anya2-replay tool; nothing in the app depends on it.
import Foundation

public struct ReplayClip {
    public struct ArrayLocation: Decodable {
        public let offset: Int
        public let count: Int
        public let shape: [Int]
        public let dtype: String
    }

    public struct CameraJSON: Decodable {
        public let frames: [Int]
        public let warps: [[Double]]
    }

    public struct Meta: Decodable {
        public let clip: String
        public let stem: String
        public let corners: [[Double]]
        public let homography: [Double]?   // what Python's cv2.getPerspectiveTransform returned
        public let fps: Double
        public let stride: Int
        public let camera: CameraJSON?
        public let inputs: [String: ArrayLocation]
        public let reference: [String: ArrayLocation]
    }

    public let directory: URL
    public let meta: Meta
    private let inputBytes: Data
    private let referenceBytes: Data

    public init(directory: URL) throws {
        self.directory = directory
        meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: directory.appendingPathComponent("meta.json")))
        inputBytes = try Data(contentsOf: directory.appendingPathComponent("inputs.bin"), options: .alwaysMapped)
        referenceBytes = try Data(contentsOf: directory.appendingPathComponent("reference.bin"), options: .alwaysMapped)
    }

    // MARK: Inputs

    public var geometry: Geometry {
        let homography = Homography(courtCorners: meta.corners.map { (x: $0[0], y: $0[1]) })
        guard let camera = meta.camera else { return Geometry(referenceHomography: homography) }
        let track = CameraTrack(frames: camera.frames, warps: camera.warps.map(Homography.init))
        return Geometry(referenceHomography: homography, cameraTrack: track)
    }

    public func poseDetections(_ pass: String) -> PoseDetections {
        let confidences = float32Array(inputBytes, meta.inputs["\(pass).confidences"]!)
        let shape = meta.inputs["\(pass).confidences"]!.shape
        return PoseDetections(samples: shape[0], people: shape[1],
                              keypoints: float32Array(inputBytes, meta.inputs["\(pass).keypoints"]!),
                              boxes: float32Array(inputBytes, meta.inputs["\(pass).boxes"]!),
                              confidences: confidences, fps: meta.fps, stride: meta.stride)
    }

    // MARK: Reference outputs

    public func referenceFloat32(_ name: String) -> [Float] { float32Array(referenceBytes, meta.reference[name]!) }

    public func referenceFloat64(_ name: String) -> [Double] {
        let location = meta.reference[name]!
        return referenceBytes.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            (0..<location.count).map { raw.loadUnaligned(fromByteOffset: location.offset + $0 * 8, as: Double.self) }
        }
    }

    public func referenceBool(_ name: String) -> [Bool] {
        let location = meta.reference[name]!
        // The explicit raw-buffer type matters: without it, `raw[i] != 0`
        // type-checks against Data's deprecated TYPED-pointer overload, which
        // indexes in multi-byte elements and silently reads the wrong bytes.
        return referenceBytes.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            (0..<location.count).map { raw[location.offset + $0] != 0 }
        }
    }

    public func referenceJSON() throws -> [String: Any] {
        let data = try Data(contentsOf: directory.appendingPathComponent("reference.json"))
        return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    private func float32Array(_ bytes: Data, _ location: ArrayLocation) -> [Float] {
        bytes.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            (0..<location.count).map { raw.loadUnaligned(fromByteOffset: location.offset + $0 * 4, as: Float.self) }
        }
    }
}
