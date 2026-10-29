// FeatureColumns.swift
//
// Lets the live pipeline build a detector's per-sample features a chunk at a
// time: start empty, then append the rows of each window it computes. Each
// feature struct lists its columns once, here.
import Foundation

protocol PerSampleColumns {
    static var numberColumns: [WritableKeyPath<Self, [Double]>] { get }
    static var flagColumns: [WritableKeyPath<Self, [Bool]>] { get }
    static func empty(fps: Double) -> Self
}

extension PerSampleColumns {
    /// Append rows `rows` of `other` (a window's features) to these columns.
    mutating func appendRows(of other: Self, _ rows: Range<Int>) {
        for column in Self.numberColumns { self[keyPath: column] += other[keyPath: column][rows] }
        for column in Self.flagColumns { self[keyPath: column] += other[keyPath: column][rows] }
    }
}

extension NearServeFeatures: PerSampleColumns {
    static let numberColumns: [WritableKeyPath<Self, [Double]>] = [
        \.ready, \.trophy, \.still, \.leftWristAboveShoulders, \.rightWristAboveShoulders,
        \.leftWristAboveHead, \.rightWristAboveHead, \.higherWristAboveShoulders,
        \.lowerWristAboveShoulders, \.higherWristAboveHead, \.higherWristSide, \.wristGap,
    ]
    static let flagColumns: [WritableKeyPath<Self, [Bool]>] = [\.valid, \.inServeZone]

    static func empty(fps: Double) -> Self {
        NearServeFeatures(fps: fps, valid: [], ready: [], trophy: [], leftWristAboveShoulders: [],
                          rightWristAboveShoulders: [], leftWristAboveHead: [], rightWristAboveHead: [],
                          higherWristAboveShoulders: [], lowerWristAboveShoulders: [], higherWristAboveHead: [],
                          higherWristSide: [], wristGap: [], still: [], inServeZone: [])
    }
}

extension FarServeFeatures: PerSampleColumns {
    static let numberColumns: [WritableKeyPath<Self, [Double]>] = [
        \.ready, \.trophy, \.still, \.boxTravel, \.leftWristAboveShoulders, \.rightWristAboveShoulders,
        \.leftWristAboveHead, \.rightWristAboveHead, \.higherWristAboveShoulders,
        \.lowerWristAboveShoulders, \.higherWristAboveHead, \.higherWristSide,
        \.leftWristSideways, \.rightWristSideways,
    ]
    static let flagColumns: [WritableKeyPath<Self, [Bool]>] = [\.valid, \.eligible]

    static func empty(fps: Double) -> Self {
        FarServeFeatures(fps: fps, valid: [], ready: [], trophy: [], leftWristAboveShoulders: [],
                         rightWristAboveShoulders: [], leftWristAboveHead: [], rightWristAboveHead: [],
                         higherWristAboveShoulders: [], lowerWristAboveShoulders: [], higherWristAboveHead: [],
                         higherWristSide: [], leftWristSideways: [], rightWristSideways: [], still: [],
                         boxTravel: [], eligible: [])
    }
}
