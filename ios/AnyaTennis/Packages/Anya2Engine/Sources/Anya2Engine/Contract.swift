// Contract.swift  --  port of pipeline/anya2/contract.py
//
// The one thing a detector emits: an Event. Detectors never exchange anything
// else, so each can be tuned without the others knowing.
import Foundation

public enum EventKind: String, Codable {
    case nearServe = "near_serve"
    case farServe = "far_serve"
    case pointEnd = "point_end"
}

/// One detected point boundary.
/// `t` is always SECONDS ON THE SOURCE VIDEO's timeline -- never a pose-sample
/// index or a proxy frame number.
public struct Event: Codable, Equatable {
    public var t: Double
    public var p: Double                 // confidence, 0..1
    public var kind: EventKind
    public var track: Int?               // player slot (0, 1 near; 2, 3 far), if any
    public var detail: [String: Double]  // diagnostics for humans; far serves also carry "toss"

    public init(t: Double, p: Double, kind: EventKind, track: Int? = nil,
                detail: [String: Double] = [:]) {
        self.t = t; self.p = p; self.kind = kind; self.track = track; self.detail = detail
    }
}
