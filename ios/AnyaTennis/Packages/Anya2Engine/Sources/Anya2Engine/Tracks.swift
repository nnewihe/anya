// Tracks.swift  --  port of pipeline/anya2/tracks.py
//
// Turn the two all-persons pose passes into at most FOUR player slots:
//     slot 0, 1   near side
//     slot 2, 3   far side
// Singles is just the case where slots 1 and 3 stay empty (NaN).
//
// Per pose sample, every detected person is projected onto the court and:
//   1. FILTERED     too small, outside the loose tracking zone, or too tall for
//                   where they claim to stand -> dropped      (candidates)
//   2. SIDED        near or far, by which baseline their feet are closer to
//   3. SHORTLISTED  at most two per side, chosen by how much they belong there:
//                   near their own baseline, confident, continuous  (shortlist)
//   4. ASSIGNED     slots already following someone claim first, by continuity;
//                   free slots take the rest, best claim first      (assignSide)
//
// A candidate that would need to move faster than MAX_SPEED from a slot's
// last position never gets that slot: a missing frame is honest, a wrong one
// poisons everything after it.
//
// TRACKED vs ELIGIBLE: the tracking zone is deliberately loose, so a player
// walking to the ball carts keeps their slot. Separately, `eligible` records
// the user's strict rule (box x-centre inside the doubles court + 3 ft);
// detectors use it to decide who may be serving.
//
// Evidence and history: docs/anya2/NOTES.md, "tracks.py".
import Foundation

public enum TrackRules {
    public static let MIN_H_PX = 12.0           // smaller boxes are spectators on other courts / noise
    public static let X_TRACK_PAD = 6.0         // metres outside the singles lines a player may wander
    public static let NEAR_BACK_M = 6.0         // ...and behind the near baseline
    public static let FAR_BACK_M = 12.0         // ...and behind the far one (far projections run deep)
    public static let HEIGHT_MAX_RATIO = 1.40   // observed / expected box height; normal p90 is 1.24
    public static let PLAYER_HEIGHT_M = 1.75
    public static let LOST_FRAMES = 60          // samples a slot keeps following an unseen player
    public static let MAX_SPEED = 9.0           // m/s; faster than a sprint means a different person
    public static let W_CONT = 2.0              // continuity bonus: W_CONT * exp(-distance / CONT_SCALE)
    public static let CONT_SCALE = 2.0          // metres
    public static let W_CONF = 0.6              // weight on detection confidence
    public static let W_HOME = 1.0              // weight on closeness to the player's OWN baseline
}

public enum Slots {
    public static let count = 4
    public static let near = [0, 1]
    public static let far = [2, 3]
}

/// One pose pass, as perceive writes it: up to `people` persons per sample,
/// NaN-padded, in analysis-frame pixels (960 x 540). Stored as float32, as in
/// Python's npz files.
public struct PoseDetections {
    public let samples: Int
    public let people: Int
    public var keypoints: [Float]      // [samples, people, 17, 3]
    public var boxes: [Float]          // [samples, people, 4]
    public var confidences: [Float]    // [samples, people]
    public let fps: Double
    public let stride: Int             // source frames per pose sample

    public init(samples: Int, people: Int, keypoints: [Float], boxes: [Float],
                confidences: [Float], fps: Double, stride: Int) {
        self.samples = samples; self.people = people; self.keypoints = keypoints
        self.boxes = boxes; self.confidences = confidences; self.fps = fps; self.stride = stride
    }
}

/// The four player slots over time (`<stem>_anya2_tracks.npz`). float32, as in
/// Python: the detectors read these back and the rounding is part of the result.
public struct PlayerTracks {
    public let samples: Int
    public let fps: Double
    public let stride: Int
    public var keypoints: [Float]      // [samples, 4, 17, 3]
    public var boxes: [Float]          // [samples, 4, 4]
    public var court: [Float]          // [samples, 4, 2] ground point, court metres
    public var eligible: [Bool]        // [samples, 4]

    public init(samples: Int, fps: Double, stride: Int, keypoints: [Float], boxes: [Float],
                court: [Float], eligible: [Bool]) {
        self.samples = samples; self.fps = fps; self.stride = stride; self.keypoints = keypoints
        self.boxes = boxes; self.court = court; self.eligible = eligible
    }

    public static func empty(samples: Int, fps: Double, stride: Int) -> PlayerTracks {
        PlayerTracks(samples: samples, fps: fps, stride: stride,
                     keypoints: [Float](repeating: .nan, count: samples * Slots.count * 51),
                     boxes: [Float](repeating: .nan, count: samples * Slots.count * 4),
                     court: [Float](repeating: .nan, count: samples * Slots.count * 2),
                     eligible: [Bool](repeating: false, count: samples * Slots.count))
    }

    public var durationSeconds: Double { Double(samples) / fps }

    /// One slot's keypoints, 51 columns per sample, widened to Double.
    public func keypoints(slot: Int) -> SampleTable {
        var values = [Double](repeating: .nan, count: samples * 51)
        for i in 0..<samples {
            let source = (i * Slots.count + slot) * 51
            for c in 0..<51 { values[i * 51 + c] = Double(keypoints[source + c]) }
        }
        return SampleTable(rows: samples, columns: 51, values: values)
    }

    /// One slot's boxes, 4 columns per sample, widened to Double.
    public func boxes(slot: Int) -> SampleTable {
        var values = [Double](repeating: .nan, count: samples * 4)
        for i in 0..<samples {
            for c in 0..<4 { values[i * 4 + c] = Double(boxes[(i * Slots.count + slot) * 4 + c]) }
        }
        return SampleTable(rows: samples, columns: 4, values: values)
    }

    /// One slot's court x (axis 0) or y (axis 1) over time.
    public func court(slot: Int, axis: Int) -> [Double] {
        (0..<samples).map { Double(court[($0 * Slots.count + slot) * 2 + axis]) }
    }

    public func eligible(slot: Int) -> [Bool] {
        (0..<samples).map { eligible[$0 * Slots.count + slot] }
    }

    /// Whether the slot ever held anyone.
    public func wasTracked(slot: Int) -> Bool {
        (0..<samples).contains { boxes[($0 * Slots.count + slot) * 4].isFinite }
    }
}

/// One player slot: where it last saw its player, and how long ago.
struct PlayerSlot {
    var court: (x: Double, y: Double)? = nil   // last position, court metres
    var since = 1_000_000_000                  // samples since last seen

    /// Following someone recently enough to use continuity.
    var anchored: Bool { court != nil && since <= TrackRules.LOST_FRAMES }

    mutating func miss() { since += 1 }   // the last position is KEPT as a re-acquisition hint
    mutating func hit(_ position: (x: Double, y: Double)) { court = position; since = 1 }
}

private struct Candidate {
    let person: Int
    let court: (x: Double, y: Double)
    let confidence: Double
    let heightPx: Double
    let homeness: Double
}

private enum Side { case near, far }

/// Build the four player slots from the near and far pose passes (tracks.build).
public func buildTracks(near: PoseDetections, far: PoseDetections?, geometry: Geometry) -> PlayerTracks {
    let merged = mergePasses(near: near, far: far)
    let samples = merged.samples, people = merged.people, fps = merged.fps
    var out = PlayerTracks.empty(samples: samples, fps: fps, stride: merged.stride)
    var slotsBySide: [Side: [PlayerSlot]] = [.near: [PlayerSlot(), PlayerSlot()],
                                              .far: [PlayerSlot(), PlayerSlot()]]
    let slotIDs: [Side: [Int]] = [.near: Slots.near, .far: Slots.far]

    for sample in 0..<samples {
        let sourceFrame = sample * merged.stride
        let bySide = candidates(at: sample, in: merged,
                                homography: geometry.homography(atSourceFrame: sourceFrame),
                                inverse: geometry.inverseHomography(atSourceFrame: sourceFrame))
        for side in [Side.near, .far] {
            var slots = slotsBySide[side]!
            let shortlisted = shortlist(bySide[side]!, slots: slots, fps: fps)
            let chosen = shortlisted.isEmpty ? [:] : assignSide(shortlisted, slots: slots, fps: fps)
            for localIndex in 0..<slots.count {
                let slotID = slotIDs[side]![localIndex]
                guard let candidateIndex = chosen[localIndex] else {
                    slots[localIndex].miss()
                    continue
                }
                let candidate = shortlisted[candidateIndex]
                let source = sample * people + candidate.person
                let target = sample * Slots.count + slotID
                for c in 0..<51 { out.keypoints[target * 51 + c] = merged.keypoints[source * 51 + c] }
                for c in 0..<4 { out.boxes[target * 4 + c] = merged.boxes[source * 4 + c] }
                out.court[target * 2] = Float(candidate.court.x)
                out.court[target * 2 + 1] = Float(candidate.court.y)
                out.eligible[target] = Court.inBounds(courtX: candidate.court.x)
                slots[localIndex].hit(candidate.court)
            }
            slotsBySide[side] = slots
        }
    }
    return out
}

/// Pool the near and far passes into one person list per sample. Side is
/// decided by the COURT, not by which pass found a person, so both are pooled.
/// They must be on one timeline; a mismatch means one is stale.
private func mergePasses(near: PoseDetections, far: PoseDetections?) -> PoseDetections {
    guard let far else { return near }
    precondition(abs(near.samples - far.samples) <= 2 && abs(near.fps - far.fps) <= 0.01,
                 "near and far pose passes are not on the same timeline")
    let samples = min(near.samples, far.samples), people = near.people + far.people
    var keypoints = [Float](), boxes = [Float](), confidences = [Float]()
    keypoints.reserveCapacity(samples * people * 51)
    for sample in 0..<samples {
        for (pass, count) in [(near, near.people), (far, far.people)] {
            let first = sample * count
            keypoints += pass.keypoints[(first * 51)..<((first + count) * 51)]
            boxes += pass.boxes[(first * 4)..<((first + count) * 4)]
            confidences += pass.confidences[first..<(first + count)]
        }
    }
    return PoseDetections(samples: samples, people: people, keypoints: keypoints, boxes: boxes,
                          confidences: confidences, fps: near.fps, stride: near.stride)
}

/// Every person in this sample who may be tracked, split by side.
private func candidates(at sample: Int, in passes: PoseDetections,
                        homography: Homography, inverse: Homography) -> [Side: [Candidate]] {
    var bySide: [Side: [Candidate]] = [.near: [], .far: []]
    for person in 0..<passes.people {
        let index = sample * passes.people + person
        let confidence = Double(passes.confidences[index])
        guard confidence.isFinite else { continue }
        let b = index * 4
        let x1 = Double(passes.boxes[b]), y1 = Double(passes.boxes[b + 1])
        let x2 = Double(passes.boxes[b + 2]), y2 = Double(passes.boxes[b + 3])
        // float32 subtraction, as Python's `float(box[3] - box[1])` on a float32 row.
        let heightPx = Double(passes.boxes[b + 3] - passes.boxes[b + 1])
        guard heightPx.isFinite && heightPx >= TrackRules.MIN_H_PX else { continue }
        let position = Court.project(homography, box: (x1, y1, x2, y2))
        let side: Side = Court.isNear(courtY: position.y) ? .near : .far
        guard isTrackable(position, side: side),
              isHeightPlausible(heightPx, courtY: position.y, inverse: inverse) else { continue }
        bySide[side]!.append(Candidate(person: person, court: position, confidence: confidence,
                                       heightPx: heightPx, homeness: homeness(position.y, side: side)))
    }
    return bySide
}

/// Inside the loose tracking zone for this side. NaN -> false.
private func isTrackable(_ p: (x: Double, y: Double), side: Side) -> Bool {
    guard p.x.isFinite && p.y.isFinite else { return false }
    guard -TrackRules.X_TRACK_PAD <= p.x && p.x <= Court.COURT_W + TrackRules.X_TRACK_PAD else { return false }
    switch side {
    case .near: return -TrackRules.NEAR_BACK_M <= p.y && p.y < Court.MID_Y
    case .far: return Court.MID_Y <= p.y && p.y <= Court.COURT_L + TrackRules.FAR_BACK_M
    }
}

/// False only when the box is too TALL for its projected depth. One-sided:
/// small boxes (crouching, partial, clipped) are normal. Unknown depth passes.
private func isHeightPlausible(_ heightPx: Double, courtY: Double, inverse: Homography) -> Bool {
    guard heightPx.isFinite && courtY.isFinite else { return true }
    let expectedPx = TrackRules.PLAYER_HEIGHT_M * pixelsPerMetre(inverse, courtY: courtY)
    if expectedPx < 1 { return true }
    return heightPx <= TrackRules.HEIGHT_MAX_RATIO * expectedPx
}

/// Image px per court metre at depth `courtY`, from the court's width there.
func pixelsPerMetre(_ inverse: Homography, courtY: Double) -> Double {
    let left = inverse.apply(x: 0, y: courtY), right = inverse.apply(x: Court.COURT_W, y: courtY)
    let dx = right.x - left.x, dy = right.y - left.y
    return (dx * dx + dy * dy).squareRoot() / Court.COURT_W
}

/// 1.0 on the side's own baseline (or deeper), 0.0 at mid-court. NaN -> 0.
private func homeness(_ courtY: Double, side: Side) -> Double {
    guard courtY.isFinite else { return 0 }
    switch side {
    case .near: return clip(1 - courtY / Court.MID_Y, 0, 1)
    case .far: return clip((courtY - Court.MID_Y) / (Court.COURT_L - Court.MID_Y), 0, 1)
    }
}

private func distanceMetres(_ a: (x: Double, y: Double), _ b: (x: Double, y: Double)) -> Double {
    let dx = a.x - b.x, dy = a.y - b.y
    return (dx * dx + dy * dy).squareRoot()
}

/// Would reaching this distance from the slot's last position exceed MAX_SPEED?
private func isTooFast(_ distance: Double, slot: PlayerSlot, fps: Double) -> Bool {
    distance / Double(max(slot.since, 1)) * fps > TrackRules.MAX_SPEED
}

/// Keep at most `slots.count` candidates: the ones that most BELONG on this side.
/// belonging = W_HOME * home + W_CONF * conf + best continuity to any anchored slot.
/// Without this, two people who grabbed the slots first keep them forever.
private func shortlist(_ candidates: [Candidate], slots: [PlayerSlot], fps: Double) -> [Candidate] {
    guard candidates.count > slots.count else { return candidates }
    let scored: [(belonging: Double, candidate: Candidate)] = candidates.map { candidate in
        var continuity = 0.0
        for slot in slots where slot.anchored {
            let d = distanceMetres(candidate.court, slot.court!)
            if isTooFast(d, slot: slot, fps: fps) { continue }
            continuity = max(continuity, TrackRules.W_CONT * exp(-d / TrackRules.CONT_SCALE))
        }
        let belonging = TrackRules.W_HOME * candidate.homeness
            + TrackRules.W_CONF * candidate.confidence + continuity
        return (belonging, candidate)
    }
    return stableSorted(scored) { $0.belonging > $1.belonging }.prefix(slots.count).map(\.candidate)
}

/// Assign candidates to this side's slots; returns [slot index: candidate index].
///   1. Anchored slots claim first, closest candidate first (continuity beats a
///      stranger's confidence, so partners don't swap slots).
///   2. Free slots take what is left, strongest claim (conf + home) first; each
///      goes to the free slot whose stale last position is nearest.
private func assignSide(_ candidates: [Candidate], slots: [PlayerSlot], fps: Double) -> [Int: Int] {
    var chosen: [Int: Int] = [:]
    var taken = Set<Int>()

    var continuityPairs: [(score: Double, slot: Int, candidate: Int)] = []
    for (slotIndex, slot) in slots.enumerated() where slot.anchored {
        for (candidateIndex, candidate) in candidates.enumerated() {
            let d = distanceMetres(candidate.court, slot.court!)
            if isTooFast(d, slot: slot, fps: fps) { continue }
            continuityPairs.append((-TrackRules.W_CONT * exp(-d / TrackRules.CONT_SCALE), slotIndex, candidateIndex))
        }
    }
    // Python sorts the (score, slot, candidate) tuples: ties fall to slot, then candidate.
    continuityPairs.sort { ($0.score, $0.slot, $0.candidate) < ($1.score, $1.slot, $1.candidate) }
    for pair in continuityPairs where chosen[pair.slot] == nil && !taken.contains(pair.candidate) {
        chosen[pair.slot] = pair.candidate
        taken.insert(pair.candidate)
    }

    var freeSlots = (0..<slots.count).filter { chosen[$0] == nil }
    guard !freeSlots.isEmpty else { return chosen }
    let claims = candidates.indices
        .filter { !taken.contains($0) }
        .map { (claim: -(TrackRules.W_CONF * candidates[$0].confidence + TrackRules.W_HOME * candidates[$0].homeness),
                candidate: $0) }
        .sorted { ($0.claim, $0.candidate) < ($1.claim, $1.candidate) }
    for claim in claims {
        guard !freeSlots.isEmpty else { break }
        let position = candidates[claim.candidate].court
        let slotIndex = freeSlots.min { a, b in
            let da = slots[a].court.map { distanceMetres(position, $0) } ?? .infinity
            let db = slots[b].court.map { distanceMetres(position, $0) } ?? .infinity
            return (da, a) < (db, b)
        }!
        freeSlots.removeAll { $0 == slotIndex }
        chosen[slotIndex] = claim.candidate
        taken.insert(claim.candidate)
    }
    return chosen
}
