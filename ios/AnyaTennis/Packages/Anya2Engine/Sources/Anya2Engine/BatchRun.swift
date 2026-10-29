// BatchRun.swift  --  stages 5-8 of pipeline/anya2/run.py, over a whole clip
//
//   5 tracks       <=2 near + <=2 far player slots        (Tracks.swift)
//   6 end signals  walking probability + near_end         (EndSignals.swift)
//   7 detectors    near serves, far serves, point ends    (NearServe, FarServe, PointEnd)
//   8 assemble     serves + ends -> segments              (Orchestrator.swift)
//
// This is the reference the live (incremental) pipeline must reproduce
// exactly. Stages 1-4 (calibration, camera track, the two pose passes) feed it.
import Foundation

public struct BatchResult {
    public let tracks: PlayerTracks
    public let endSignals: EndSignals
    public let nearServes: [Event]
    public let farServes: [Event]
    public let pointEnds: [Event]
    public let live: [Double]   // the normalised live score the orchestrator used
    public let reel: Reel
}

public func runBatch(near: PoseDetections, far: PoseDetections?, geometry: Geometry,
                     walkingModel: WalkingModel, reelConfig: ReelConfig = ReelConfig()) -> BatchResult {
    let tracks = buildTracks(near: near, far: far, geometry: geometry)                        // 5
    // The walking classifier reads the FIXED calibration homography, never the camera track.
    let endSignals = computeEndSignals(tracks: tracks, fixedHomography: geometry.referenceHomography,
                                       walkingModel: walkingModel)                            // 6
    let nearServes = detectNearServes(tracks: tracks)                                          // 7
    let farServes = detectFarServes(tracks: tracks)
    let union = nonRallyUnion(endSignals)
    let activity = playerActivity(tracks).map { Array($0.prefix(union.count)) }
    let live = normaliseLiveScore(smoothedLiveScore(activity: activity, union: union, fps: tracks.fps,
                                                    smoothSeconds: reelConfig.live_smooth_s))
    let pointEnds = detectPointEnds(live: live, fps: tracks.fps)
    let reel = buildReel(near: nearServes, far: farServes, ends: pointEnds, live: live,
                         liveFPS: tracks.fps, tracks: tracks, config: reelConfig)            // 8
    return BatchResult(tracks: tracks, endSignals: endSignals, nearServes: nearServes, farServes: farServes,
                       pointEnds: pointEnds, live: live, reel: reel)
}
