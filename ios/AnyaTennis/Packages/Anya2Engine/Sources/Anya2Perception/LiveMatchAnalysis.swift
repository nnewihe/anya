// LiveMatchAnalysis.swift
//
// Everything that happens to one pose sample while recording, in order:
//   perception (Perceiver)  ->  camera track (LiveCameraTrack, every 3rd
//   sample)  ->  stages 5-8 (LiveAnalysis)
//
// The two clocks are joined here. A pose sample may enter the tracker only
// once the camera track covers its frame with a FINAL warp: the tracker
// projects each player onto the court with the warp NEAREST that frame, and
// until a later camera sample exists the nearest one is not yet known. So
// samples queue here for ~0.6 s (two camera samples of smoothing) first.
//
// At Stop, `finish()` completes the camera track, releases the queue and
// returns exactly what the batch pipeline gives for the same samples.
import Anya2Engine
import Foundation

public final class LiveMatchAnalysis {
    public let poseStride: Int
    public let camera: LiveCameraTrack
    public let geometry: Geometry
    public let analysis: LiveAnalysis
    private var waiting: [(sourceFrame: Int, sample: PooledSample)] = []
    private var nextSourceFrame = 0
    private var warpsInGeometry = 0      // committed camera samples already handed to Geometry
    private var finished: BatchResult?

    /// `sourceFPS`: the recording's frame rate. Pose samples are every
    /// `poseStride`-th source frame, camera samples every `sampleStep`-th.
    public init(reference: CameraReference, referenceHomography: Homography, sourceFPS: Double,
                walkingModel: WalkingModel, reelConfig: ReelConfig = ReelConfig()) {
        let pose = PerceiveRules.stride(sourceFPS: sourceFPS)
        poseStride = pose.stride
        camera = LiveCameraTrack(reference: reference, sampleStep: CameraRules.sampleStep(sourceFPS: sourceFPS))
        geometry = Geometry(referenceHomography: referenceHomography)
        analysis = LiveAnalysis(fps: pose.fps, stride: pose.stride, geometry: geometry, walkingModel: walkingModel,
                                reelConfig: reelConfig)
    }

    /// The source frame the next pose sample must come from.
    public var expectedSourceFrame: Int { nextSourceFrame }

    /// Add the next pose sample. `cameraFrame`: its 960x540 greyscale frame,
    /// needed when the frame is also a camera sample (`camera.isSampleFrame`).
    public func add(_ perceived: PerceivedSample, sourceFrame: Int, cameraFrame: GrayFrame?) {
        precondition(sourceFrame == nextSourceFrame, "pose samples must arrive in order, without gaps")
        nextSourceFrame += poseStride
        if camera.isSampleFrame(sourceFrame) {
            if let cameraFrame { camera.add(cameraFrame, sourceFrame: sourceFrame) }
            else { camera.add(warp: nil, sourceFrame: sourceFrame) }   // unreadable frame: a failed sample
        }
        waiting.append((sourceFrame, perceived.pooled))
        release(finished: false)
    }

    /// Replay one logged sample after a restart: the camera's registration
    /// result is replayed rather than recomputed.
    public func replay(_ perceived: PerceivedSample, sourceFrame: Int, cameraWarp: Homography??) {
        precondition(sourceFrame == nextSourceFrame)
        nextSourceFrame += poseStride
        if let cameraWarp { camera.add(warp: cameraWarp, sourceFrame: sourceFrame) }
        waiting.append((sourceFrame, perceived.pooled))
        release(finished: false)
    }

    /// The whole match. Safe to call again (a retried export): the first
    /// result is kept.
    public func finish() -> BatchResult {
        if let finished { return finished }
        geometry.replaceCameraTrack(camera.finish())
        release(finished: true)
        let result = analysis.finish()
        finished = result
        return result
    }

    /// Hand the tracker every waiting sample the camera track now covers.
    private func release(finished: Bool) {
        if !finished {
            guard let covered = camera.committedThroughFrame else { return }
            let ready = waiting.prefix { $0.sourceFrame <= covered }
            guard !ready.isEmpty else { return }
            // Hand Geometry only the newly final warps (copying the whole
            // track each time would be quadratic over a 2-hour match).
            let committed = camera.committed
            if warpsInGeometry == 0 {
                geometry.replaceCameraTrack(CameraTrack(frames: Array(camera.committedFrames), warps: committed))
            } else if committed.count > warpsInGeometry {
                geometry.extendCameraTrack(frames: camera.committedFrames[warpsInGeometry...],
                                           warps: committed[warpsInGeometry...])
            }
            warpsInGeometry = committed.count
            analysis.append(ready.map(\.sample))
            waiting.removeFirst(ready.count)
        } else {
            analysis.append(waiting.map(\.sample))
            waiting.removeAll()
        }
    }
}
