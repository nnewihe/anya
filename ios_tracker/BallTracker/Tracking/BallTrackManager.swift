import Foundation

/// A pixel-centre detection plus its YOLO confidence.
struct TrackerDetection {
    let x: Double
    let y: Double
    let conf: Double
}

typealias PerspectiveScale = (Double) -> Double

private let noPerspective: PerspectiveScale = { _ in 1.0 }

/// Cheap perspective model needing only the analysis-frame height. Multiplier
/// in (farFloor, 1.0]: ~1.0 near the bottom, shrinking toward farFloor at top.
func makeImageRowPerspective(frameHeight: Double, farFloor: Double = 0.35) -> PerspectiveScale {
    let h = max(1.0, frameHeight)
    return { y in max(farFloor, min(1.0, y / h)) }
}

enum TrackState: String {
    case none, fading, stopped, moving, coasting, lost
}

/// Per-frame answer handed back to the caller.
/// Port of mobile/lib/engine/ball_tracker.dart TrackStatus.
struct TrackStatus {
    let hasMovingTrace: Bool
    let state: TrackState
    let position: (x: Double, y: Double)?
    let speedPxS: Double
    let timeSinceDetection: Double
    let coasting: Bool
    let ballCount: Int
    let maneuverProb: Double
    let racketProb: Double
    let bounceProb: Double
    let trace: [(x: Double, y: Double)]
}

private final class ConfirmedTrack {
    let fps: Double
    let dt: Double
    let motionWindowS: Double
    let corroborationWindowS: Double
    let persp: PerspectiveScale

    var imm: IMMEstimator
    var baseQ: [Mat]
    var lastDetectionT: Double
    var hits = 1
    var lastMeasuredPos: (Double, Double)
    var history: [(t: Double, x: Double, y: Double)] = []
    var detTimes: [Double] = []

    init(fps: Double, x: Double, y: Double, vx: Double, vy: Double, t: Double,
         motionWindowS: Double, corroborationWindowS: Double,
         qSmooth: Double = 5.0, qRacket: Double = 300.0, qPos: Double = 1.0,
         qBounceVx: Double = 20.0, qBounceVy: Double = 300.0,
         muInit: [Double]? = nil, m: [[Double]]? = nil,
         perspectiveScale: PerspectiveScale? = nil) {
        self.fps = fps
        self.dt = 1.0 / max(fps, 1e-6)
        self.motionWindowS = motionWindowS
        self.corroborationWindowS = corroborationWindowS
        self.persp = perspectiveScale ?? noPerspective
        self.lastDetectionT = t
        self.lastMeasuredPos = (x, y)

        let f = Mat(from: [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        let h = Mat(from: [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        let r = Mat.identity(2).scaled(10.0)
        let p0 = Mat.identity(4).scaled(100.0)
        let x0 = Mat.colVec([x, y, vx, vy])

        let qRacketPos = qRacket / 10.0

        func mk(_ q: Mat) -> KalmanFilter {
            KalmanFilter(F: f.clone(), H: h.clone(), R: r.clone(), Q: q,
                         P: p0.clone(), x: x0.clone())
        }

        // Model 0 — smooth in-flight CV.
        let kf0 = mk(Mat.diag([qPos, qPos, qSmooth, qSmooth]))
        // Model 1 — racket impact: isotropic high-Q on position AND velocity.
        let kf1 = mk(Mat.diag([qRacketPos, qRacketPos, qRacket, qRacket]))
        // Model 2 — court bounce: anisotropic Q.
        let kf2 = mk(Mat.diag([qPos, qRacketPos, qBounceVx, qBounceVy]))

        let mu = muInit ?? [0.90, 0.05, 0.05]
        let trans = m ?? [
            [0.92, 0.04, 0.04],
            [0.70, 0.25, 0.05],
            [0.70, 0.05, 0.25],
        ]
        imm = IMMEstimator([kf0, kf1, kf2], mu, trans)
        baseQ = [kf0.Q.clone(), kf1.Q.clone(), kf2.Q.clone()]

        history.append((t, x, y))
        detTimes.append(t)
    }

    func predict() {
        let scale = persp(yPos)
        for i in 0..<imm.filters.count {
            imm.filters[i].Q = baseQ[i].scaled(scale)
        }
        imm.predict()
    }

    func update(x: Double, y: Double, t: Double) {
        imm.update(Mat.colVec([x, y]))
        lastDetectionT = t
        lastMeasuredPos = (x, y)
        hits += 1
        detTimes.append(t)
    }

    var position: (Double, Double) { (imm.x.d[0], imm.x.d[1]) }
    var yPos: Double { imm.x.d[1] }
    func speedPxS() -> Double {
        (imm.x.d[2] * imm.x.d[2] + imm.x.d[3] * imm.x.d[3]).squareRoot()
    }
    func positionUncertainty() -> Double {
        imm.P.at(0, 0) + imm.P.at(1, 1) // trace of P[:2,:2]
    }
    var maneuverProb: Double { 1.0 - imm.mu[0] }
    var racketProb: Double { imm.mu[1] }
    var bounceProb: Double { imm.mu[2] }

    func record(t: Double, now: Double) {
        let p = position
        history.append((t, p.0, p.1))
        let cutoff = now - motionWindowS
        while let first = history.first, first.t < cutoff {
            history.removeFirst()
        }
        let detCutoff = now - corroborationWindowS
        while let first = detTimes.first, first < detCutoff {
            detTimes.removeFirst()
        }
    }

    func recentDetCount() -> Int { detTimes.count }

    func recentSpanPx() -> Double {
        guard history.count >= 2, let last = history.last else { return 0.0 }
        var maxD = 0.0
        for h in history {
            let d = ((h.x - last.x) * (h.x - last.x) + (h.y - last.y) * (h.y - last.y)).squareRoot()
            if d > maxD { maxD = d }
        }
        return maxD
    }

    func trace() -> [(x: Double, y: Double)] { history.map { ($0.x, $0.y) } }
    func traceWithTime() -> [(t: Double, x: Double, y: Double)] { history }
}

private final class Tentative {
    var points: [(t: Double, x: Double, y: Double)] = []
    var lastT: Double

    init(x: Double, y: Double, t: Double) {
        lastT = t
        points.append((t, x, y))
    }

    func add(x: Double, y: Double, t: Double) {
        points.append((t, x, y))
        lastT = t
    }

    var lastXy: (Double, Double) { (points[points.count - 1].x, points[points.count - 1].y) }

    func expectedNext(_ t: Double) -> (Double, Double) {
        let lx = points[points.count - 1].x
        let ly = points[points.count - 1].y
        if points.count < 2 { return (lx, ly) }
        let p0 = points[points.count - 2]
        let p1 = points[points.count - 1]
        let segDt = p1.t - p0.t
        if segDt <= 0 { return (lx, ly) }
        let vx = (p1.x - p0.x) / segDt
        let vy = (p1.y - p0.y) / segDt
        let dt = t - p1.t
        return (lx + vx * dt, ly + vy * dt)
    }

    func spanPx() -> Double {
        guard points.count >= 2, let last = points.last else { return 0.0 }
        var maxD = 0.0
        for p in points {
            let d = ((p.x - last.x) * (p.x - last.x) + (p.y - last.y) * (p.y - last.y)).squareRoot()
            if d > maxD { maxD = d }
        }
        return maxD
    }

    func velocity() -> (Double, Double) {
        guard points.count >= 2, let p0 = points.first, let p1 = points.last else {
            return (0.0, 0.0)
        }
        let dt = p1.t - p0.t
        if dt <= 0 { return (0.0, 0.0) }
        return ((p1.x - p0.x) / dt, (p1.y - p0.y) / dt)
    }
}

/// Maintains a single confirmed ball trajectory and reports whether a moving
/// trace is currently alive. Port of mobile/lib/engine/ball_tracker.dart
/// BallTrackManager (which passes all 10 Python self-test scenarios).
final class BallTrackManager {
    let fps: Double
    let dt: Double
    let persp: PerspectiveScale
    let gateBasePx: Double
    let gateUncertaintyK: Double
    let seedGatePx: Double
    let seedCoherencePx: Double
    let confirmHits: Int
    let confirmWindowS: Double
    let missTimeoutS: Double
    let motionWindowS: Double
    let moveThreshPx: Double
    let minRecentDets: Int
    let corroborationWindowS: Double
    let hijackAfterS: Double
    let qSmooth: Double
    let qManeuver: Double
    let qPos: Double
    let qBounceVx: Double
    let qBounceVy: Double
    let fallbackGateK: Double
    let coastGateK: Double
    let coastGateCapPx: Double

    private var track: ConfirmedTrack?
    private var tentatives: [Tentative] = []
    private(set) var lastDetectionTime: Double?
    private var lastTrace: [(t: Double, x: Double, y: Double)] = []

    init(fps: Double,
         perspectiveScale: PerspectiveScale? = nil,
         gateBasePx: Double = 50.0,
         gateUncertaintyK: Double = 0.6,
         seedGatePx: Double = 100.0,
         seedCoherencePx: Double = 38.0,
         confirmHits: Int = 3,
         confirmWindowS: Double = 0.6,
         missTimeoutS: Double = 2.0,
         motionWindowS: Double = 0.5,
         moveThreshPx: Double = 30.0,
         minRecentDets: Int = 3,
         corroborationWindowS: Double = 2.0,
         hijackAfterS: Double = 0.15,
         qSmooth: Double = 5.0,
         qManeuver: Double = 300.0,
         qPos: Double = 1.0,
         qBounceVx: Double = 20.0,
         qBounceVy: Double = 300.0,
         fallbackGateK: Double = 1.8,
         coastGateK: Double = 0.5,
         coastGateCapPx: Double = 400.0) {
        self.fps = fps
        self.dt = 1.0 / max(fps, 1e-6)
        self.persp = perspectiveScale ?? noPerspective
        self.gateBasePx = gateBasePx
        self.gateUncertaintyK = gateUncertaintyK
        self.seedGatePx = seedGatePx
        self.seedCoherencePx = seedCoherencePx
        self.confirmHits = confirmHits
        self.confirmWindowS = confirmWindowS
        self.missTimeoutS = missTimeoutS
        self.motionWindowS = motionWindowS
        self.moveThreshPx = moveThreshPx
        self.minRecentDets = minRecentDets
        self.corroborationWindowS = corroborationWindowS
        self.hijackAfterS = hijackAfterS
        self.qSmooth = qSmooth
        self.qManeuver = qManeuver
        self.qPos = qPos
        self.qBounceVx = qBounceVx
        self.qBounceVy = qBounceVy
        self.fallbackGateK = fallbackGateK
        self.coastGateK = coastGateK
        self.coastGateCapPx = coastGateCapPx
    }

    func reset() {
        track = nil
        tentatives = []
        lastDetectionTime = nil
        lastTrace = []
    }

    func update(detections: [TrackerDetection], now: Double) -> TrackStatus {
        // 1. Predict the confirmed track forward.
        track?.predict()

        // 2. Associate one detection to the confirmed track.
        var used = [Bool](repeating: false, count: detections.count)
        if let t = track, !detections.isEmpty {
            let (tx, ty) = t.position
            let scale = persp(t.yPos)
            var gate = gateBasePx * scale +
                gateUncertaintyK * max(t.positionUncertainty(), 0.0).squareRoot()
            let tsd = now - t.lastDetectionT
            gate += min(coastGateK * t.speedPxS() * tsd, coastGateCapPx)
            var bestI = -1
            var bestD = gate
            for i in 0..<detections.count {
                let d = ((detections[i].x - tx) * (detections[i].x - tx) +
                         (detections[i].y - ty) * (detections[i].y - ty)).squareRoot()
                if d <= bestD { bestD = d; bestI = i }
            }

            // Fallback gate around the last measured position.
            if bestI < 0 {
                let (lmx, lmy) = t.lastMeasuredPos
                let fbGate = gateBasePx * fallbackGateK * scale
                var fbBestI = -1
                var fbBestD = fbGate
                for i in 0..<detections.count {
                    let d = ((detections[i].x - lmx) * (detections[i].x - lmx) +
                             (detections[i].y - lmy) * (detections[i].y - lmy)).squareRoot()
                    if d <= fbBestD { fbBestD = d; fbBestI = i }
                }
                bestI = fbBestI
            }

            if bestI >= 0 {
                t.update(x: detections[bestI].x, y: detections[bestI].y, t: now)
                used[bestI] = true
                lastDetectionTime = now
            }
        }

        // 3. Feed leftovers to tentative seeds; try to promote one.
        for i in 0..<detections.count where !used[i] {
            feedTentative(x: detections[i].x, y: detections[i].y, now: now)
        }
        pruneTentatives(now)
        tryPromote(now)

        // 4. Record position into the motion window.
        track?.record(t: now, now: now)

        let status = makeStatus(ballCount: detections.count, now: now)
        if status.state == .lost { track = nil }
        return status
    }

    private func feedTentative(x: Double, y: Double, now: Double) {
        let scale = persp(y)
        var best: Tentative?
        var bestD = Double.infinity
        for tnt in tentatives {
            let e = tnt.expectedNext(now)
            let gate = (tnt.points.count < 2 ? seedGatePx : seedCoherencePx) * scale
            let d = ((x - e.0) * (x - e.0) + (y - e.1) * (y - e.1)).squareRoot()
            if d <= gate && d < bestD { bestD = d; best = tnt }
        }
        if let best {
            best.add(x: x, y: y, t: now)
        } else {
            tentatives.append(Tentative(x: x, y: y, t: now))
        }
    }

    private func pruneTentatives(_ now: Double) {
        tentatives = tentatives.filter { now - $0.points[0].t <= confirmWindowS }
    }

    private func tryPromote(_ now: Double) {
        // The hijack guard protects a *healthy* track from being stolen by a
        // seed. A track sitting on a stationary object (a ball basket, a ball
        // on the ground) is fed a detection every frame, so its lastDetectionT
        // never goes stale and the guard would block promotion forever — the
        // real ball could never take over. Upstream (pipeline/anya_base.py)
        // never hits this because exclusion zones drop stationary ball-like
        // objects before they reach the tracker; this port has no such filter,
        // so only a moving track gets the protection. Promotable seeds must
        // already be coherent moving trajectories, so clutter still can't win.
        if let t = track, now - t.lastDetectionT <= hijackAfterS,
           t.recentSpanPx() > moveThreshPx * persp(t.yPos) { return }
        let promotable = tentatives.filter {
            $0.points.count >= confirmHits &&
            $0.spanPx() > moveThreshPx * persp($0.lastXy.1)
        }
        guard !promotable.isEmpty else { return }
        let sorted = promotable.sorted { a, b in
            if a.points.count != b.points.count {
                return a.points.count < b.points.count
            }
            return a.spanPx() < b.spanPx()
        }
        let best = sorted[sorted.count - 1] // max (len, span)
        let xy = best.lastXy
        let v = best.velocity()
        let tLast = best.points[best.points.count - 1].t
        let nt = ConfirmedTrack(
            fps: fps, x: xy.0, y: xy.1, vx: v.0, vy: v.1, t: tLast,
            motionWindowS: motionWindowS, corroborationWindowS: corroborationWindowS,
            qSmooth: qSmooth, qRacket: qManeuver, qPos: qPos,
            qBounceVx: qBounceVx, qBounceVy: qBounceVy,
            perspectiveScale: persp)
        // Backfill motion window + corroboration from the seed.
        nt.history.removeAll()
        nt.detTimes.removeAll()
        for p in best.points {
            nt.history.append((p.t, p.x, p.y))
            nt.detTimes.append(p.t)
        }
        track = nt
        lastDetectionTime = tLast
        tentatives.removeAll { $0 === best }
    }

    private func makeStatus(ballCount: Int, now: Double) -> TrackStatus {
        guard let t = track else {
            lastTrace = []
            return TrackStatus(
                hasMovingTrace: false,
                state: .none,
                position: nil,
                speedPxS: 0.0,
                timeSinceDetection: 0.0,
                coasting: false,
                ballCount: ballCount,
                maneuverProb: 0.0,
                racketProb: 0.0,
                bounceProb: 0.0,
                trace: [])
        }

        let tsd = now - t.lastDetectionT
        let lost = tsd > missTimeoutS
        let coasting = !lost && tsd > (1.5 * dt)
        let moving = t.recentSpanPx() > moveThreshPx * persp(t.yPos)
        let corroborated = t.recentDetCount() >= minRecentDets

        lastTrace = t.traceWithTime()

        let state: TrackState
        let alive: Bool
        if lost {
            state = .lost
            alive = false
        } else if !corroborated {
            state = .fading
            alive = false
        } else if !moving {
            state = .stopped
            alive = false
        } else {
            state = coasting ? .coasting : .moving
            alive = true
        }

        return TrackStatus(
            hasMovingTrace: alive,
            state: state,
            position: t.position,
            speedPxS: t.speedPxS(),
            timeSinceDetection: tsd,
            coasting: coasting,
            ballCount: ballCount,
            maneuverProb: t.maneuverProb,
            racketProb: t.racketProb,
            bounceProb: t.bounceProb,
            trace: t.trace())
    }

    func tracePoints() -> [(t: Double, x: Double, y: Double)] { lastTrace }
}
