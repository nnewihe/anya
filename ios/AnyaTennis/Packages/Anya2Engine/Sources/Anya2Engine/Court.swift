// Court.swift  --  port of pipeline/anya2/court.py (+ walking/court.py)
//
// Court geometry: the image -> court-metres map, and the two player gates.
//
// Court frame (metres)
//   origin           near-left SINGLES corner
//   +x               along the near baseline
//   +y               away from the camera
//   near baseline    y = 0          (a server stands at y < 0)
//   far baseline     y = COURT_L = 23.77
//   singles lines    x = 0 and x = COURT_W = 8.23
//
// The tapped corners are the SINGLES corners; the doubles alleys are a fixed
// 1.37 m outside them, so doubles needs no extra calibration.
//
// The two player gates (the user's rules), on the box's BOTTOM-CENTRE:
//   isNear()    the box bottom is closer to the near baseline than the far one
//   inBounds()  the box x-centre is inside the doubles court plus 3 ft
// NaN always answers false: a player we cannot place is excluded, not admitted.
//
// Evidence and history: docs/anya2/NOTES.md, "court.py".
import Foundation

public enum Court {
    public static let COURT_W = 8.23                 // singles width, metres
    public static let COURT_L = 23.77                // baseline to baseline, metres
    public static let MID_Y = COURT_L / 2            // 11.885
    public static let ALLEY_W = 1.37                 // doubles alley, each side
    public static let FT_TO_M = 0.3048
    public static let MARGIN_M = 3.0 * FT_TO_M       // the 3 ft margin
    public static let X_LO = -(ALLEY_W + MARGIN_M)   // -2.2844
    public static let X_HI = COURT_W + ALLEY_W + MARGIN_M  // 10.5144

    /// The frame every pose pass and the tapped corners are expressed in.
    public static let ANALYSIS_WIDTH = 960, ANALYSIS_HEIGHT = 540

    /// True where the point is closer to the near baseline than the far. NaN -> false.
    @inline(__always) public static func isNear(courtY y: Double) -> Bool {
        y.isFinite && abs(y) < abs(y - COURT_L)
    }

    /// True where x is inside the doubles court plus 3 ft. NaN -> false.
    @inline(__always) public static func inBounds(courtX x: Double) -> Bool {
        x.isFinite && x >= X_LO && x <= X_HI
    }

    /// Court metres of a box's ground point (bottom-centre); NaN in, NaN out.
    public static func project(_ homography: Homography,
                               box: (x1: Double, y1: Double, x2: Double, y2: Double)) -> (x: Double, y: Double) {
        let groundX = 0.5 * (box.x1 + box.x2), groundY = box.y2
        guard groundX.isFinite && groundY.isFinite else { return (.nan, .nan) }
        return homography.apply(x: groundX, y: groundY)
    }
}

/// A 3x3 projective map, row-major.
public struct Homography {
    public var m: [Double]   // 9 values

    public init(_ m: [Double]) { precondition(m.count == 9); self.m = m }

    public static let identity = Homography([1, 0, 0, 0, 1, 0, 0, 0, 1])

    /// Image (analysis px) -> court metres from the four tapped singles corners,
    /// in the order near-left, near-right, far-right, far-left
    /// (walking.court.load_homography, i.e. cv2.getPerspectiveTransform).
    public init(courtCorners corners: [(x: Double, y: Double)]) {
        // Python builds both point sets as float32 arrays, so the court it
        // maps onto is 8.22999954 x 23.77000046 m, not 8.23 x 23.77.
        let w = Float(Court.COURT_W), l = Float(Court.COURT_L)
        self = Homography(from: corners.map { (Float($0.x), Float($0.y)) },
                          to: [(0, 0), (w, 0), (w, l), (0, l)])
    }

    /// The homography taking 4 source points to 4 destination points, computed
    /// EXACTLY as cv2.getPerspectiveTransform does.
    ///
    /// Exactly, not just correctly: this 8x8 system is ill-conditioned (pixel
    /// coordinates times court metres), so any two correct solvers disagree in
    /// the 7th digit -- enough to tip a walking-classifier split now and then.
    /// So OpenCV's own recipe is reproduced: u-equations in rows 0-3 and
    /// v-equations in rows 4-7, the src*dst products taken in FLOAT32 (they are
    /// Point2f products in C++), then OpenCV's LU with partial pivoting.
    public init(from source: [(Float, Float)], to destination: [(Float, Float)]) {
        var a = [Double](repeating: 0, count: 64)   // 8x8, row-major
        var b = [Double](repeating: 0, count: 8)
        for i in 0..<4 {
            let (x, y) = source[i], (u, v) = destination[i]
            a[i * 8 + 0] = Double(x); a[i * 8 + 1] = Double(y); a[i * 8 + 2] = 1
            a[(i + 4) * 8 + 3] = Double(x); a[(i + 4) * 8 + 4] = Double(y); a[(i + 4) * 8 + 5] = 1
            a[i * 8 + 6] = Double(-x * u); a[i * 8 + 7] = Double(-y * u)
            a[(i + 4) * 8 + 6] = Double(-x * v); a[(i + 4) * 8 + 7] = Double(-y * v)
            b[i] = Double(u); b[i + 4] = Double(v)
        }
        Homography.solveLikeOpenCV(&a, &b, size: 8)
        self.m = b + [1]
    }

    /// OpenCV's LUImpl (core/src/matrix_decomp.cpp): Gaussian elimination with
    /// partial pivoting, then back substitution. Solves a x = b in place in `b`.
    private static func solveLikeOpenCV(_ a: inout [Double], _ b: inout [Double], size m: Int) {
        for i in 0..<m {
            var k = i
            for j in (i + 1)..<m where abs(a[j * m + i]) > abs(a[k * m + i]) { k = j }
            if k != i {
                for j in i..<m { a.swapAt(i * m + j, k * m + j) }
                b.swapAt(i, k)
            }
            let d = -1 / a[i * m + i]
            for j in (i + 1)..<m {
                let alpha = a[j * m + i] * d
                for k in (i + 1)..<m { a[j * m + k] += alpha * a[i * m + k] }
                b[j] += alpha * b[i]
            }
        }
        for i in stride(from: m - 1, through: 0, by: -1) {
            var s = b[i]
            for k in (i + 1)..<m { s -= a[i * m + k] * b[k] }
            b[i] = s / a[i * m + i]
        }
    }

    /// `cv2.perspectiveTransform` of one point.
    @inline(__always) public func apply(x: Double, y: Double) -> (x: Double, y: Double) {
        let w = m[6] * x + m[7] * y + m[8]
        let scale = abs(w) > Double(Float.ulpOfOne) ? 1 / w : 0
        return ((m[0] * x + m[1] * y + m[2]) * scale, (m[3] * x + m[4] * y + m[5]) * scale)
    }

    public static func * (lhs: Homography, rhs: Homography) -> Homography {
        var out = [Double](repeating: 0, count: 9)
        for row in 0..<3 {
            for col in 0..<3 {
                var total = 0.0
                for k in 0..<3 { total += lhs.m[row * 3 + k] * rhs.m[k * 3 + col] }
                out[row * 3 + col] = total
            }
        }
        return Homography(out)
    }

    public var inverse: Homography {
        let a = m
        let c00 = a[4] * a[8] - a[5] * a[7], c01 = -(a[3] * a[8] - a[5] * a[6])
        let c02 = a[3] * a[7] - a[4] * a[6]
        let det = a[0] * c00 + a[1] * c01 + a[2] * c02
        let adjugate = [c00, -(a[1] * a[8] - a[2] * a[7]), a[1] * a[5] - a[2] * a[4],
                        c01, a[0] * a[8] - a[2] * a[6], -(a[0] * a[5] - a[2] * a[3]),
                        c02, -(a[0] * a[7] - a[1] * a[6]), a[0] * a[4] - a[1] * a[3]]
        return Homography(adjugate.map { $0 / det })
    }
}

/// Image -> court metres AT A GIVEN SOURCE FRAME:  H_at(frame) = H_ref * W[frame].
///
/// `referenceHomography` comes from the tapped corners; `cameraTrack` gives the
/// warp from each sampled frame back onto the calibration frame (CameraTrack).
/// With no camera movement the warp is the identity, so this is exactly the
/// fixed homography -- through the same code path.
/// Frame numbers are SOURCE frames: the one clock every decimated pass shares.
public final class Geometry {
    public let referenceHomography: Homography
    public private(set) var cameraTrack: CameraTrack
    private var homographyBySample: [Int: Homography] = [:]
    private var inverseBySample: [Int: Homography] = [:]

    public init(referenceHomography: Homography, cameraTrack: CameraTrack = .identity) {
        self.referenceHomography = referenceHomography
        self.cameraTrack = cameraTrack
    }

    /// While recording, the camera track grows as its samples become final.
    /// The caller must only ask about frames the track already covers
    /// (a later sample could be the nearer one); see AnalysisScheduler.
    public func replaceCameraTrack(_ track: CameraTrack) {
        cameraTrack = track
        homographyBySample.removeAll(keepingCapacity: true)
        inverseBySample.removeAll(keepingCapacity: true)
    }

    public func homography(atSourceFrame frame: Int) -> Homography {
        let sample = cameraTrack.sampleIndex(nearestTo: frame)
        if let cached = homographyBySample[sample] { return cached }
        let h = referenceHomography * cameraTrack.warps[sample]
        homographyBySample[sample] = h
        return h
    }

    public func inverseHomography(atSourceFrame frame: Int) -> Homography {
        let sample = cameraTrack.sampleIndex(nearestTo: frame)
        if let cached = inverseBySample[sample] { return cached }
        let h = homography(atSourceFrame: frame).inverse
        inverseBySample[sample] = h
        return h
    }
}

/// Sampled image -> calibration-image warps, indexed by SOURCE frame
/// (the data half of camera.CameraTrack; registration itself is separate).
public struct CameraTrack {
    public var frames: [Int]
    public var warps: [Homography]

    public init(frames: [Int], warps: [Homography]) {
        precondition(frames.count == warps.count && !frames.isEmpty)
        self.frames = frames; self.warps = warps
    }

    /// The no-op track: a camera that never moved.
    public static let identity = CameraTrack(frames: [0], warps: [.identity])

    /// Index of the sample NEAREST `frame` (not the previous one): a jostle
    /// lands between two samples, and nearest halves the worst-case error.
    public func sampleIndex(nearestTo frame: Int) -> Int {
        // np.searchsorted (left): first index with frames[i] >= frame.
        var low = 0, high = frames.count
        while low < high {
            let mid = (low + high) / 2
            if frames[mid] < frame { low = mid + 1 } else { high = mid }
        }
        if low <= 0 { return 0 }
        if low >= frames.count { return frames.count - 1 }
        return (frames[low] - frame) < (frame - frames[low - 1]) ? low : low - 1
    }
}
