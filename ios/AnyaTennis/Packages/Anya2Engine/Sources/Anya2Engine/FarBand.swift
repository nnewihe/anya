// FarBand.swift  (perceive.far_band)
//
// The far player is small, so the far pose pass runs on a native-resolution
// crop around the far baseline instead of the 960x540 frame.
//
// CAUSAL, unlike main: main grows the band to cover where the baseline lands
// under EVERY camera-track warp of the whole match (`_band_union`), which a
// live recording cannot know at the start. Here the band comes from the
// calibration frame alone -- what main computes when the camera never moves.
// See README "Perception" for the corpus comparison of the two rules.
import Foundation

public enum FarBandRules {
    public static let FAR_BAND_UP_M = 2.8     // how high above the ground: a raised racket at the trophy
    public static let FAR_BAND_PAD_M = 4.0    // along the court, each side of the far baseline
    public static let FAR_BAND_EDGE_PX = 2.0  // a box this close to the crop's bottom edge is cut off
}

/// The crop, in SOURCE pixels, and how source pixels relate to analysis pixels.
public struct FarBand: Equatable {
    public let x1: Int, y1: Int, x2: Int, y2: Int
    public let scaleX: Double, scaleY: Double   // source px per analysis px

    public var width: Int { x2 - x1 }
    public var height: Int { y2 - y1 }
    /// The crop actually encoded: rounded DOWN to even sizes for 4:2:0 video
    /// (proxy.ensure_crop_proxy). The bottom-edge rule still uses `height`.
    public var encodedWidth: Int { width & ~1 }
    public var encodedHeight: Int { height & ~1 }

    public init(x1: Int, y1: Int, x2: Int, y2: Int, scaleX: Double, scaleY: Double) {
        self.x1 = x1; self.y1 = y1; self.x2 = x2; self.y2 = y2; self.scaleX = scaleX; self.scaleY = scaleY
    }

    /// `imageToCourt`: the clicked homography (analysis px -> court metres).
    public init(imageToCourt: Homography, sourceWidth: Int, sourceHeight: Int,
                padM: Double = FarBandRules.FAR_BAND_PAD_M, upM: Double = FarBandRules.FAR_BAND_UP_M) {
        // 1. A grid along the far baseline +/- padM, projected into the image.
        let courtToImage = imageToCourt.inverse
        var points: [(x: Double, y: Double)] = []
        for y in [Court.COURT_L - padM, Court.COURT_L + padM] {
            for i in 0..<9 {
                let x = Court.X_LO + (Court.X_HI - Court.X_LO) * Double(i) / 8   // np.linspace(X_LO, X_HI, 9)
                points.append(courtToImage.apply(x: x, y: y))
            }
        }
        // 2. Their bounding box in source pixels.
        scaleX = Double(sourceWidth) / Double(Court.ANALYSIS_WIDTH)
        scaleY = Double(sourceHeight) / Double(Court.ANALYSIS_HEIGHT)
        let xs = points.map(\.x), ys = points.map(\.y)
        x1 = max(0, Int((xs.min()! * scaleX).rounded(.down)))
        x2 = min(sourceWidth, Int((xs.max()! * scaleX).rounded(.up)))
        let top = max(0, Int((ys.min()! * scaleY).rounded(.down)))
        y2 = min(sourceHeight, Int((ys.max()! * scaleY).rounded(.up)))
        // 3. Room above the ground for a raised racket: upM metres at the
        //    band's own px-per-metre (the ground homography carries no height).
        let pxPerMetre = (xs.max()! - xs.min()!) * scaleX / (Court.X_HI - Court.X_LO)
        y1 = max(0, top - pythonRound(upM * pxPerMetre))
    }

    /// Far-crop pixels -> analysis pixels (perceive._to_analysis_frame).
    @inline(__always) public func toAnalysis(x: Double, y: Double) -> (x: Double, y: Double) {
        (x / scaleX + Double(x1) / scaleX, y / scaleY + Double(y1) / scaleY)
    }
}
