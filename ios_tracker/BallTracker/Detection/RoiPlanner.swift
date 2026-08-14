import CoreGraphics
import Foundation

/// Plans which source-pixel regions the small (480x288) ball model runs on
/// each frame, so the full-frame 1920-input model never has to run per frame.
///
/// Steady state (the tracker has a position): one crop centred on the track,
/// sized so the model sees the ball at the *same effective resolution* as the
/// full-frame 1920 model (crop width = inputW / analysisScale). Accuracy on
/// that crop then carries over from the full-frame model by construction —
/// same pixels-per-ball, same weights.
///
/// On a miss the crop escalates before giving up: brief coast keeps the tight
/// ROI (the Kalman prediction is still good), a longer gap doubles the crop
/// (half resolution — the 960-model regime — but 4x the area to reacquire in),
/// and only a dead track falls back to the scan.
///
/// Scan (no track): a distance-tiered SAHI-style sweep. The bottom of the
/// frame (near court) is covered by half-resolution tiles — near-ball motion
/// streaks are large, so downsampling is affordable — while the top half
/// (far court) gets full-effective-resolution tiles, which is exactly where
/// the 960x544 model lost the far ball. Tiles overlap by more than a fast
/// ball's streak length so no streak is ever split across all the tiles that
/// could have seen it.
struct RoiCrop {
    let rect: CGRect      // source px, top-left origin
    let maxBoxPx: Float   // streak-size bound in *this crop's* model-input px
}

final class RoiPlanner {
    /// Keep the tight track-centred ROI while the gap to the last detection is
    /// below this. BallTrackManager treats > 1.5 frames as coasting; a tight
    /// ROI is still safe well past that because the crop half-width (~160
    /// source px) dwarfs per-frame ball motion.
    static let tightHoldSec = 0.35
    /// Between tight and this, double the crop around the prediction: half the
    /// effective resolution, four times the reacquisition area. Past it the
    /// prediction is stale enough that only a scan can be trusted (must stay
    /// below BallTrackManager.missTimeoutS = 2.0, after which position is nil
    /// anyway).
    static let wideHoldSec = 0.9
    /// Perspective floor for the near-court streak-bound boost, matching
    /// makeImageRowPerspective's default farFloor.
    static let perspFarFloor: CGFloat = 0.35
    /// How much larger the streak bound may grow at the bottom of the frame,
    /// where near-court balls streak longest (box size tracks ball *speed*, not
    /// range — see BallDetector.defaultMaxBoxPx). 1.0 → 2x at the very bottom,
    /// tapering to 1x (unchanged) at the far court, so far balls keep the tight
    /// cap that rejects large non-ball blobs.
    static let maxBoxNearBoost: CGFloat = 1.0
    /// Fraction of the tight-crop half-width that one frame of predicted travel
    /// may reach before the crop is grown to keep the streak inside.
    static let travelGrowFrac: CGFloat = 0.6

    private let inputW: CGFloat
    private let inputH: CGFloat

    /// Diagnostics for the harness: how many frames ran which mode, and total
    /// model invocations.
    private(set) var roiFrames = 0
    private(set) var scanFrames = 0
    private(set) var tileInferences = 0

    init(inputWidth: Int, inputHeight: Int) {
        self.inputW = CGFloat(inputWidth)
        self.inputH = CGFloat(inputHeight)
    }

    /// The streak-size bound, constant in analysis space (defaultMaxBoxPx is
    /// quoted at the 960-wide reference), converted to a given crop's input px.
    /// `cropCenterY`/`frameHeight` (source px) apply the near-court boost so the
    /// fastest near serves aren't clipped by a flat cap; a nil pair (or a
    /// degenerate frame) leaves the bound at the un-boosted far-court value.
    private func maxBox(cropWidth: CGFloat, analysisScale: CGFloat,
                        cropCenterY: CGFloat, frameHeight: CGFloat) -> Float {
        let boundAnalysis = CGFloat(BallDetector.defaultMaxBoxPx)
            * TrackerEngine.analysisWidth / CGFloat(BallDetector.referenceWidth)
        let f = frameHeight > 0
            ? max(Self.perspFarFloor, min(1.0, cropCenterY / frameHeight))
            : Self.perspFarFloor
        let boost = 1.0 + Self.maxBoxNearBoost
            * (f - Self.perspFarFloor) / (1.0 - Self.perspFarFloor)
        return Float(boundAnalysis * boost / analysisScale * (inputW / cropWidth))
    }

    /// Crops for one frame. `status` is the tracker's verdict from the
    /// *previous* frame (nil before the first); `fps` drives the velocity lead.
    func crops(frameSize: CGSize, analysisScale: CGFloat,
               fps: Double, status: TrackStatus?) -> [RoiCrop] {
        let plans: [RoiCrop]
        if let s = status, let pos = s.position {
            let dt = 1.0 / max(fps, 1e-6)
            // Velocity-lead: centre the crop where the ball will be next frame,
            // not where it was. A serve moves the ball most of a crop-width per
            // frame, so centring on the last position leaves it at the edge.
            let cx = CGFloat(pos.x + s.velocityPxS.x * dt) / analysisScale
            let cy = CGFloat(pos.y + s.velocityPxS.y * dt) / analysisScale
            // Grow the crop when one frame of travel would otherwise reach the
            // edge, so a fast ball's streak stays inside — without paying 4x
            // area on the common slow-ball case.
            let travelSrc = CGFloat(s.speedPxS) * CGFloat(dt) / analysisScale
            let tightHalf = inputW / analysisScale / 2
            let speedFactor: CGFloat = travelSrc > tightHalf * Self.travelGrowFrac ? 2 : 1
            let gapFactor: CGFloat = s.timeSinceDetection <= Self.tightHoldSec ? 1 : 2
            let factor = max(speedFactor, gapFactor)
            if s.timeSinceDetection <= Self.wideHoldSec {
                plans = [centered(cx: cx, cy: cy, factor: factor,
                                  frame: frameSize, analysisScale: analysisScale)]
                roiFrames += 1
            } else {
                plans = scanTiles(frame: frameSize, analysisScale: analysisScale)
                scanFrames += 1
            }
        } else {
            plans = scanTiles(frame: frameSize, analysisScale: analysisScale)
            scanFrames += 1
        }
        tileInferences += plans.count
        return plans
    }

    /// A crop of `factor`x the native ROI size centred on (cx, cy), shifted
    /// (never shrunk, so the letterbox scale stays exact) to fit the frame.
    private func centered(cx: CGFloat, cy: CGFloat, factor: CGFloat,
                          frame: CGSize, analysisScale: CGFloat) -> RoiCrop {
        let w = min(inputW / analysisScale * factor, frame.width)
        let h = min(inputH / analysisScale * factor, frame.height)
        let x = min(max(cx - w / 2, 0), frame.width - w)
        let y = min(max(cy - h / 2, 0), frame.height - h)
        return RoiCrop(rect: CGRect(x: x, y: y, width: w, height: h),
                       maxBoxPx: maxBox(cropWidth: w, analysisScale: analysisScale,
                                        cropCenterY: y + h / 2, frameHeight: frame.height))
    }

    /// The distance-tiered acquisition sweep, covering the whole frame.
    private func scanTiles(frame: CGSize, analysisScale: CGFloat) -> [RoiCrop] {
        var out: [RoiCrop] = []

        // Near tier — bottom band at half effective resolution (2x crops).
        // Streaks there are large; the shipped 960 model's per-pixel scale.
        let nearW = min(inputW / analysisScale * 2, frame.width)
        let nearH = min(inputH / analysisScale * 2, frame.height)
        let nearY = frame.height - nearH
        let nearBox = maxBox(cropWidth: nearW, analysisScale: analysisScale,
                             cropCenterY: nearY + nearH / 2, frameHeight: frame.height)
        for x in tilePositions(span: frame.width, tile: nearW, overlap: nearW / 2) {
            out.append(RoiCrop(rect: CGRect(x: x, y: nearY, width: nearW, height: nearH),
                               maxBoxPx: nearBox))
        }

        // Far tier — top half at full effective resolution, where the far
        // ball's few raw pixels are the whole detection signal. The streak bound
        // is computed per-tile so the boost tapers correctly down the frame.
        let farW = min(inputW / analysisScale, frame.width)
        let farH = min(inputH / analysisScale, frame.height)
        let farSpanY = frame.height / 2
        for y in tilePositions(span: farSpanY, tile: farH, overlap: farH / 4) {
            let farBox = maxBox(cropWidth: farW, analysisScale: analysisScale,
                                cropCenterY: y + farH / 2, frameHeight: frame.height)
            for x in tilePositions(span: frame.width, tile: farW, overlap: farW / 4) {
                out.append(RoiCrop(rect: CGRect(x: x, y: y, width: farW, height: farH),
                                   maxBoxPx: farBox))
            }
        }
        return out
    }

    /// Tile origins covering [0, span] with the requested overlap, end-aligned
    /// so the last tile never runs past the span.
    private func tilePositions(span: CGFloat, tile: CGFloat, overlap: CGFloat) -> [CGFloat] {
        guard tile < span else { return [0] }
        let stride = tile - overlap
        var xs: [CGFloat] = []
        var x: CGFloat = 0
        while x + tile < span {
            xs.append(x)
            x += stride
        }
        xs.append(span - tile)
        return xs
    }

    /// Merge duplicate detections from overlapping crops: keep the highest-
    /// confidence detection within `radius` source px of any kept one.
    static func dedup(_ dets: [BallDetection], radius: CGFloat = 8) -> [BallDetection] {
        guard dets.count > 1 else { return dets }
        let sorted = dets.sorted { $0.conf > $1.conf }
        var kept: [BallDetection] = []
        for d in sorted {
            let c = d.center
            let dup = kept.contains { k in
                let kc = k.center
                return hypot(c.x - kc.x, c.y - kc.y) < radius
            }
            if !dup { kept.append(d) }
        }
        return kept
    }
}
