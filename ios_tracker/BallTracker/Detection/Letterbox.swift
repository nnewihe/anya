import CoreImage
import CoreVideo

/// The uniform-scale + centered-pad transform mapping source-frame pixels into
/// the model's input, so detections can be mapped back out. `cropX`/`cropY`
/// carry the origin of the source crop the input was cut from (0 for the
/// full frame), so unmapped points land in full-frame pixel space.
struct LetterboxTransform {
    let scale: CGFloat  // source px -> model-input px
    let padX: CGFloat
    let padY: CGFloat
    var cropX: CGFloat = 0
    var cropY: CGFloat = 0

    func unmap(_ p: CGPoint) -> CGPoint {
        CGPoint(x: (p.x - padX) / scale + cropX, y: (p.y - padY) / scale + cropY)
    }
}

/// Aspect-fit letterbox of any CVPixelBuffer into a fixed-size 32BGRA buffer
/// with the YOLO gray-114 padding, GPU-accelerated via CoreImage. Mirrors the
/// Ultralytics letterbox (uniform scale, centered pad) the parity fixtures
/// were generated with.
final class Letterbox {
    let width: Int
    let height: Int
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let pool: CVPixelBufferPool
    private let gray = CIImage(color: CIColor(red: 114.0 / 255, green: 114.0 / 255, blue: 114.0 / 255))

    init?(width: Int, height: Int) {
        self.width = width
        self.height = height
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        var pool: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(nil, nil, attrs as CFDictionary, &pool) == kCVReturnSuccess,
              let pool else { return nil }
        self.pool = pool
    }

    /// `crop`, when set, letterboxes only that source-pixel region (top-left
    /// origin) into the model input; the returned transform unmaps detections
    /// back to full-frame pixels.
    func apply(to src: CVPixelBuffer,
               crop: CGRect? = nil) -> (buffer: CVPixelBuffer, transform: LetterboxTransform)? {
        let fullW = CGFloat(CVPixelBufferGetWidth(src))
        let fullH = CGFloat(CVPixelBufferGetHeight(src))
        let region = crop ?? CGRect(x: 0, y: 0, width: fullW, height: fullH)
        let sw = region.width
        let sh = region.height
        guard sw > 0, sh > 0 else { return nil }
        let r = min(CGFloat(width) / sw, CGFloat(height) / sh)
        let padX = (CGFloat(width) - sw * r) / 2
        let padY = (CGFloat(height) - sh * r) / 2

        var out: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &out) == kCVReturnSuccess,
              let out else { return nil }

        // CoreImage's origin is bottom-left; the pad is vertically centered, so
        // the same translation works in both orientations. A crop is cut in
        // CI (bottom-left) coordinates and shifted to the origin first.
        var img = CIImage(cvPixelBuffer: src)
        if crop != nil {
            let ciRect = CGRect(x: region.minX, y: fullH - region.maxY,
                                width: sw, height: sh)
            img = img.cropped(to: ciRect)
                .transformed(by: CGAffineTransform(translationX: -ciRect.minX,
                                                   y: -ciRect.minY))
        }
        let scaled = img
            .transformed(by: CGAffineTransform(scaleX: r, y: r)
                .translatedBy(x: padX / r, y: padY / r))
        let bounds = CGRect(x: 0, y: 0, width: width, height: height)
        let composed = scaled.composited(over: gray).cropped(to: bounds)
        ciContext.render(composed, to: out, bounds: bounds,
                         colorSpace: CGColorSpaceCreateDeviceRGB())

        return (out, LetterboxTransform(scale: r, padX: padX, padY: padY,
                                        cropX: region.minX, cropY: region.minY))
    }
}
