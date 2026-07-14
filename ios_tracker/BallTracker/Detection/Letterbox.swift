import CoreImage
import CoreVideo

/// The uniform-scale + centered-pad transform mapping source-frame pixels into
/// the model's input, so detections can be mapped back out.
struct LetterboxTransform {
    let scale: CGFloat  // source px -> model-input px
    let padX: CGFloat
    let padY: CGFloat

    func unmap(_ p: CGPoint) -> CGPoint {
        CGPoint(x: (p.x - padX) / scale, y: (p.y - padY) / scale)
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

    func apply(to src: CVPixelBuffer) -> (buffer: CVPixelBuffer, transform: LetterboxTransform)? {
        let sw = CGFloat(CVPixelBufferGetWidth(src))
        let sh = CGFloat(CVPixelBufferGetHeight(src))
        let r = min(CGFloat(width) / sw, CGFloat(height) / sh)
        let padX = (CGFloat(width) - sw * r) / 2
        let padY = (CGFloat(height) - sh * r) / 2

        var out: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &out) == kCVReturnSuccess,
              let out else { return nil }

        // CoreImage's origin is bottom-left; the pad is vertically centered, so
        // the same translation works in both orientations.
        let scaled = CIImage(cvPixelBuffer: src)
            .transformed(by: CGAffineTransform(scaleX: r, y: r)
                .translatedBy(x: padX / r, y: padY / r))
        let bounds = CGRect(x: 0, y: 0, width: width, height: height)
        let composed = scaled.composited(over: gray).cropped(to: bounds)
        ciContext.render(composed, to: out, bounds: bounds,
                         colorSpace: CGColorSpaceCreateDeviceRGB())

        return (out, LetterboxTransform(scale: r, padX: padX, padY: padY))
    }
}
