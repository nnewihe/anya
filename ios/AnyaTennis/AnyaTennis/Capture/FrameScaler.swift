// FrameScaler.swift
//
// The two pictures analysis needs from each 4K camera frame, made on the GPU:
//   near   the whole frame at 960x540 (the analysis frame, like _proxy540.mp4)
//   far    the far band at NATIVE resolution (like _farband.mp4): the far
//          player is ~40 px tall in 4K and would be ~10 px after scaling
// Both come out as BGRA, the layout the letterbox and the H.264 writer take.
import CoreImage
import CoreVideo
import Foundation

final class FrameScaler {
    let band: FarBandRect
    private let context = CIContext(options: [.workingColorSpace: NSNull(), .cacheIntermediates: false])
    private var nearPool: CVPixelBufferPool?
    private var farPool: CVPixelBufferPool?

    struct FarBandRect { let x: Int, y: Int, width: Int, height: Int }

    init(band: FarBandRect) {
        self.band = band
        nearPool = Self.pool(width: 960, height: 540)
        farPool = Self.pool(width: band.width, height: band.height)
    }

    /// The 960x540 frame. Lanczos, so the 4x downscale does not alias the
    /// court lines camera registration matches on.
    func near(_ frame: CVPixelBuffer) -> CVPixelBuffer? {
        let image = CIImage(cvPixelBuffer: frame)
        let scale = 960.0 / image.extent.width
        let scaled = image.applyingFilter("CILanczosScaleTransform",
                                          parameters: [kCIInputScaleKey: scale, kCIInputAspectRatioKey: 1.0])
        return render(scaled, from: nearPool, width: 960, height: 540)
    }

    /// The far band, pixel for pixel.
    func far(_ frame: CVPixelBuffer) -> CVPixelBuffer? {
        let image = CIImage(cvPixelBuffer: frame)
        // Core Image's origin is bottom-left; the band is in top-left pixels.
        let flippedY = Int(image.extent.height) - band.y - band.height
        let cropped = image.cropped(to: CGRect(x: band.x, y: flippedY, width: band.width, height: band.height))
            .transformed(by: CGAffineTransform(translationX: CGFloat(-band.x), y: CGFloat(-flippedY)))
        return render(cropped, from: farPool, width: band.width, height: band.height)
    }

    private func render(_ image: CIImage, from pool: CVPixelBufferPool?, width: Int, height: Int) -> CVPixelBuffer? {
        guard let pool else { return nil }
        var output: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &output)
        guard let output else { return nil }
        context.render(image, to: output, bounds: CGRect(x: 0, y: 0, width: width, height: height), colorSpace: nil)
        return output
    }

    private static func pool(width: Int, height: Int) -> CVPixelBufferPool? {
        var pool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(nil, [kCVPixelBufferPoolMinimumBufferCountKey: 6] as CFDictionary, [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: width, kCVPixelBufferHeightKey: height,
            kCVPixelBufferIOSurfacePropertiesKey: [:],
        ] as CFDictionary, &pool)
        return pool
    }
}
