// UltralyticsLetterbox.swift
//
// Fit a frame into the pose model's input exactly as Ultralytics' predict does
// for a .pt model (LetterBox with auto=True), which is what perceive.py runs:
//   - scale by gain = min(imgsz / height, imgsz / width) -- up or down
//   - pad only to the next multiple of 32 (a 4:1 far band becomes 960 x ~256,
//     not a 960 square: square padding changes the model's confidences)
//   - split odd padding the way Ultralytics does: round(d - 0.1) before
//   - fill the padding with grey 114
// and resize with cv2.resize(INTER_LINEAR), reproduced bit for bit (below).
import Anya2Engine
import CoreVideo
import Foundation

public struct LetterboxGeometry: Equatable {
    public let sourceWidth: Int, sourceHeight: Int
    public let gain: Double
    public let resizedWidth: Int, resizedHeight: Int
    public let padLeft: Int, padTop: Int
    public let inputWidth: Int, inputHeight: Int

    public init(sourceWidth: Int, sourceHeight: Int, imageSize: Int) {
        self.sourceWidth = sourceWidth; self.sourceHeight = sourceHeight
        gain = min(Double(imageSize) / Double(sourceHeight), Double(imageSize) / Double(sourceWidth))
        resizedWidth = pythonRound(Double(sourceWidth) * gain)
        resizedHeight = pythonRound(Double(sourceHeight) * gain)
        let padX = Double((imageSize - resizedWidth) % 32) / 2, padY = Double((imageSize - resizedHeight) % 32) / 2
        padLeft = pythonRound(padX - 0.1); padTop = pythonRound(padY - 0.1)
        inputWidth = resizedWidth + padLeft + pythonRound(padX + 0.1)
        inputHeight = resizedHeight + padTop + pythonRound(padY + 0.1)
    }

    /// Model-input pixels back to source pixels, clipped to the source frame
    /// (Ultralytics' scale_boxes / scale_coords).
    @inline(__always) public func toSource(x: Double, y: Double) -> (x: Double, y: Double) {
        (clip((x - Double(padLeft)) / gain, 0, Double(sourceWidth)),
         clip((y - Double(padTop)) / gain, 0, Double(sourceHeight)))
    }
}

/// Letterbox a BGRA frame (or a region of it) into a new BGRA pixel buffer
/// of the geometry's input size.
public func letterbox(_ source: CVPixelBuffer, region: (x: Int, y: Int, width: Int, height: Int)? = nil,
                      geometry: LetterboxGeometry, pool: CVPixelBufferPool? = nil) -> CVPixelBuffer? {
    var output: CVPixelBuffer?
    if let pool {
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &output)
    } else {
        CVPixelBufferCreate(nil, geometry.inputWidth, geometry.inputHeight, kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &output)
    }
    guard let output else { return nil }
    CVPixelBufferLockBaseAddress(source, .readOnly)
    CVPixelBufferLockBaseAddress(output, [])
    defer {
        CVPixelBufferUnlockBaseAddress(output, [])
        CVPixelBufferUnlockBaseAddress(source, .readOnly)
    }
    let region = region ?? (0, 0, CVPixelBufferGetWidth(source), CVPixelBufferGetHeight(source))
    let sourceStride = CVPixelBufferGetBytesPerRow(source)
    let outputStride = CVPixelBufferGetBytesPerRow(output)
    let src = CVPixelBufferGetBaseAddress(source)!.assumingMemoryBound(to: UInt8.self)
    let dst = CVPixelBufferGetBaseAddress(output)!.assumingMemoryBound(to: UInt8.self)

    // Grey 114 everywhere, then the resized frame on top.
    for y in 0..<geometry.inputHeight {
        let row = dst + y * outputStride
        for x in 0..<geometry.inputWidth {
            row[x * 4] = 114; row[x * 4 + 1] = 114; row[x * 4 + 2] = 114; row[x * 4 + 3] = 255
        }
    }
    let columns = columnTaps(source: region.width, destination: geometry.resizedWidth)
    let rows = rowTaps(source: region.height, destination: geometry.resizedHeight)
    var upper = [Int32](repeating: 0, count: geometry.resizedWidth * 3)
    var lower = upper

    /// The horizontal pass of one source row (clamped into the region).
    func horizontal(_ sourceRow: Int, into out: inout [Int32]) {
        let line = src + (region.y + min(max(sourceRow, 0), region.height - 1)) * sourceStride + region.x * 4
        for (x, tap) in columns.enumerated() {
            let next = min(tap.index + 1, region.width - 1)
            for channel in 0..<3 {
                out[x * 3 + channel] = Int32(line[tap.index * 4 + channel]) * tap.weight0
                    + Int32(line[next * 4 + channel]) * tap.weight1
            }
        }
    }
    for (y, tap) in rows.enumerated() {
        horizontal(tap.index, into: &upper)
        horizontal(tap.index + 1, into: &lower)
        let row = dst + (y + geometry.padTop) * outputStride + geometry.padLeft * 4
        for x in 0..<geometry.resizedWidth {
            for channel in 0..<3 {
                let i = x * 3 + channel
                row[x * 4 + channel] = verticalBlend(upper[i], lower[i], tap.weight0, tap.weight1)
            }
            row[x * 4 + 3] = 255
        }
    }
    return output
}

// ── cv2.resize(INTER_LINEAR) on 8-bit images, bit for bit ─────────────────
//
// Bit for bit, not just bilinear: the far player is small and borderline, and
// a +-1 grey-level difference in the input moved a far confidence by 0.02 and
// a keypoint by 9 px on the check frames. OpenCV's 8-bit path
// (imgproc/src/resize.cpp) is fixed point: weights are rounded to 1/2048, the
// horizontal pass sums in ints, and the vertical pass (VResizeLinearVec_32s8u,
// NEON) shifts by 4, keeps the high 16 bits of each product, adds with int16
// saturation and rounds off the last 2 bits.

struct ResizeTap { let index: Int; let weight0: Int32; let weight1: Int32 }

/// Where destination coordinate d samples the source, in OpenCV's float32
/// arithmetic: (floor, fraction).
private func sourcePosition(_ d: Int, scale: Double) -> (index: Int, fraction: Float) {
    let f = Float((Double(d) + 0.5) * scale - 0.5)
    let index = Int(f.rounded(.down))
    return (index, f - Float(index))
}

private func fixedPointWeights(_ fraction: Float) -> (Int32, Int32) {
    (Int32(((1 - fraction) * 2048).rounded(.toNearestOrEven)), Int32((fraction * 2048).rounded(.toNearestOrEven)))
}

/// Columns: before the first pixel all weight goes on pixel 0, past the last
/// all weight on the last.
func columnTaps(source: Int, destination: Int) -> [ResizeTap] {
    let scale = Double(source) / Double(destination)
    return (0..<destination).map { d in
        var (index, fraction) = sourcePosition(d, scale: scale)
        if index < 0 { fraction = 0; index = 0 }
        if index + 1 >= source { fraction = 0; index = source - 1 }
        let (w0, w1) = fixedPointWeights(fraction)
        return ResizeTap(index: index, weight0: w0, weight1: w1)
    }
}

/// Rows are NOT clamped like columns: OpenCV keeps the computed weights and
/// only clamps which source rows it reads (`horizontal` does that). The two
/// weights can then sum to 2047 or 2049, and that shows in the output.
func rowTaps(source: Int, destination: Int) -> [ResizeTap] {
    let scale = Double(source) / Double(destination)
    return (0..<destination).map { d in
        let (index, fraction) = sourcePosition(d, scale: scale)
        let (w0, w1) = fixedPointWeights(fraction)
        return ResizeTap(index: index, weight0: w0, weight1: w1)
    }
}

@inline(__always) func verticalBlend(_ upper: Int32, _ lower: Int32, _ weight0: Int32, _ weight1: Int32) -> UInt8 {
    func highProduct(_ value: Int32, _ weight: Int32) -> Int32 { (Int32(Int16(clamping: value >> 4)) * weight) >> 16 }
    let sum = Int32(Int16(clamping: highProduct(upper, weight0) + highProduct(lower, weight1)))
    return UInt8(clamping: (sum + 2) >> 2)
}
