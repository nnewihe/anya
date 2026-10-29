// Grayscale.swift
//
// cv2.cvtColor(BGR2GRAY) on 8-bit frames, bit for bit: OpenCV's fixed-point
// weights (0.114, 0.587, 0.299 scaled by 2^14), rounded. Camera registration
// runs ORB on these, so they must match what camera.py feeds it.
import CoreVideo
import Foundation

/// The greyscale of a BGRA pixel buffer, as a dense [height * width] array.
public func grayscale(of frame: CVPixelBuffer) -> (pixels: [UInt8], width: Int, height: Int) {
    CVPixelBufferLockBaseAddress(frame, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(frame, .readOnly) }
    let width = CVPixelBufferGetWidth(frame), height = CVPixelBufferGetHeight(frame)
    let stride = CVPixelBufferGetBytesPerRow(frame)
    let base = CVPixelBufferGetBaseAddress(frame)!.assumingMemoryBound(to: UInt8.self)
    var gray = [UInt8](repeating: 0, count: width * height)
    for y in 0..<height {
        let row = base + y * stride
        for x in 0..<width {
            let b = Int32(row[x * 4]), g = Int32(row[x * 4 + 1]), r = Int32(row[x * 4 + 2])
            gray[y * width + x] = UInt8((b * 1868 + g * 9617 + r * 4899 + (1 << 13)) >> 14)
        }
    }
    return (gray, width, height)
}
