// CalibrationFrame.swift
//
// The 960x540 frame the court corners were tapped on. Camera registration
// matches every later frame against it, so it is kept for the whole match:
//   calibration.gray   the greyscale bytes registration uses (cv2-exact)
//   calibration.png    the colour frame, for showing the user
import Anya2Perception
import CoreGraphics
import CoreImage
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers

struct CalibrationFrame {
    static let width = 960, height = 540
    let gray: [UInt8]

    /// Save a 960x540 BGRA frame for this match.
    static func save(_ frame: CVPixelBuffer, matchID: String) throws {
        let converted = grayscale(of: frame)
        try Data(converted.pixels).write(to: MatchStore.file("calibration.gray", of: matchID), options: .atomic)
        let image = CIImage(cvPixelBuffer: frame)
        guard let cgImage = CIContext().createCGImage(image, from: image.extent),
              let destination = CGImageDestinationCreateWithURL(
                  MatchStore.file("calibration.png", of: matchID) as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { return }
        CGImageDestinationAddImage(destination, cgImage, nil)
        CGImageDestinationFinalize(destination)
    }

    static func load(matchID: String) throws -> CalibrationFrame {
        let data = try Data(contentsOf: MatchStore.file("calibration.gray", of: matchID))
        guard data.count == width * height else { throw CocoaError(.fileReadCorruptFile) }
        return CalibrationFrame(gray: [UInt8](data))
    }

    func withGray<T>(_ body: (GrayFrame) throws -> T) rethrows -> T {
        try gray.withUnsafeBufferPointer {
            try body(GrayFrame(pixels: $0.baseAddress!, width: Self.width, height: Self.height, bytesPerRow: Self.width))
        }
    }
}
