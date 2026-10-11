// PoseModelTests.swift
//
// The Swift letterbox + Core ML + decode against Ultralytics' own predict on
// real frames (tools/make_pose_check_frames.py writes the frames and
// reference.json). Skipped unless both are configured:
//   ANYA2_POSE_FRAMES  the frames directory
//   ANYA2_POSE_MODEL   an .mlpackage: the fp32 copy for the gate
//                      (export_pose_coreml.py --fp32-out PATH)
import Anya2Engine
@testable import Anya2Perception
import CoreGraphics
import CoreML
import CoreVideo
import Foundation
import ImageIO
import XCTest

final class PoseModelTests: XCTestCase {
    struct ReferencePerson: Decodable { let box: [Double]; let conf: Double; let keypoints: [Double] }
    struct ReferenceFrame: Decodable { let imgsz: Int; let people: [ReferencePerson] }

    func testLetterboxGeometryMatchesUltralytics() {
        // 960x540 at 640 -> 640x360 + 12 above, 12 below = 640x384.
        let near = LetterboxGeometry(sourceWidth: 960, sourceHeight: 540, imageSize: 640)
        XCTAssertEqual([near.resizedWidth, near.resizedHeight, near.padLeft, near.padTop, near.inputWidth, near.inputHeight],
                       [640, 360, 0, 12, 640, 384])
        // A 774x199 far band at 960 -> 960x247 + 4 above, 5 below = 960x256.
        let far = LetterboxGeometry(sourceWidth: 774, sourceHeight: 199, imageSize: 960)
        XCTAssertEqual([far.resizedWidth, far.resizedHeight, far.padTop, far.inputHeight], [960, 247, 4, 256])
    }

    func testPeopleMatchUltralytics() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let frames = environment["ANYA2_POSE_FRAMES"], let modelPath = environment["ANYA2_POSE_MODEL"] else {
            throw XCTSkip("set ANYA2_POSE_FRAMES and ANYA2_POSE_MODEL")
        }
        let reference = try JSONDecoder().decode([String: ReferenceFrame].self,
                                                 from: Data(contentsOf: URL(fileURLWithPath: frames + "/reference.json")))
        let model = try PoseModel(packageURL: URL(fileURLWithPath: modelPath), computeUnits: .cpuOnly)
        var worstIoU = 1.0, worstConfidence = 0.0, worstKeypoint = 0.0, people = 0
        for (name, expected) in reference.sorted(by: { $0.key < $1.key }) {
            let frame = try loadBGRA(frames + "/" + name)
            let geometry = LetterboxGeometry(sourceWidth: CVPixelBufferGetWidth(frame),
                                             sourceHeight: CVPixelBufferGetHeight(frame), imageSize: expected.imgsz)
            let found = try model.people(in: XCTUnwrap(letterbox(frame, geometry: geometry)), geometry: geometry)
            XCTAssertEqual(found.count, expected.people.count, name)
            people += expected.people.count
            for want in expected.people {
                let wantBox = (want.box[0], want.box[1], want.box[2], want.box[3])
                guard let match = found.max(by: { intersectionOverUnion($0.box, wantBox) < intersectionOverUnion($1.box, wantBox) })
                else { continue }
                worstIoU = min(worstIoU, intersectionOverUnion(match.box, wantBox))
                worstConfidence = max(worstConfidence, abs(match.confidence - want.conf))
                for k in 0..<17 where want.keypoints[k * 3 + 2] > 0.5 && match.keypoints[k * 3 + 2] > 0.5 {
                    worstKeypoint = max(worstKeypoint, abs(match.keypoints[k * 3] - want.keypoints[k * 3]),
                                        abs(match.keypoints[k * 3 + 1] - want.keypoints[k * 3 + 1]))
                }
            }
        }
        print(String(format: "pose: %d people; worst IoU %.3f, conf diff %.4f, keypoint diff %.2f px",
                     people, worstIoU, worstConfidence, worstKeypoint))
        // export_pose_coreml.py's gate. With the fp32 model these hold exactly
        // (IoU 1.000, conf 0.0000, 0.55 px) because the letterbox is cv2's
        // pixel for pixel. The shipped fp16 model does NOT pass them (CPU:
        // 0.988 / 0.016 / 2.7 px) and is judged end to end on the corpus.
        XCTAssertGreaterThanOrEqual(worstIoU, 0.99)
        XCTAssertLessThanOrEqual(worstConfidence, 0.005)
        XCTAssertLessThanOrEqual(worstKeypoint, 1.0)
    }

    /// A PNG as a BGRA pixel buffer, colour values untouched.
    private func loadBGRA(_ path: String) throws -> CVPixelBuffer {
        let source = try XCTUnwrap(CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil))
        let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, image.width, image.height, kCVPixelFormatType_32BGRA, nil, &buffer)
        let pixels = try XCTUnwrap(buffer)
        CVPixelBufferLockBaseAddress(pixels, [])
        defer { CVPixelBufferUnlockBaseAddress(pixels, []) }
        let context = try XCTUnwrap(CGContext(
            data: CVPixelBufferGetBaseAddress(pixels), width: image.width, height: image.height, bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixels), space: image.colorSpace ?? CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue))
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return pixels
    }
}
