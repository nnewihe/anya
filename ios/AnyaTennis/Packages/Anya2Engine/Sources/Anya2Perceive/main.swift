// anya2-perceive: the app's perception (letterbox, Core ML pose, decode) on a
// corpus clip's own proxies, written in perceive.py's layout so Python can
// run stages 5-8 on it and score it against ground truth
// (tools/score_swift_perception.py does both).
//
//   anya2-perceive MODEL.mlpackage PROXY540.mp4 FARBAND.mp4 X1,Y1,X2,Y2 SRC_W,SRC_H OUT_DIR [cpu|all]
//
// Writes OUT_DIR/near.bin and far.bin: int32 samples, then float32
// keypoints [samples, 8, 17, 3], boxes [samples, 8, 4], confidences [samples, 8].
import Accelerate
import Anya2Engine
import Anya2Perception
import AVFoundation
import CoreML
import Foundation

let arguments = CommandLine.arguments
guard arguments.count >= 7 else {
    print("usage: anya2-perceive MODEL.mlpackage PROXY540.mp4 FARBAND.mp4 X1,Y1,X2,Y2 SRC_W,SRC_H OUT_DIR [cpu|all]")
    exit(2)
}
let crop = arguments[4].split(separator: ",").map { Int($0)! }
let sourceSize = arguments[5].split(separator: ",").map { Int($0)! }
let outDir = URL(fileURLWithPath: arguments[6])
let computeUnits: MLComputeUnits = arguments.count > 7 && arguments[7] == "cpu" ? .cpuOnly : .all
// "bt601": convert YUV to BGR with the BT.601 video-range matrix, as
// ffmpeg/OpenCV do for perceive.py, instead of the file's own tagged matrix.
let bt601 = arguments.contains("bt601")
try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

let band = FarBand(x1: crop[0], y1: crop[1], x2: crop[2], y2: crop[3],
                   scaleX: Double(sourceSize[0]) / Double(Court.ANALYSIS_WIDTH),
                   scaleY: Double(sourceSize[1]) / Double(Court.ANALYSIS_HEIGHT))
let perceiver = try Perceiver(compiledModelURL: try MLModel.compileModel(at: URL(fileURLWithPath: arguments[1])),
                               computeUnits: computeUnits, band: band)

/// Every `stride`-th frame of a video as BGRA, in order.
final class FrameReader {
    let reader: AVAssetReader
    let output: AVAssetReaderTrackOutput
    let fps: Double
    init(_ path: String) throws {
        let asset = AVURLAsset(url: URL(fileURLWithPath: path))
        let semaphore = DispatchSemaphore(value: 0)
        var track: AVAssetTrack?
        var rate: Float = 0
        Task {
            track = try? await asset.loadTracks(withMediaType: .video).first
            rate = (try? await track?.load(.nominalFrameRate)) ?? 0
            semaphore.signal()
        }
        semaphore.wait()
        reader = try AVAssetReader(asset: asset)
        output = AVAssetReaderTrackOutput(track: track!, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String:
                bt601 ? kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange : kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        reader.startReading()
        fps = Double(rate)
    }
    func next() -> CVPixelBuffer? {
        guard let sample = output.copyNextSampleBuffer(), let pixels = CMSampleBufferGetImageBuffer(sample) else { return nil }
        return bt601 ? toBGRA601(pixels) : pixels
    }
}

let near = try FrameReader(arguments[2]), far = try FrameReader(arguments[3])
let stride = PerceiveRules.stride(sourceFPS: near.fps).stride
var nearRows = PassColumns(), farRows = PassColumns()
var frame = 0, samples = 0
let started = Date()
while let nearFrame = near.next(), let farFrame = far.next() {
    defer { frame += 1 }
    guard frame % stride == 0 else { continue }
    let perceived = try autoreleasepool { try perceiver.perceive(nearFrame: nearFrame, farBandFrame: farFrame) }
    nearRows.append(perceived.near); farRows.append(perceived.far)
    samples += 1
    if samples % 2000 == 0 {
        print(String(format: "  %d samples, %.1f samples/s", samples, Double(samples) / Date().timeIntervalSince(started)))
    }
}
try nearRows.write(to: outDir.appendingPathComponent("near.bin"), samples: samples)
try farRows.write(to: outDir.appendingPathComponent("far.bin"), samples: samples)
print(String(format: "%d samples (stride %d, %.2f fps) in %.0f s", samples, stride, near.fps / Double(stride),
             Date().timeIntervalSince(started)))

struct PassColumns {
    var keypoints: [Float] = [], boxes: [Float] = [], confidences: [Float] = []
    mutating func append(_ row: PassRow) {
        keypoints += row.keypoints; boxes += row.boxes; confidences += row.confidences
    }
    func write(to url: URL, samples: Int) throws {
        var data = Data()
        withUnsafeBytes(of: Int32(samples)) { data.append(contentsOf: $0) }
        for column in [keypoints, boxes, confidences] { column.withUnsafeBytes { data.append(contentsOf: $0) } }
        try data.write(to: url)
    }
}

/// NV12 video range -> BGRA with the BT.601 matrix (vImage).
func toBGRA601(_ yuv: CVPixelBuffer) -> CVPixelBuffer {
    let width = CVPixelBufferGetWidth(yuv), height = CVPixelBufferGetHeight(yuv)
    var out: CVPixelBuffer?
    CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &out)
    CVPixelBufferLockBaseAddress(yuv, .readOnly); CVPixelBufferLockBaseAddress(out!, [])
    defer { CVPixelBufferUnlockBaseAddress(out!, []); CVPixelBufferUnlockBaseAddress(yuv, .readOnly) }
    var y = vImage_Buffer(data: CVPixelBufferGetBaseAddressOfPlane(yuv, 0), height: vImagePixelCount(height),
                          width: vImagePixelCount(width), rowBytes: CVPixelBufferGetBytesPerRowOfPlane(yuv, 0))
    var cbcr = vImage_Buffer(data: CVPixelBufferGetBaseAddressOfPlane(yuv, 1), height: vImagePixelCount(height / 2),
                             width: vImagePixelCount(width / 2), rowBytes: CVPixelBufferGetBytesPerRowOfPlane(yuv, 1))
    var dst = vImage_Buffer(data: CVPixelBufferGetBaseAddress(out!), height: vImagePixelCount(height),
                            width: vImagePixelCount(width), rowBytes: CVPixelBufferGetBytesPerRow(out!))
    var range = vImage_YpCbCrPixelRange(Yp_bias: 16, CbCr_bias: 128, YpRangeMax: 235, CbCrRangeMax: 240,
                                        YpMax: 235, YpMin: 16, CbCrMax: 240, CbCrMin: 16)
    var info = vImage_YpCbCrToARGB()
    vImageConvert_YpCbCrToARGB_GenerateConversion(kvImage_YpCbCrToARGBMatrix_ITU_R_601_4, &range, &info,
                                                  kvImage420Yp8_CbCr8, kvImageARGB8888, vImage_Flags(kvImageNoFlags))
    // ARGB order [3,2,1,0] -> B G R A in memory.
    let permute: [UInt8] = [3, 2, 1, 0]
    vImageConvert_420Yp8_CbCr8ToARGB8888(&y, &cbcr, &dst, &info, permute, 255, vImage_Flags(kvImageNoFlags))
    return out!
}
