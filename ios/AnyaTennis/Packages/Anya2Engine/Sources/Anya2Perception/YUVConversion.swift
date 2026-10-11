// YUVConversion.swift
//
// Decoded video -> BGRA the way perceive.py sees it. Python reads proxies
// through ffmpeg/OpenCV, which convert YUV with the BT.601 video-range
// matrix whatever the file is tagged with; AVFoundation honours the tag
// (BT.709 for HD), which shifts colours by ~2 grey levels on average. The
// detectors were tuned on the BT.601 colours, and scoring the app's
// perception on the corpus showed it matters: point-end recall on Data/21
// 7/12 with the tagged matrix, 9/12 with BT.601 (Python 10/12).
import Accelerate
import CoreVideo

/// An NV12 video-range frame (kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
/// as BGRA, converted with the BT.601 matrix.
public func bgraBT601(fromVideoRangeYUV yuv: CVPixelBuffer) -> CVPixelBuffer {
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
