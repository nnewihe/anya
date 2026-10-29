// CameraPreview.swift
//
// The live camera picture, and the conversion between a point on screen and
// a pixel of the 960x540 analysis frame (both ways). The preview fits the
// whole frame on screen (aspect fit), so every pixel is tappable.
import AVFoundation
import SwiftUI

final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }

    /// Screen point -> analysis pixel (960x540).
    func analysisPoint(from point: CGPoint) -> CGPoint {
        let device = previewLayer.captureDevicePointConverted(fromLayerPoint: point)
        return CGPoint(x: device.x * 960, y: device.y * 540)
    }

    /// Analysis pixel -> screen point.
    func screenPoint(from analysis: CGPoint) -> CGPoint {
        previewLayer.layerPointConverted(fromCaptureDevicePoint: CGPoint(x: analysis.x / 960, y: analysis.y / 540))
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    var onView: ((PreviewView) -> Void)?

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspect
        if let connection = view.previewLayer.connection, connection.isVideoRotationAngleSupported(0) {
            connection.videoRotationAngle = 0
        }
        view.backgroundColor = .black
        DispatchQueue.main.async { onView?(view) }
        return view
    }

    func updateUIView(_ view: PreviewView, context: Context) {}
}
