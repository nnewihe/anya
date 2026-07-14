import AVFoundation
import Foundation

/// Owns the AVCaptureSession and hands frames (with presentation timestamps)
/// to a callback on the delegate queue. The callback runs synchronously there:
/// with `alwaysDiscardsLateVideoFrames` AVFoundation drops frames that arrive
/// while it is busy, so inference always sees the freshest frame.
final class CameraManager: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    let previewLayer = AVCaptureVideoPreviewLayer()

    /// Called on the delegate queue for every frame that isn't dropped.
    var onFrame: ((CVPixelBuffer, Double) -> Void)?

    private let sessionQueue = DispatchQueue(label: "camera.session")
    private let videoQueue = DispatchQueue(label: "camera.frames")
    private let output = AVCaptureVideoDataOutput()
    private var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
    private var rotationObservation: NSKeyValueObservation?

    enum CameraError: Error {
        case permissionDenied
        case noCamera
    }

    static func requestPermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .video)
        default: return false
        }
    }

    /// Configure for 1080p at the highest frame rate the format supports
    /// (60 fps preferred). Returns the configured frame rate.
    func configure() throws -> Double {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video, position: .back) else {
            throw CameraError.noCamera
        }

        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .hd1920x1080

        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw CameraError.noCamera }
        session.addInput(input)

        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: videoQueue)
        guard session.canAddOutput(output) else { throw CameraError.noCamera }
        session.addOutput(output)

        // Prefer 60 fps if the active format supports it.
        var fps = 30.0
        let ranges = device.activeFormat.videoSupportedFrameRateRanges
        if let best = ranges.map(\.maxFrameRate).max(), best >= 60 {
            try device.lockForConfiguration()
            device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 60)
            device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 60)
            device.unlockForConfiguration()
            fps = 60.0
        }

        previewLayer.session = session
        previewLayer.videoGravity = .resizeAspect

        // Drive BOTH the preview and the data-output connection from the same
        // rotation angle: buffer content then matches what the preview shows,
        // so the overlay's aspect-fit mapping is valid in any orientation.
        let coordinator = AVCaptureDevice.RotationCoordinator(device: device,
                                                              previewLayer: previewLayer)
        rotationCoordinator = coordinator
        applyRotation(coordinator.videoRotationAngleForHorizonLevelPreview)
        rotationObservation = coordinator.observe(
            \.videoRotationAngleForHorizonLevelPreview, options: [.new]
        ) { [weak self] _, change in
            guard let angle = change.newValue else { return }
            self?.applyRotation(angle)
        }

        return fps
    }

    private func applyRotation(_ angle: CGFloat) {
        for conn in [output.connection(with: .video), previewLayer.connection] {
            if let conn, conn.isVideoRotationAngleSupported(angle) {
                conn.videoRotationAngle = angle
            }
        }
    }

    func start() {
        sessionQueue.async { [self] in
            if !session.isRunning { session.startRunning() }
        }
    }

    func stop() {
        sessionQueue.async { [self] in
            if session.isRunning { session.stopRunning() }
        }
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let t = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        onFrame?(pixelBuffer, t)
    }
}
