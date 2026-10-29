// CameraSession.swift
//
// The back wide camera at 4K 30 fps, plus the microphone, handing every
// frame and audio buffer to `onVideo` / `onAudio` on one serial queue.
//
// Set up for measurement, not for pretty video:
//   - stabilisation OFF: it crops and shifts the picture frame to frame,
//     which would move the court under the tapped corners
//   - zoom 1x on the wide lens, never the ultra-wide (different geometry)
//   - late frames are NOT discarded: the master recording must have every
//     frame, and the analysis clock counts them
//   - the UI is locked to landscape-right, the sensor's own orientation, so
//     buffers need no rotation and preview points map straight to pixels
import AVFoundation
import UIKit

final class CameraSession: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate,
                           AVCaptureAudioDataOutputSampleBufferDelegate {
    static let width = 3840, height = 2160, fps = 30.0

    let session = AVCaptureSession()
    let queue = DispatchQueue(label: "anya.capture", qos: .userInitiated)
    var onVideo: ((CMSampleBuffer) -> Void)?
    var onAudio: ((CMSampleBuffer) -> Void)?

    private let videoOutput = AVCaptureVideoDataOutput()
    private let audioOutput = AVCaptureAudioDataOutput()
    private var configured = false

    enum SetupError: LocalizedError {
        case cameraPermission, noCamera, no4K
        var errorDescription: String? {
            switch self {
            case .cameraPermission: "Anya needs the camera to record the match. Allow it in Settings."
            case .noCamera: "No back camera found."
            case .no4K: "This iPhone cannot record 4K at 30 fps."
            }
        }
    }

    static func requestPermissions() async -> Bool {
        let video: Bool
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: video = true
        case .notDetermined: video = await AVCaptureDevice.requestAccess(for: .video)
        default: video = false
        }
        // The microphone is optional: without it the recording is silent.
        if AVCaptureDevice.authorizationStatus(for: .audio) == .notDetermined {
            _ = await AVCaptureDevice.requestAccess(for: .audio)
        }
        return video
    }

    func configure() throws {
        guard !configured else { return }
        guard AVCaptureDevice.authorizationStatus(for: .video) == .authorized else { throw SetupError.cameraPermission }
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            throw SetupError.noCamera
        }
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        guard session.canSetSessionPreset(.hd4K3840x2160) else { throw SetupError.no4K }
        session.sessionPreset = .hd4K3840x2160
        let input = try AVCaptureDeviceInput(device: camera)
        guard session.canAddInput(input) else { throw SetupError.noCamera }
        session.addInput(input)

        try camera.lockForConfiguration()
        camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: CMTimeScale(Self.fps))
        camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: CMTimeScale(Self.fps))
        camera.videoZoomFactor = 1
        camera.unlockForConfiguration()

        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
                                        kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        videoOutput.alwaysDiscardsLateVideoFrames = false
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(videoOutput) else { throw SetupError.noCamera }
        session.addOutput(videoOutput)
        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoStabilizationSupported { connection.preferredVideoStabilizationMode = .off }
            if connection.isVideoRotationAngleSupported(0) { connection.videoRotationAngle = 0 }
        }

        if AVCaptureDevice.authorizationStatus(for: .audio) == .authorized,
           let microphone = AVCaptureDevice.default(for: .audio),
           let audioInput = try? AVCaptureDeviceInput(device: microphone),
           session.canAddInput(audioInput), session.canAddOutput(audioOutput) {
            session.addInput(audioInput)
            audioOutput.setSampleBufferDelegate(self, queue: queue)
            session.addOutput(audioOutput)
        }
        configured = true
    }

    /// Set the frame handlers on the capture queue, so a frame in flight
    /// never sees them half-changed. Returns once the switch is done.
    func setHandlers(video: ((CMSampleBuffer) -> Void)?, audio: ((CMSampleBuffer) -> Void)?) {
        queue.sync {
            onVideo = video
            onAudio = audio
        }
    }

    func start() { queue.async { [session] in if !session.isRunning { session.startRunning() } } }
    func stop() { queue.async { [session] in if session.isRunning { session.stopRunning() } } }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        if output === videoOutput { onVideo?(sampleBuffer) } else { onAudio?(sampleBuffer) }
    }
}
