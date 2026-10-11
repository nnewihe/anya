// ProxyRecorder.swift
//
// What analysis reads: the pose-sample frames (every 2nd frame at 30 fps =
// 15 Hz) as two small H.264 streams, cut into 2-second segment files:
//   proxies/near-00001.mp4   960x540 whole frame
//   proxies/far-00001.mp4    the far band, native resolution
// Segments let analysis run behind the camera without holding frames in
// memory, survive a kill (finished segments are on disk), and are the same
// pictures Python's _proxy540.mp4 / _farband.mp4 hold, so a recorded match
// can be replayed on the desktop.
//
// Frames are placed on the timeline by their presentation time, not by
// counting: if the camera ever drops a frame, later frames keep their true
// source-frame numbers, and a missed sample is filled with the next frame.
import AVFoundation

final class ProxyRecorder {
    static let samplesPerSegment = 30            // 2 s at 15 Hz

    let poseStride: Int                          // source frames per pose sample
    let folder: URL
    /// Called (on the capture queue) when a segment file pair is complete.
    var onSegment: ((MatchManifest.Segment) -> Void)?

    private let scaler: FrameScaler
    private let sourceFPS: Double
    private let firstSourceFrame: Int            // where this part starts on the timeline
    private var firstPTS: CMTime?
    private var nextSample: Int                  // next pose-sample source frame wanted
    private var nextSegmentIndex: Int
    private var current: (index: Int, firstSourceFrame: Int, count: Int, near: SegmentWriter, far: SegmentWriter)?
    private var pendingFinishes: [Task<Void, Never>] = []

    init(folder: URL, scaler: FrameScaler, sourceFPS: Double, poseStride: Int, firstSourceFrame: Int,
         firstSegmentIndex: Int) {
        self.folder = folder; self.scaler = scaler; self.sourceFPS = sourceFPS
        self.poseStride = poseStride; self.firstSourceFrame = firstSourceFrame
        nextSample = firstSourceFrame
        nextSegmentIndex = firstSegmentIndex
    }

    /// The timeline frame of a camera frame.
    func sourceFrame(of sample: CMSampleBuffer) -> Int {
        let pts = CMSampleBufferGetPresentationTimeStamp(sample)
        if firstPTS == nil { firstPTS = pts }
        return firstSourceFrame + Int(((pts - firstPTS!).seconds * sourceFPS).rounded())
    }

    /// Feed every camera frame, in order.
    func append(_ sample: CMSampleBuffer, sourceFrame: Int) {
        guard sourceFrame >= nextSample, let pixels = CMSampleBufferGetImageBuffer(sample) else { return }
        guard let near = scaler.near(pixels), let far = scaler.far(pixels) else { return }
        // Normally one sample; more only if the camera dropped the wanted frame.
        while nextSample <= sourceFrame {
            write(near: near, far: far, sampleFrame: nextSample)
            nextSample += poseStride
        }
    }

    /// Close the open segment and wait for every segment to be on disk.
    func finish() async {
        closeSegment()
        for task in pendingFinishes { await task.value }
        pendingFinishes.removeAll()
    }

    private func write(near: CVPixelBuffer, far: CVPixelBuffer, sampleFrame: Int) {
        if current == nil {
            let index = nextSegmentIndex
            nextSegmentIndex += 1
            let nearName = String(format: "proxies/near-%05d.mp4", index)
            let farName = String(format: "proxies/far-%05d.mp4", index)
            guard let nearWriter = try? SegmentWriter(url: folder.appendingPathComponent(nearName),
                                                      width: 960, height: 540, bitRate: 8_000_000),
                  let farWriter = try? SegmentWriter(url: folder.appendingPathComponent(farName),
                                                     width: scaler.band.width, height: scaler.band.height,
                                                     bitRate: 10_000_000)
            else { return }
            current = (index, sampleFrame, 0, nearWriter, farWriter)
        }
        guard var segment = current else { return }
        let time = CMTime(value: CMTimeValue(segment.count), timescale: 15)
        segment.near.append(near, at: time)
        segment.far.append(far, at: time)
        segment.count += 1
        current = segment
        if segment.count == Self.samplesPerSegment { closeSegment() }
    }

    private func closeSegment() {
        guard let segment = current else { return }
        current = nil
        let record = MatchManifest.Segment(
            index: segment.index, firstSourceFrame: segment.firstSourceFrame, count: segment.count,
            nearFile: String(format: "proxies/near-%05d.mp4", segment.index),
            farFile: String(format: "proxies/far-%05d.mp4", segment.index))
        let callback = onSegment
        pendingFinishes.append(Task {
            await segment.near.finish()
            await segment.far.finish()
            callback?(record)
        })
    }
}

/// One H.264 file of BGRA frames at fixed timestamps.
final class SegmentWriter {
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor

    init(url: URL, width: Int, height: Int, bitRate: Int) throws {
        writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: bitRate,
                                              AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel],
        ])
        input.expectsMediaDataInRealTime = true
        adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: nil)
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
    }

    func append(_ frame: CVPixelBuffer, at time: CMTime) {
        // Real-time input: if the encoder is momentarily busy, wait briefly
        // rather than lose a pose sample.
        var waited = 0
        while !input.isReadyForMoreMediaData && waited < 50 { usleep(1000); waited += 1 }
        adaptor.append(frame, withPresentationTime: time)
    }

    func finish() async {
        input.markAsFinished()
        await writer.finishWriting()
    }
}
