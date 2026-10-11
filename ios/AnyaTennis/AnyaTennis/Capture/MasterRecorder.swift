// MasterRecorder.swift
//
// The full-quality recording of one part: 4K HEVC + AAC in a .mov.
// Fragmented every 2 s (movieFragmentInterval): if the app is killed, the
// file is still playable up to the last fragment, so a crash costs seconds
// of the match, not the match.
import AVFoundation

final class MasterRecorder {
    let url: URL
    private let writer: AVAssetWriter
    private let video: AVAssetWriterInput
    private let audio: AVAssetWriterInput?
    private var started = false
    private(set) var framesWritten = 0

    init(url: URL, width: Int, height: Int, withAudio: Bool) throws {
        self.url = url
        writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        writer.movieFragmentInterval = CMTime(seconds: 2, preferredTimescale: 600)
        video = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width, AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: 40_000_000,
                // A key frame every second keeps passthrough cuts tight.
                AVVideoMaxKeyFrameIntervalKey: 30,
                AVVideoExpectedSourceFrameRateKey: 30,
            ],
        ])
        video.expectsMediaDataInRealTime = true
        writer.add(video)
        if withAudio {
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC, AVNumberOfChannelsKey: 1,
                AVSampleRateKey: 48_000, AVEncoderBitRateKey: 96_000,
            ])
            input.expectsMediaDataInRealTime = true
            writer.add(input)
            audio = input
        } else {
            audio = nil
        }
    }

    /// Returns false when the encoder could not keep up and the frame was
    /// lost (counted, so the timeline still advances).
    @discardableResult
    func append(video sample: CMSampleBuffer) -> Bool {
        if !started {
            guard writer.startWriting() else { return false }
            writer.startSession(atSourceTime: CMSampleBufferGetPresentationTimeStamp(sample))
            started = true
        }
        framesWritten += 1
        guard video.isReadyForMoreMediaData else { return false }
        return video.append(sample)
    }

    func append(audio sample: CMSampleBuffer) {
        guard started, let audio, audio.isReadyForMoreMediaData else { return }
        audio.append(sample)
    }

    func finish() async {
        guard started, writer.status == .writing else { return }
        video.markAsFinished()
        audio?.markAsFinished()
        await writer.finishWriting()
    }

    var error: Error? { writer.error }
}
