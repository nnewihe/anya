// The live pipeline must be the batch pipeline, delivered early.
//
// For every corpus clip the pose samples are fed in random-sized chunks (a
// seeded generator, so a failure reproduces), and:
//   - everything committed part-way through must be unchanged at the end
//     (committed means final);
//   - finish() must equal runBatch on the same detections, exactly.
import XCTest
@testable import Anya2Engine

final class LiveAnalysisTests: XCTestCase {
    /// SplitMix64: a tiny seeded generator, so chunk sizes are reproducible.
    struct SeededGenerator: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
    }

    func testLiveEqualsBatchUnderRandomChunking() throws {
        let model = try WalkingModel.bundled()
        for clip in try CorpusParityTests().corpusClips() {
            let name = clip.meta.clip
            let near = clip.poseDetections("near"), far = clip.poseDetections("far")
            let batch = runBatch(near: near, far: far, geometry: clip.geometry, walkingModel: model)

            let live = LiveAnalysis(fps: near.fps, stride: near.stride, geometry: clip.geometry, walkingModel: model)
            var generator = SeededGenerator(state: UInt64(Int(name) ?? 0))
            let total = min(near.samples, far.samples)
            var next = 0, snapshotTaken = false
            var midWalking: [[Double]] = [], midLive: [[Double]] = [], midTrophy: [Double] = []
            while next < total {
                let chunk = Int.random(in: 1...300, using: &generator)
                let samples = (next..<min(total, next + chunk)).map { PooledSample(near: near, far: far, sample: $0) }
                live.append(samples)
                next += samples.count
                // Halfway through, remember what has been committed so far.
                if !snapshotTaken && next >= total / 2 {
                    midWalking = live.walkingProbability; midLive = live.smoothedLive
                    midTrophy = live.nearServe[0]!.trophy
                    snapshotTaken = true
                    XCTAssertGreaterThan(live.samplesFinal, 0, "\(name): nothing final mid-match")
                    XCTAssertLessThanOrEqual(next - live.samplesFinal,
                                             LiveLookahead.FEATURES + LiveLookahead.LIVE_SMOOTHING,
                                             "\(name): live score lagging more than the lookahead")
                }
            }
            let result = live.finish()

            // Committed values never changed.
            for lead in 0..<2 {
                XCTAssertEqual(Array(live.walkingProbability[lead].prefix(midWalking[lead].count)), midWalking[lead],
                               "\(name): committed walking probability changed")
                XCTAssertEqual(Array(live.smoothedLive[lead].prefix(midLive[lead].count)), midLive[lead],
                               "\(name): committed live score changed")
            }
            XCTAssertTrue(zip(live.nearServe[0]!.trophy, midTrophy).allSatisfy { $0 == $1 || ($0.isNaN && $1.isNaN) },
                          "\(name): committed trophy score changed")

            // ...and the whole result is the batch result, exactly.
            assertSame(result.endSignals.walkingProbability, batch.endSignals.walkingProbability, "\(name) walking")
            for signal in NearEndRules.SIGNAL_NAMES {
                assertSame(result.endSignals.nearEnd[signal]!, batch.endSignals.nearEnd[signal]!, "\(name) \(signal)")
            }
            assertSame(result.live, batch.live, "\(name) live score")
            XCTAssertEqual(result.nearServes, batch.nearServes, "\(name) near serves")
            XCTAssertEqual(result.farServes, batch.farServes, "\(name) far serves")
            XCTAssertEqual(result.pointEnds, batch.pointEnds, "\(name) point ends")
            XCTAssertEqual(result.reel.segments.count, batch.reel.segments.count, "\(name) segment count")
            for (a, b) in zip(result.reel.segments, batch.reel.segments) {
                XCTAssertEqual([a.start, a.stop, a.serveT, a.endT], [b.start, b.stop, b.serveT, b.endT], "\(name) segment")
                XCTAssertEqual(a.side, b.side); XCTAssertEqual(a.endSource, b.endSource); XCTAssertEqual(a.notes, b.notes)
            }
        }
    }
}
