import XCTest
@testable import Anya2Engine

final class NumpyCompatTests: XCTestCase {
    func testPythonRoundIsHalfToEven() {
        XCTAssertEqual(pythonRound(10.5), 10)
        XCTAssertEqual(pythonRound(11.5), 12)
        XCTAssertEqual(pythonRound(0.7 * 15), 10)   // READY_WIN_S at 15 Hz
        XCTAssertEqual(pythonRound(-0.5), 0)
    }

    func testConvolveSameCentresEvenWidthsLeft() {
        // numpy: np.convolve([1,2,3,4,5], np.ones(4), "same") == [3, 6, 10, 14, 12]
        XCTAssertEqual(convolveSameOnes([1, 2, 3, 4, 5], width: 4), [3, 6, 10, 14, 12])
    }

    func testPercentileMatchesNumpyLinear() {
        // numpy: np.percentile([3, 1, 4, 1, 5, 9, 2, 6], 90) == 6.9
        XCTAssertEqual(percentile([3, 1, 4, 1, 5, 9, 2, 6], 90), 6.9, accuracy: 1e-12)
        XCTAssertEqual(median([3, 1, 4, 1]), 2)
    }

    func testFmaxIgnoresNaN() {
        XCTAssertEqual(fmaxNaN(.nan, 2), 2)
        XCTAssertTrue(fmaxNaN(.nan, .nan).isNaN)
    }
}
