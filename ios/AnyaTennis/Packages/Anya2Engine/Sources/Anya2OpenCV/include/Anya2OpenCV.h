// Anya2OpenCV.h
//
// The few OpenCV calls camera registration needs (camera.py: ORB features,
// k-NN matching with the ratio test, RANSAC and least-squares homographies),
// behind a plain Objective-C interface so Swift never sees C++.
// Every call takes and returns plain buffers; no OpenCV type crosses over.
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// ORB keypoints of one greyscale frame: positions (x, y float32 pairs) and
/// 32-byte descriptors, in OpenCV's order.
@interface A2Features : NSObject
@property (nonatomic, readonly) NSInteger count;
@property (nonatomic, readonly) NSData *points;        // float32 [count, 2]
@property (nonatomic, readonly) NSData *descriptors;   // uint8 [count, 32]
@end

/// A homography fit: row-major 3x3 (float64), and which pairs were inliers.
@interface A2HomographyFit : NSObject
@property (nonatomic, readonly) NSData *matrix;        // float64 [9]
@property (nonatomic, readonly) NSData *inliers;       // uint8 [pairs]
@end

@interface A2OpenCV : NSObject

/// cv2.ORB_create(nfeatures).detectAndCompute(gray, None).
/// nil when there are no descriptors.
+ (nullable A2Features *)orbFeaturesOfGray:(const uint8_t *)gray width:(NSInteger)width height:(NSInteger)height
                               bytesPerRow:(NSInteger)bytesPerRow maxFeatures:(NSInteger)maxFeatures;

/// knnMatch(query, train, k=2) and Lowe's ratio test. `approximate`: FLANN
/// LSH (table_number 6, key_size 12, multi_probe_level 1, checks 32), else
/// brute-force Hamming. Returns int32 pairs [n, 2] (queryIdx, trainIdx).
+ (NSData *)ratioMatchesOfQuery:(NSData *)query train:(NSData *)train approximate:(BOOL)approximate
                          ratio:(double)ratio;

/// cv2.findHomography(src, dst, RANSAC, threshold, maxIters, confidence).
/// float32 [n, 2] points. nil when OpenCV returns no fit.
+ (nullable A2HomographyFit *)ransacHomographyFrom:(NSData *)source to:(NSData *)destination
                                       threshold:(double)threshold maxIterations:(NSInteger)maxIterations
                                      confidence:(double)confidence;

/// cv2.findHomography(src, dst, 0): plain least squares. float64 [n, 2]
/// points. nil when OpenCV returns no fit.
+ (nullable NSData *)leastSquaresHomographyFrom:(NSData *)source to:(NSData *)destination;

/// camera._court_mask: the quad (int32 corners [4, 2]) filled with
/// cv2.fillConvexPoly, then dilated by a (2 pad + 1) square. uint8 [h, w], 0/255.
+ (NSData *)courtMaskWithCorners:(NSData *)corners width:(NSInteger)width height:(NSInteger)height
                        padPixels:(NSInteger)pad;

@end

NS_ASSUME_NONNULL_END
