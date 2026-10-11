// Anya2OpenCV.mm -- see Anya2OpenCV.h.
#import "Anya2OpenCV.h"

#import <opencv2/opencv.hpp>
#import <opencv2/features2d.hpp>
#import <opencv2/calib3d.hpp>
#import <opencv2/flann.hpp>

@implementation A2Features
- (instancetype)initWithCount:(NSInteger)count points:(NSData *)points descriptors:(NSData *)descriptors {
    if ((self = [super init])) { _count = count; _points = points; _descriptors = descriptors; }
    return self;
}
@end

@implementation A2HomographyFit
- (instancetype)initWithMatrix:(NSData *)matrix inliers:(NSData *)inliers {
    if ((self = [super init])) { _matrix = matrix; _inliers = inliers; }
    return self;
}
@end

static cv::Mat descriptorMat(NSData *data) {
    return cv::Mat((int)(data.length / 32), 32, CV_8U, (void *)data.bytes);
}

static cv::Mat pointMat(NSData *data, int type) {
    size_t width = type == CV_32FC2 ? sizeof(float) * 2 : sizeof(double) * 2;
    return cv::Mat((int)(data.length / width), 1, type, (void *)data.bytes);
}

static NSData *matrixData(const cv::Mat &m) {
    cv::Mat as64;
    m.convertTo(as64, CV_64F);
    return [NSData dataWithBytes:as64.ptr<double>() length:9 * sizeof(double)];
}

@implementation A2OpenCV

+ (A2Features *)orbFeaturesOfGray:(const uint8_t *)gray width:(NSInteger)width height:(NSInteger)height
                      bytesPerRow:(NSInteger)bytesPerRow maxFeatures:(NSInteger)maxFeatures {
    // An OpenCV assertion throws; never let it end the app.
    try {
        cv::Mat image((int)height, (int)width, CV_8U, (void *)gray, (size_t)bytesPerRow);
        cv::Ptr<cv::ORB> orb = cv::ORB::create((int)maxFeatures);
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
        if (descriptors.empty()) return nil;
        NSMutableData *points = [NSMutableData dataWithLength:keypoints.size() * 2 * sizeof(float)];
        float *p = (float *)points.mutableBytes;
        for (size_t i = 0; i < keypoints.size(); i++) { p[2 * i] = keypoints[i].pt.x; p[2 * i + 1] = keypoints[i].pt.y; }
        cv::Mat packed = descriptors.isContinuous() ? descriptors : descriptors.clone();
        return [[A2Features alloc] initWithCount:(NSInteger)keypoints.size() points:points
                                     descriptors:[NSData dataWithBytes:packed.data length:packed.total()]];
    } catch (const cv::Exception &) {
        return nil;
    }
}

+ (NSData *)ratioMatchesOfQuery:(NSData *)query train:(NSData *)train approximate:(BOOL)approximate
                          ratio:(double)ratio {
    // An OpenCV assertion throws; never let it end the app.
    try {
        std::vector<std::vector<cv::DMatch>> pairs;
        cv::Mat q = descriptorMat(query), t = descriptorMat(train);
        if (approximate) {
            cv::FlannBasedMatcher matcher(cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1),
                                          cv::makePtr<cv::flann::SearchParams>(32));
            matcher.knnMatch(q, t, pairs, 2);
        } else {
            cv::BFMatcher matcher(cv::NORM_HAMMING);
            matcher.knnMatch(q, t, pairs, 2);
        }
        NSMutableData *out = [NSMutableData data];
        for (const auto &pair : pairs) {
            if (pair.size() < 2) continue;
            if (pair[0].distance < ratio * pair[1].distance) {
                int32_t indices[2] = {pair[0].queryIdx, pair[0].trainIdx};
                [out appendBytes:indices length:sizeof(indices)];
            }
        }
        return out;
    } catch (const cv::Exception &) {
        return [NSData data];
    }
}

+ (A2HomographyFit *)ransacHomographyFrom:(NSData *)source to:(NSData *)destination
                                threshold:(double)threshold maxIterations:(NSInteger)maxIterations
                               confidence:(double)confidence {
    // An OpenCV assertion throws; never let it end the app.
    try {
        cv::Mat mask;
        cv::Mat H = cv::findHomography(pointMat(source, CV_32FC2), pointMat(destination, CV_32FC2), cv::RANSAC,
                                       threshold, mask, (int)maxIterations, confidence);
        if (H.empty() || mask.empty()) return nil;
        cv::Mat packed = mask.isContinuous() ? mask : mask.clone();
        return [[A2HomographyFit alloc] initWithMatrix:matrixData(H)
                                               inliers:[NSData dataWithBytes:packed.data length:packed.total()]];
    } catch (const cv::Exception &) {
        return nil;
    }
}

+ (NSData *)leastSquaresHomographyFrom:(NSData *)source to:(NSData *)destination {
    // An OpenCV assertion throws; never let it end the app.
    try {
        cv::Mat H = cv::findHomography(pointMat(source, CV_64FC2), pointMat(destination, CV_64FC2), 0);
        return H.empty() ? nil : matrixData(H);
    } catch (const cv::Exception &) {
        return nil;
    }
}

+ (NSData *)courtMaskWithCorners:(NSData *)corners width:(NSInteger)width height:(NSInteger)height
                        padPixels:(NSInteger)pad {
    // An OpenCV assertion throws; never let it end the app.
    try {
        if (corners.length != 4 * 2 * sizeof(int32_t) || width <= 0 || height <= 0) return [NSData data];
        cv::Mat mask = cv::Mat::zeros((int)height, (int)width, CV_8U);
        cv::Mat quad(4, 1, CV_32SC2, (void *)corners.bytes);
        cv::fillConvexPoly(mask, quad, cv::Scalar(255));
        int k = 2 * (int)pad + 1;
        cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k)));
        return [NSData dataWithBytes:mask.data length:mask.total()];
    } catch (const cv::Exception &) {
        return [NSData data];
    }
}

@end
