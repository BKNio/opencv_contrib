/*///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "tldTracker.hpp"


namespace cv
{

Ptr<TrackerTLD> TrackerTLD::createTracker(const TrackerTLD::Params &parameters)
{
    return makePtr<tld::TrackerTLDImpl>(parameters);
}

namespace tld
{

TrackerTLDImpl::TrackerTLDImpl(const TrackerTLD::Params &parameters) :
    params(parameters), isTrackerOK(false)
{
    isInit = false;
    medianFlow = TrackerMedianFlow::createTracker();
    cascadeClassifier =  makePtr<CascadeClassifier>
                (
                parameters.preFernMeasurements, parameters.preFerns, parameters.preFernPatchSize,
                parameters.numberOfMeasurements, parameters.numberOfFerns, parameters.fernPatchSize,
                parameters.numberOfExamples, parameters.examplePatchSize,
                parameters.groupRectanglesTheta
                );
}

void TrackerTLDImpl::read(const cv::FileNode& fn)
{
    params.read(fn);
}

void TrackerTLDImpl::write(cv::FileStorage& fs) const
{
    params.write(fs);
}

bool TrackerTLDImpl::initImpl(const Mat& image, const Rect2d& boundingBox)
{

    model = makePtr<TrackerTLDModel>();
    Mat actImage;

    isTrackerOK = true;

    if(image.type() == CV_8UC3)
        cvtColor(image, actImage, COLOR_BGR2GRAY);
    else if(image.type() == CV_8U)
        image.copyTo(actImage);
    else
        CV_Error(Error::StsBadArg, "wrong input image type: should be CV_8UC3 or CV_8U");

    pExpert = makePtr<PExpert>(image.size());
    nExpert = makePtr<NExpert>();

    medianFlow->init(actImage, boundingBox);
    cascadeClassifier->init(actImage, boundingBox, pExpert->generatePositiveExamples(image, boundingBox, 5, 30));

    return true;

}

#define DEBUG_OUTPUT

bool TrackerTLDImpl::updateImpl(const Mat& image, Rect2d& boundingBox)
{
    Mat actImage;

    if(image.type() == CV_8UC3)
        cvtColor(image, actImage, COLOR_BGR2GRAY);
    else if(image.type() == CV_8U)
        image.copyTo(actImage);
    else
        CV_Error(Error::StsBadArg, "wrong input image type: should be CV_8UC3 or CV_8U");

    bool isDetected = true;

    std::vector<Rect> detections = cascadeClassifier->detect(actImage);

#ifdef DEBUG_OUTPUT
    Mat debugOutput;
    cvtColor(actImage, debugOutput, COLOR_GRAY2BGR);

    if(!detections.empty())
        rectangle(debugOutput, detections.front(), Scalar(255,0,0), 2);
#endif

    Rect2d objectFromTracker;
    double trackerConfidence;
    if(isTrackerOK)
        isTrackerOK = medianFlow->update(actImage, objectFromTracker);

    if(isTrackerOK)
        trackerConfidence = cascadeClassifier->nnClassifier->calcConfidence(image(objectFromTracker));

#ifdef DEBUG_OUTPUT
    if(isTrackerOK)
        rectangle(debugOutput, objectFromTracker, Scalar(0, 255, 0), 2);
#endif

    if(!detections.empty() && isTrackerOK)
    {
        double maxOverlap= 0.;
        Rect bestDetectedRect;
        for(size_t i = 0; i < detections.size(); ++i)
        {
            const Rect &actRect = detections[i];
            const double actOverlap = overlap(actRect, objectFromTracker);

            if(actOverlap > maxOverlap)
            {
                maxOverlap = actOverlap;
                bestDetectedRect = actRect;
            }

        }

        if(maxOverlap > 0.5)
        {
            boundingBox = Rect( cvRound((bestDetectedRect.x + objectFromTracker.x) / 2),
                                cvRound((bestDetectedRect.y + objectFromTracker.y) / 2),
                                cvRound((bestDetectedRect.width + objectFromTracker.width) / 2),
                                cvRound((bestDetectedRect.height + objectFromTracker.height) / 2)
                                );

            cascadeClassifier->addPositiveExamples(pExpert->generatePositiveExamples(actImage, boundingBox, 5, 5));
            cascadeClassifier->addNegativeExamples(nExpert->getNegativeExamples(actImage, boundingBox, detections));

#ifdef DEBUG_OUTPUT
            rectangle(debugOutput, boundingBox, Scalar(255, 0, 139), 2);
#endif
        }
        else
        {
            if(cascadeClassifier->nnClassifier->calcConfidence(actImage(objectFromTracker)) >=
                    cascadeClassifier->nnClassifier->calcConfidence(actImage(detections.front())))
            {
                boundingBox = objectFromTracker;
            }
            else
            {
                boundingBox = detections.front();
                medianFlow = TrackerMedianFlow::createTracker();
                isTrackerOK = medianFlow->init(actImage, detections.front());
            }

#ifdef DEBUG_OUTPUT
            rectangle(debugOutput, boundingBox, Scalar(0, 0, 255));
#endif

        }
    }
    else if(isTrackerOK && trackerConfidence > 0.5)
    {
            cascadeClassifier->addPositiveExamples(pExpert->generatePositiveExamples(actImage, objectFromTracker, 5, 5));
    }
    else if(!detections.empty())
    {
        double detectorConfidence = cascadeClassifier->nnClassifier->calcConfidence(actImage(detections.front()));
        if(detectorConfidence > 0.5)
        {
            medianFlow = TrackerMedianFlow::createTracker();
            isTrackerOK = medianFlow->init(actImage, detections.front());
        }
    }
    else
        isDetected = false;

    imshow("debugoutput", debugOutput);
    waitKey(1);

    return isDetected;
}

std::vector< Mat_<uchar> > TrackerTLDImpl::PExpert::generatePositiveExamples(const Mat_<uchar> &image, const Rect &bb, int numberOfsurroundBbs, int numberOfSyntheticWarped)
{
    const float shiftRangePercent = .01f;
    const float scaleRange = .01f;
    const float angleRangeDegree = 10.f;

    const float shiftXRange = shiftRangePercent * bb.width;
    const float shiftYRange = shiftRangePercent * bb.height;

    std::vector< Mat_<uchar> > positiveExamples;

    positiveExamples.push_back(image(bb));

    std::vector<Rect> nClosestRects = generateClosestN(bb, numberOfsurroundBbs);

    for(std::vector<Rect>::const_iterator positiveRect = nClosestRects.begin(); positiveRect != nClosestRects.end(); ++positiveRect)
    {
        const std::vector<float> &rotationRandomValues = generateRandomValues(angleRangeDegree, numberOfSyntheticWarped);
        const std::vector<float> &scaleRandomValues = generateRandomValues(scaleRange, numberOfSyntheticWarped);
        const std::vector<float> &shiftXRandomValues = generateRandomValues(shiftXRange, numberOfSyntheticWarped);
        const std::vector<float> &shiftYRandomValues = generateRandomValues(shiftYRange, numberOfSyntheticWarped);

        for(int index = 0; index < numberOfSyntheticWarped; ++index)
        {
            Mat_<uchar> warpedOOI = getWarped(image, *positiveRect, shiftXRandomValues[index], shiftYRandomValues[index], scaleRandomValues[index], rotationRandomValues[index]);

            for(int j = 0; j < warpedOOI.rows; ++j)
                for(int i = 0; i < warpedOOI.cols; ++i)
                    warpedOOI.at<uchar>(j,i) += rng.gaussian(5.);

            positiveExamples.push_back(warpedOOI);
        }

        positiveExamples.push_back(image(*positiveRect).clone());
    }

    return positiveExamples;

}

bool TrackerTLDImpl::PExpert::isRectOK(const Rect &rect) const
{
    return rect.tl().x >= 0 && rect.tl().y >= 0 && rect.br().x <= frameSize.width && rect.br().y <= frameSize.height;
}

std::vector<Rect> TrackerTLDImpl::PExpert::generateClosestN(const Rect &bBox, int n) const
{
    CV_Assert(n >= 0);

    const float rangeStart = 1.f;
    const float rangeEnd = 0.75f;

    std::vector<Rect> ret; ret.reserve(n);

    if(n == 0)
        return ret;

    const float dx = float(bBox.width) / 10;
    const float dy = float(bBox.height) / 10;

    const Point tl = bBox.tl();
    Vec2f sumUp(0.f, 0.f);

    std::multimap<float, Rect> storage;

    for(int stepX = -n; stepX <= n; ++stepX)
    {
        for(int stepY = -n; stepY <= n; ++stepY)
        {
            const Rect actRect(Point(cvRound(tl.x + dx*stepX), cvRound(tl.y + dy*stepY)), bBox.size());

            sumUp += Vec2f(actRect.tl().x, actRect.tl().y);

            const Rect overlap = bBox & actRect;

            const float overlapValue = float(overlap.area()) / (actRect.area() + bBox.area() - overlap.area());

            storage.insert(std::make_pair(overlapValue, actRect));
        }
    }

    std::multimap<float, Rect>::iterator closestIt = storage.lower_bound(rangeStart);

    CV_Assert(closestIt != storage.end());
    CV_Assert(closestIt != storage.begin());

    for(; closestIt != storage.begin(); --closestIt)
    {
        if(closestIt->first <= rangeStart && closestIt->first > rangeEnd)
            if(isRectOK(closestIt->second))
            {
                ret.push_back(closestIt->second);
                if(ret.size() == size_t(n))
                    break;
            }
    }
    return ret;
}

std::vector<float> TrackerTLDImpl::PExpert::generateRandomValues(float range, int quantity) const
{
    std::vector<float> values;

    for(int i = 0; i < quantity; ++i)
        values.push_back(rng.uniform(-range, range));

    float accum = std::accumulate(values.begin(), values.end(), 0.f);

    accum /= quantity;

    for(int i = 0; i < quantity; ++i)
        values[i] -= accum;

    return values;
}

Mat_<uchar> TrackerTLDImpl::PExpert::getWarped(const Mat_<uchar> &originalFrame, Rect bb, float shiftX, float shiftY, float scale, float rotation)
{

    Mat shiftTransform = cv::Mat::eye(3, 3, CV_32F);
    shiftTransform.at<float>(0,2) = shiftX;
    shiftTransform.at<float>(1,2) = shiftY;

    Mat scaleTransform = cv::Mat::eye(3, 3, CV_32F);
    scaleTransform.at<float>(0,0) = 1 - scale;
    scaleTransform.at<float>(1,1) = scaleTransform.at<float>(0,0);

    Mat rotationShiftTransform = cv::Mat::eye(3, 3, CV_32F);
    rotationShiftTransform.at<float>(0,2) = bb.tl().x + float(bb.width) / 2;
    rotationShiftTransform.at<float>(1,2) = bb.tl().y + float(bb.height) / 2;

    const float angle = (rotation * CV_PI) / 180.f;

    Mat rotationTransform = cv::Mat::eye(3, 3, CV_32F);
    rotationTransform.at<float>(0,0) = rotationTransform.at<float>(1,1) = std::cos(angle);
    rotationTransform.at<float>(0,1) = std::sin(angle);
    rotationTransform.at<float>(1,0) = - rotationTransform.at<float>(0,1);

    const Mat resultTransform = rotationShiftTransform * rotationTransform * rotationShiftTransform.inv() * scaleTransform * shiftTransform;;

    Mat_<uchar> dst;
    warpAffine(originalFrame, dst, resultTransform(cv::Rect(0,0,3,2)), dst.size());

    return dst(bb);
}

std::vector<Mat_<uchar> > TrackerTLDImpl::NExpert::getNegativeExamples(const Mat_<uchar> &image, const Rect &object, std::vector<Rect> &detectedObjects)
{
    std::vector< Mat_<uchar> > negativeExamples;

    for(size_t i = 0; i < detectedObjects.size(); ++i)
    {
        const Rect &actDetectedObject = detectedObjects[i];

        if(overlap(actDetectedObject, object) < 0.5)
            negativeExamples.push_back(image(actDetectedObject).clone());
    }

    return negativeExamples;
}
}

}
