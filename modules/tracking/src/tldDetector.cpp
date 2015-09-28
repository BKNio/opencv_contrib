/*M///////////////////////////////////////////////////////////////////////////////////////
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

#include <sys/time.h>

#include <map>
#include <numeric>

#include "tldDetector.hpp"

//#define SHOW_POSITIVE_EXAMPLE
//#define SHOW_ADDITIONAL_POSITIVE
//#define SHOW_ADDITIONAL_NEGATIVE

namespace cv
{
namespace tld
{

tldCascadeClassifier::tldCascadeClassifier(const Mat_<uchar> &zeroFrame, const Rect &bb, int maxNumberOfExamples, int numberOfMeasurements,
                                           int numberOfFerns, Size patchSize, int preMeasure, int preFerns, double actThreshold):
    minimalBBSize(20, 20), standardPatchSize(15, 15),originalBBSize(bb.size()),
    frameSize(zeroFrame.size()), scaleStep(1.2f), hypothesis(generateHypothesis(frameSize, originalBBSize, minimalBBSize, scaleStep))
{

    if(bb.width < minimalBBSize.width || bb.height < minimalBBSize.height)
        CV_Error(Error::StsBadArg, "Initial bounding box is too small");

    const Mat_<uchar> ooi = zeroFrame(bb);

    varianceClassifier = makePtr<tldVarianceClassifier>(ooi);
    preFernClassifier = makePtr<tldFernClassifier>(preMeasure, preFerns, standardPatchSize, actThreshold);
    fernClassifier = makePtr<tldFernClassifier>(numberOfMeasurements, numberOfFerns, patchSize);
    nnClassifier = makePtr<tldNNClassifier>(maxNumberOfExamples, standardPatchSize, 0.5);

    addSyntheticPositive(zeroFrame, bb, 5, 30);

#ifdef SHOW_POSITIVE_EXAMPLE
    imshow("positive example ooi", ooi);
    waitKey();
#endif



}

std::vector< std::pair<Rect, double> > tldCascadeClassifier::detect(const Mat_<uchar> &scaledImage) const
{
    timeval varianceStart, fernStart, nnStart, nnStop, mergeStop;

    answers.clear();
    gettimeofday(&varianceStart, NULL);
    varianceClassifier->isObjects(hypothesis, scaledImage, answers);
    gettimeofday(&fernStart, NULL);
    preFernClassifier->isObjects(hypothesis, scaledImage, answers);
    fernClassifier->isObjects(hypothesis, scaledImage, answers);
    gettimeofday(&nnStart, NULL);
    nnClassifier->isObjects(hypothesis, scaledImage, answers);
    gettimeofday(&nnStop, NULL);
    const std::vector< std::pair<Rect, double> > &result = prepareFinalResult(scaledImage);
    gettimeofday(&mergeStop, NULL);


//    std::cout << " var " << std::fixed << fernStart.tv_sec - varianceStart.tv_sec + double(fernStart.tv_usec - varianceStart.tv_usec) / 1e6;
//    std::cout << " fern " << std::fixed << nnStart.tv_sec - fernStart.tv_sec + double(nnStart.tv_usec - fernStart.tv_usec) / 1e6;
//    std::cout << " nn " << std::fixed << nnStop.tv_sec - nnStart.tv_sec + double(nnStop.tv_usec - nnStart.tv_usec) / 1e6 <<" "<< nnClassifier->positiveExamples.size() + nnClassifier->negativeExamples.size();
//    std::cout << " merge " << std::fixed << mergeStop.tv_sec - nnStop.tv_sec + double(mergeStop.tv_usec - nnStop.tv_usec) / 1e6;
//    std::cout << " total " << std::fixed << mergeStop.tv_sec - varianceStart.tv_sec + double(mergeStop.tv_usec - varianceStart.tv_usec) / 1e6 << std::endl;

    return result;
}

void tldCascadeClassifier::addSyntheticPositive(const Mat_<uchar> &image, const Rect bb, int numberOfsurroundBbs, int numberOfSyntheticWarped)
{
//    timeval start, stop;


//    gettimeofday(&start, NULL);

    const float shiftRangePercent = .01f;
    const float scaleRange = .01f;
    const float angleRangeDegree = 10.f;

    const float shiftXRange = shiftRangePercent * bb.width;
    const float shiftYRange = shiftRangePercent * bb.height;

    addPositiveExample(image(bb));

    std::vector<Rect> nClosestRects = generateClosestN(bb, numberOfsurroundBbs);

    Mat_<uchar> copy; image.copyTo(copy);
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

            addPositiveExample(warpedOOI);

//            imshow("warped", warpedOOI);
//            waitKey(0);
        }

        addPositiveExample(image(*positiveRect));
    }

//    gettimeofday(&stop, NULL);

//    std::cout << " add addSyntheticPositive " << std::fixed << stop.tv_sec - start.tv_sec + double(stop.tv_usec - start.tv_usec) / 1e6 << std::endl;

}

void tldCascadeClassifier::addPositiveExample(const Mat_<uchar> &example)
{    preFernClassifier->integratePositiveExample(example);
    fernClassifier->integratePositiveExample(example);
    nnClassifier->integratePositiveExample(example);
}

void tldCascadeClassifier::addNegativeExample(const Mat_<uchar> &example)
{
    preFernClassifier->integrateNegativeExample(example);
    fernClassifier->integrateNegativeExample(example);
    nnClassifier->integrateNegativeExample(example);
}

std::vector<Hypothesis> tldCascadeClassifier::generateHypothesis(const Size frameSize, const Size bbSize, const Size minimalBBSize, double scaleStep)
{
    std::vector<Hypothesis> hypothesis;

    const double scaleX = frameSize.width / bbSize.width;
    const double scaleY = frameSize.height / bbSize.height;

    const double scale = std::min(scaleX, scaleY);

    const double power =log(scale) / log(scaleStep);
    double correctedScale = pow(scaleStep, power);

    CV_Assert(int(bbSize.width * correctedScale) <= frameSize.width && int(bbSize.height * correctedScale) <= frameSize.height);

    for(;;)
    {
        Size currentBBSize(cvRound(bbSize.width * correctedScale), cvRound(bbSize.height * correctedScale));

        if(currentBBSize.width < minimalBBSize.width || currentBBSize.height < minimalBBSize.height)
            break;
        addScanGrid(frameSize, currentBBSize, minimalBBSize, hypothesis, correctedScale);

        correctedScale /= scaleStep;
    }
    return hypothesis;
}

std::vector<Rect> tldCascadeClassifier::generateClosestN(const Rect &bBox, size_t n) const
{
    return generateAndSelectRects(bBox, n, 1.0f, 0.75f);
}

std::vector<Rect> tldCascadeClassifier::generateSurroundingRects(const Rect &bBox, size_t n) const
{
    return generateAndSelectRects(bBox, n, 0.2f, 0.01f);
}

std::vector<Rect> tldCascadeClassifier::generateAndSelectRects(const Rect &bBox, int n, float rangeStart, float rangeEnd) const
{
    CV_Assert(n >= 0);

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

//    sumUp -= Vec2f(actRect.tl().x, actRect.tl().y);
//    sumUp /= 4*n*n;
//    std::cout << "shit " << Vec2f(tl.x, tl.y) - sumUp << std::endl;

    std::multimap<float, Rect>::iterator closestIt = storage.lower_bound(rangeStart);

    CV_Assert(closestIt != storage.end());
    CV_Assert(closestIt != storage.begin());

    for(; closestIt != storage.begin(); --closestIt)
    {
        if(closestIt->first <= rangeStart && closestIt->first > rangeEnd)
            if(isRectOK(closestIt->second))
            {
//                closestIt->second.x += sumUp[0];
//                closestIt->second.y += sumUp[1];
                ret.push_back(closestIt->second);
                if(ret.size() == n)
                    break;
            }
    }

//    ret.push_back(bBox);

//    while(ret.size() < n)
//    {
//        const float rdx = rng.uniform(-dx, dx);
//        const float rdy = rng.uniform(-dy, dy);

//        if(tl.x - rdx < 0.f || tl.y - rdy < 0.f || tl.x + rdx < 0.f || tl.y + rdy < 0.f)
//            continue;

//        const cv::Rect randomRect1(Point(cvRound(tl.x + rdx), cvRound(tl.y + rdy)), bBox.size());
//        const cv::Rect randomRect2(Point(cvRound(tl.x - rdx), cvRound(tl.y - rdy)), bBox.size());

//        const Rect overlap1 = bBox & randomRect1;
//        const Rect overlap2 = bBox & randomRect2;

//        const float overlapValue1 = float(overlap1.area()) / (randomRect1.area() + bBox.area() - overlap1.area());
//        const float overlapValue2 = float(overlap2.area()) / (randomRect2.area() + bBox.area() - overlap2.area());

//        if(overlapValue1 >= rangeEnd && overlapValue2 >= rangeEnd)
//        {
//            ret.push_back(randomRect1), ret.push_back(randomRect2);
//        }

//    }

    return ret;
}

#define GROUP_RESPONSES

std::vector< std::pair<Rect, double> > tldCascadeClassifier::prepareFinalResult(const Mat_<uchar> &image) const
{
    std::vector< std::pair<Rect, double> > results;

#ifdef GROUP_RESPONSES
    std::vector<Rect> groupedRects;
#endif

    for(size_t index = 0; index < hypothesis.size(); ++index)
        if(answers[index])
        {
#ifndef GROUP_RESPONSES
            const double confidence = nnClassifier->calcConfidence(image(hypothesis[index].bb));
            if(confidence >= 0.5)
                results.push_back(std::make_pair(hypothesis[index].bb, confidence));
#else
            groupedRects.push_back(hypothesis[index].bb);
#endif
        }

#ifdef GROUP_RESPONSES
    myGroupRectangles(groupedRects, 0.25);
    for(std::vector<Rect>::const_iterator groupedRectsIt = groupedRects.begin(); groupedRectsIt != groupedRects.end(); ++groupedRectsIt)
        results.push_back(std::make_pair(*groupedRectsIt, nnClassifier->calcConfidence(image(*groupedRectsIt))));
#endif

    std::sort(results.begin(), results.end(), greater);

    return results;
}

std::vector<float> tldCascadeClassifier::generateRandomValues(float range, int quantity) const
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

void tldCascadeClassifier::myGroupRectangles(std::vector<Rect> &rectList, double eps) const
{
    if(rectList.size() < 2u)
        return;

    std::vector<int> labels;
    int nclasses = partition(rectList, labels, SimilarRects(eps));

    std::vector<Rect> rrects(nclasses);
    std::vector<int> rweights(nclasses, 0);
    int i, nlabels = (int)labels.size();
    for( i = 0; i < nlabels; i++ )
    {
        int cls = labels[i];
        rrects[cls].x += rectList[i].x;
        rrects[cls].y += rectList[i].y;
        rrects[cls].width += rectList[i].width;
        rrects[cls].height += rectList[i].height;
        rweights[cls]++;
    }

    for( i = 0; i < nclasses; i++ )
    {
        Rect r = rrects[i];
        //float s = 1.f/rweights[i];
        rrects[i] = Rect(saturate_cast<int>(r.x / rweights[i]),
             saturate_cast<int>(r.y / rweights[i]),
             saturate_cast<int>(r.width / rweights[i]),
             saturate_cast<int>(r.height / rweights[i]));
    }

    rectList = rrects;
}

bool tldCascadeClassifier::isRectOK(const Rect &rect) const
{
    return rect.tl().x >= 0 && rect.tl().y >= 0 && rect.br().x <= frameSize.width && rect.br().y <= frameSize.height;
}

Mat_<uchar> tldCascadeClassifier::getWarped(const Mat_<uchar> &originalFrame, Rect bb, float shiftX, float shiftY, float scale, float rotation)
{
    //const float shiftRangeX = bb.width * shiftRangePercent;
    //const float shiftRangeY = bb.height * shiftRangePercent;

    Mat shiftTransform = cv::Mat::eye(3, 3, CV_32F);
    shiftTransform.at<float>(0,2) = shiftX;//rng.uniform(-shiftRangeX, shiftRangeX);
    shiftTransform.at<float>(1,2) = shiftY;//rng.uniform(-shiftRangeY, shiftRangeY);

    Mat scaleTransform = cv::Mat::eye(3, 3, CV_32F);
    scaleTransform.at<float>(0,0) = 1 - scale;//rng.uniform(-scaleRangePercent, scaleRangePercent);
    scaleTransform.at<float>(1,1) = scaleTransform.at<float>(0,0); //1 - rng.uniform(-scaleRangePercent, scaleRangePercent);

    Mat rotationShiftTransform = cv::Mat::eye(3, 3, CV_32F);
    rotationShiftTransform.at<float>(0,2) = bb.tl().x + float(bb.width) / 2;
    rotationShiftTransform.at<float>(1,2) = bb.tl().y + float(bb.height) / 2;

    const float angle = (rotation * CV_PI) / 180.f;//rng.uniform(-rotationRangeRad, rotationRangeRad);

    Mat rotationTransform = cv::Mat::eye(3, 3, CV_32F);
    rotationTransform.at<float>(0,0) = rotationTransform.at<float>(1,1) = std::cos(angle);
    rotationTransform.at<float>(0,1) = std::sin(angle);
    rotationTransform.at<float>(1,0) = - rotationTransform.at<float>(0,1);

    const Mat resultTransform = rotationShiftTransform * rotationTransform * rotationShiftTransform.inv() * scaleTransform * shiftTransform; /*cv::Mat::eye(3, 3, CV_32F)*/;

    Mat_<uchar> dst;
    warpAffine(originalFrame, dst, resultTransform(cv::Rect(0,0,3,2)), dst.size());

    //std::cout << bb << std::endl;
    return dst(bb);
}

void tldCascadeClassifier::addScanGrid(const Size frameSize, const Size bbSize, const Size minimalBBSize, std::vector<Hypothesis> &hypothesis, double scale)
{
    CV_Assert(bbSize.width >= minimalBBSize.width && bbSize.height >= minimalBBSize.height);

    const int dx = bbSize.width / 10;
    const int dy = bbSize.height / 10;

    CV_Assert(dx > 0 && dy > 0);

    for(int currentX = 0; currentX < frameSize.width - bbSize.width - dx; currentX += dx)
        for(int currentY = 0; currentY < frameSize.height - bbSize.height - dy; currentY += dy)
        {
            hypothesis.push_back(Hypothesis(currentX, currentY, bbSize, scale));
        }
}

}
}
