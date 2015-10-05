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


#define FERN_EXPERIMENT

namespace cv
{
namespace tld
{

CascadeClassifier::CascadeClassifier(int preMeasure, int preFerns, Size preFernPathSize,
                                           int numberOfMeasurements, int numberOfFerns, Size fernPatchSize,
                                           int numberOfExamples, Size examplePatchSize, int actPositiveExampleNumbers, int actWrappedExamplesNumber, double actGroupTheta):
    minimalBBSize(20, 20), scaleStep(1.2f), groupRectanglesTheta(actGroupTheta), positiveExampleNumbers(actPositiveExampleNumbers),
    wrappedExamplesNumber(actWrappedExamplesNumber), isInited(false)
{
    varianceClassifier = makePtr<VarianceClassifier>();
    preFernClassifier = makePtr<FernClassifier>(preMeasure, preFerns, preFernPathSize);
    fernClassifier = makePtr<FernClassifier>(numberOfMeasurements, numberOfFerns, fernPatchSize);
    nnClassifier = makePtr<NNClassifier>(numberOfExamples, examplePatchSize, 0.5);

}

void CascadeClassifier::init(const Mat_<uchar> &zeroFrame, const Rect &bb)
{
    if(bb.width < minimalBBSize.width || bb.height < minimalBBSize.height)
        CV_Error(Error::StsBadArg, "Initial bounding box is too small");

    originalBBSize = bb.size();
    frameSize = zeroFrame.size();
    hypothesis = generateHypothesis(frameSize, originalBBSize, minimalBBSize, scaleStep);

    pExpert = makePtr<PExpert>(zeroFrame.size());
    nExpert = makePtr<NExpert>();


    addPositiveExamples(pExpert->generatePositiveExamples(zeroFrame, bb, 5, 25));

    isInited = true;
}


//#define TIME_MEASURE
#define DEBUG_OUTPUT
std::vector<Rect> CascadeClassifier::detect(const Mat_<uchar> &scaledImage) const
{

    CV_Assert(isInited);

#ifdef TIME_MEASURE
    timeval varianceStart, fernStart, nnStart, nnStop, mergeStop;
#endif

    answers.clear();
#ifdef TIME_MEASURE
    gettimeofday(&varianceStart, NULL);
#endif

    varianceClassifier->isObjects(hypothesis, scaledImage, answers);

/*#ifdef DEBUG_OUTPUT
    Mat_<uchar> copyVariance; scaledImage.copyTo(copyVariance);

    for(size_t index = 0; index < hypothesis.size(); ++index)
        if(answers[index])
            rectangle(copyVariance, hypothesis[index].bb, Scalar::all(255));

    imshow("after variance filter", copyVariance);
    waitKey();
#endif*/

#ifdef TIME_MEASURE
    gettimeofday(&fernStart, NULL);
#endif

    preFernClassifier->isObjects(hypothesis, scaledImage, answers);
    fernClassifier->isObjects(hypothesis, scaledImage, answers);

    //////////////////////experimental code/////////////////////////
#ifdef FERN_EXPERIMENT
    fernsPositive.clear();
    for(size_t index = 0; index < answers.size(); ++index)
        if(answers[index])
            fernsPositive.push_back(hypothesis[index].bb);
#endif
    //////////////////////experimental code/////////////////////////


#ifdef DEBUG_OUTPUT
    Mat_<uchar> copy; scaledImage.copyTo(copy);

    for(size_t index = 0; index < hypothesis.size(); ++index)
        if(answers[index])
            rectangle(copy, hypothesis[index].bb, Scalar::all(255));

    imshow("after fern", copy);
//    waitKey();
#endif


#ifdef TIME_MEASURE
    gettimeofday(&nnStart, NULL);
#endif

    nnClassifier->isObjects(hypothesis, scaledImage, answers);

#ifdef DEBUG_OUTPUT
    Mat_<uchar> copyNN; scaledImage.copyTo(copyNN);

    for(size_t index = 0; index < hypothesis.size(); ++index)
        if(answers[index])
            rectangle(copyNN, hypothesis[index].bb, Scalar::all(255));

    imshow("after nn filter", copyNN);
//    std::pair<Mat, Mat> model = nnClassifier->outputModel();
//    imshow("nn model positive", model.first);
//    imshow("nn model negative", model.second);

    waitKey(1);
#endif


#ifdef TIME_MEASURE
    gettimeofday(&nnStop, NULL);
#endif

    const std::vector<Rect> &result = prepareFinalResult(scaledImage);

#ifdef TIME_MEASURE
    gettimeofday(&mergeStop, NULL);

    std::cout << " var " << std::fixed << fernStart.tv_sec - varianceStart.tv_sec + double(fernStart.tv_usec - varianceStart.tv_usec) / 1e6;
    std::cout << " fern " << std::fixed << nnStart.tv_sec - fernStart.tv_sec + double(nnStart.tv_usec - fernStart.tv_usec) / 1e6;
    std::cout << " nn " << std::fixed << nnStop.tv_sec - nnStart.tv_sec + double(nnStop.tv_usec - nnStart.tv_usec) / 1e6 <<" "<< nnClassifier->positiveExamples.size() + nnClassifier->negativeExamples.size();
    std::cout << " merge " << std::fixed << mergeStop.tv_sec - nnStop.tv_sec + double(mergeStop.tv_usec - nnStop.tv_usec) / 1e6;
    std::cout << " total " << std::fixed << mergeStop.tv_sec - varianceStart.tv_sec + double(mergeStop.tv_usec - varianceStart.tv_usec) / 1e6 << std::endl;
#endif

    return result;
}

void CascadeClassifier::startPExpert(const Mat_<uchar> &image, const Rect &bb)
{
    addPositiveExamples(pExpert->generatePositiveExamples(image, bb, positiveExampleNumbers, wrappedExamplesNumber));
}

void CascadeClassifier::startNExpert(const Mat_<uchar> &image, const Rect &bb, const std::vector<Rect> &detections)
{
    addNegativeExamples(nExpert->getNegativeExamples(image, bb, detections));

#ifdef FERN_EXPERIMENT
    fernClassifier->integrateNegativeExamples(nExpert->getNegativeExamples(image, bb, fernsPositive));
#endif
}

void CascadeClassifier::addPositiveExamples(const std::vector<Mat_<uchar> > &examples)
{
    std::vector< Mat_<uchar> > exampleCopy(examples);

    exampleCopy.erase(std::remove_if(exampleCopy.begin(), exampleCopy.end(), std::bind1st(std::ptr_fun(isObjectPredicate), this)), exampleCopy.end());

    varianceClassifier->integratePositiveExamples(exampleCopy);
    preFernClassifier->integratePositiveExamples(exampleCopy);
    fernClassifier->integratePositiveExamples(exampleCopy);
    nnClassifier->integratePositiveExamples(exampleCopy);
}

void CascadeClassifier::addNegativeExamples(const std::vector<Mat_<uchar> > &examples)
{
    preFernClassifier->integrateNegativeExamples(examples);
    fernClassifier->integrateNegativeExamples(examples);
    nnClassifier->integrateNegativeExamples(examples);

}

std::vector<Hypothesis> CascadeClassifier::generateHypothesis(const Size frameSize, const Size bbSize, const Size minimalBBSize, double scaleStep)
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

#define GROUP_RESPONSES
std::vector<Rect> CascadeClassifier::prepareFinalResult(const Mat_<uchar> &image) const
{
    std::vector< std::pair<Rect, double> > tResults;

#ifdef GROUP_RESPONSES
    std::vector<Rect> groupedRects;
#endif

    std::vector<Rect> finalResult;

    for(size_t index = 0; index < hypothesis.size(); ++index)
        if(answers[index])
        {
#ifndef GROUP_RESPONSES
            const double confidence = nnClassifier->calcConfidence(image(hypothesis[index].bb));
            if(confidence >= 0.5)
                finalResult.push_back(hypothesis[index].bb);
#else
            groupedRects.push_back(hypothesis[index].bb);
#endif
        }

#ifdef GROUP_RESPONSES
    myGroupRectangles(groupedRects, groupRectanglesTheta);
    for(std::vector<Rect>::const_iterator groupedRectsIt = groupedRects.begin(); groupedRectsIt != groupedRects.end(); ++groupedRectsIt)
        tResults.push_back(std::make_pair(*groupedRectsIt, nnClassifier->calcConfidence(image(*groupedRectsIt))));
#endif

    std::sort(tResults.begin(), tResults.end(), greater);

    std::transform(tResults.begin(), tResults.end(), std::back_inserter(finalResult), std::ptr_fun(strip));

    return finalResult;
}

void CascadeClassifier::myGroupRectangles(std::vector<Rect> &rectList, double eps) const
{
    if(rectList.size() < 2u)
        return;

    std::vector<int> labels;
    int nclasses = partition(rectList, labels, SimilarRects(eps));
    int i, nlabels = (int)labels.size();
    std::vector<int> rweights(nclasses, 0);
    std::vector<Rect> rrects;
#if 0
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
        float s = 1.f/rweights[i];
        rrects[i] = Rect(saturate_cast<int>(r.x * s),
             saturate_cast<int>(r.y * s),
             saturate_cast<int>(r.width * s),
             saturate_cast<int>(r.height * s));
    }
#else

    std::vector<std::vector<int> > x(nclasses), y(nclasses), width(nclasses), height(nclasses);

    for( i = 0; i < nlabels; i++ )
    {
        int cls = labels[i];
        x[cls].push_back(rectList[i].x);
        y[cls].push_back(rectList[i].y);
        width[cls].push_back(rectList[i].width);
        height[cls].push_back(rectList[i].height);
        rweights[cls]++;
    }

    for( i = 0; i < nclasses; i++ )
    {
        std::vector<int> &_x = x[i];
        std::vector<int> &_y = y[i];
        std::vector<int> &_width = width[i];
        std::vector<int> &_height = height[i];

//        if(rweights[i] <= 2)
//        {
//            rrects[i] = Rect
//                    ( float(std::accumulate(_x.begin(), _x.end(), 0)) / rweights[i],
//                      float(std::accumulate(_y.begin(), _y.end(), 0)) / rweights[i],
//                      float(std::accumulate(_width.begin(), _width.end(), 0)) / rweights[i],
//                      float(std::accumulate(_height.begin(), _height.end(), 0)) / rweights[i]
//                      );
//        }
//        else
        {
            std::sort(_x.begin(), _x.end());
            std::sort(_y.begin(), _y.end());
            std::sort(_width.begin(), _width.end());
            std::sort(_height.begin(), _height.end());

            Rect rect( _x[_x.size() / 2], _y[_y.size() / 2], _width[_width.size() / 2],_height[_height.size() / 2]);

            if(pExpert->isRectOK(rect))
                rrects.push_back(rect);
        }
    }

#endif

    rectList = rrects;
}

void CascadeClassifier::addScanGrid(const Size frameSize, const Size bbSize, const Size minimalBBSize, std::vector<Hypothesis> &hypothesis, double scale)
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

bool CascadeClassifier::isObject(const Mat_<uchar> &candidate) const
{
    if(!varianceClassifier->isObject(candidate))
         return false;

    if(!fernClassifier->isObject(candidate))
        return false;

    if(!nnClassifier->isObject(candidate))
        return false;

    return true;
}

bool CascadeClassifier::isObjectPredicate(const CascadeClassifier *pCascadeClassifier, const Mat_<uchar> candidate)
{
    return pCascadeClassifier->isObject(candidate);
}

std::vector< Mat_<uchar> > CascadeClassifier::PExpert::generatePositiveExamples(const Mat_<uchar> &image, const Rect &bb, int numberOfsurroundBbs, int numberOfSyntheticWarped)
{
    const float shiftRangePercent = .01f;
    const float scaleRange = .01f;
    const float angleRangeDegree = 10.f;

    const float shiftXRange = shiftRangePercent * bb.width;
    const float shiftYRange = shiftRangePercent * bb.height;

    std::vector< Mat_<uchar> > positiveExamples;

    if(isRectOK(bb))
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

            for(int j = 0; j < warpedOOI.rows * warpedOOI.cols; ++j)
                    warpedOOI.at<uchar>(j) = saturate_cast<uchar>(warpedOOI.at<uchar>(j) + rng.gaussian(5.));

            positiveExamples.push_back(warpedOOI);
        }

        //positiveExamples.push_back(image(*positiveRect).clone());
    }

    return positiveExamples;

}

bool CascadeClassifier::PExpert::isRectOK(const Rect &rect) const
{
    return rect.tl().x >= 0 && rect.tl().y >= 0 && rect.br().x <= frameSize.width && rect.br().y <= frameSize.height;
}

std::vector<Rect> CascadeClassifier::PExpert::generateClosestN(const Rect &bBox, int n)
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

    std::multimap<float, Rect> storage;

    for(int stepX = -n; stepX <= n; ++stepX)
    {
        for(int stepY = -n; stepY <= n; ++stepY)
        {
            const Rect actRect(Point(cvRound(tl.x + dx*stepX), cvRound(tl.y + dy*stepY)), bBox.size());

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

std::vector<float> CascadeClassifier::PExpert::generateRandomValues(float range, int quantity)
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

Mat_<uchar> CascadeClassifier::PExpert::getWarped(const Mat_<uchar> &originalFrame, Rect bb, float shiftX, float shiftY, float scale, float rotation)
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

//#define DEBUG_OUTPUT2
std::vector<Mat_<uchar> > CascadeClassifier::NExpert::getNegativeExamples(const Mat_<uchar> &image, const Rect &object, const std::vector<Rect> &detectedObjects)
{
#ifdef DEBUG_OUTPUT2
    Mat copy; cvtColor(image, copy, CV_GRAY2BGR);
#endif
    std::vector< Mat_<uchar> > negativeExamples;

    for(size_t i = 0; i < detectedObjects.size(); ++i)
    {
        const Rect &actDetectedObject = detectedObjects[i];

        if(overlap(actDetectedObject, object) < 0.5)
        {
            negativeExamples.push_back(image(actDetectedObject).clone());
#ifdef DEBUG_OUTPUT2
            rectangle(copy, actDetectedObject, cv::Scalar(0, 165, 255));
#endif
        }
    }

#ifdef DEBUG_OUTPUT2
    imshow("neg examples", copy);
    waitKey(1);
#endif

    return negativeExamples;
}

}
}
