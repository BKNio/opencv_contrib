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

Mat_<uchar> CascadeClassifier::debugOutput;

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

    std::vector<Rect> negativeExamples;
    for(std::vector<Hypothesis>::const_iterator it = hypothesis.begin(); it != hypothesis.end(); ++it)
    {
        const double actOverlap = overlap(bb, it->bb);

        if(actOverlap < 0.5 && actOverlap > 0)
            negativeExamples.push_back(it->bb);
    }

    addPositiveExamples(pExpert->generatePositiveExamples(zeroFrame, bb, 1, 200));
    addNegativeExamples(nExpert->getNegativeExamples(zeroFrame, bb, negativeExamples, "init negative"));

    isInited = true;

}


//#define TIME_MEASURE
//#define DEBUG_OUTPUT
std::vector< std::pair<Rect, double> > CascadeClassifier::detect(const Mat_<uchar> &scaledImage) const
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

#ifdef TIME_MEASURE
    gettimeofday(&fernStart, NULL);
#endif

    //preFernClassifier->isObjects(hypothesis, scaledImage, answers);


    fernClassifier->isObjects(hypothesis, scaledImage, answers);

    fernsPositive.clear();
    for(size_t index = 0; index < answers.size(); ++index)
        if(answers[index])
            fernsPositive.push_back(hypothesis[index].bb);


    Mat_<uchar> copy; scaledImage.copyTo(copy);
    for(size_t index = 0; index < hypothesis.size(); ++index)
        if(answers[index])
            rectangle(copy, hypothesis[index].bb, Scalar::all(255));
    imshow("after fern", copy);


#ifdef TIME_MEASURE
    gettimeofday(&nnStart, NULL);
#endif

    nnClassifier->isObjects(hypothesis, scaledImage, answers);

    nnPositive.clear();
    for(size_t index = 0; index < answers.size(); ++index)
        if(answers[index])
            nnPositive.push_back(hypothesis[index].bb);


    Mat_<uchar> copyNN; scaledImage.copyTo(copyNN);
    for(size_t index = 0; index < hypothesis.size(); ++index)
        if(answers[index])
            rectangle(copyNN, hypothesis[index].bb, Scalar::all(255));
    imshow("after NN", copyNN);

//    std::pair<Mat, Mat> model = nnClassifier->outputModel();
//    imshow("nn model positive", model.first);
//    imshow("nn model negative", model.second);

#ifdef TIME_MEASURE
    gettimeofday(&nnStop, NULL);
#endif

    const std::vector< std::pair<Rect, double> > &result = prepareFinalResult(scaledImage);

//    Mat copyNN; cvtColor(scaledImage,copyNN, CV_GRAY2BGR);

//    for(std::vector<std::pair<Rect, double> >::const_iterator it = result.begin(); it != result.end(); ++it)
//    {rectangle(copyNN, it->first, Scalar::all(255 * it->second));}

//    imshow("nn", copyNN);
//    waitKey();

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

void CascadeClassifier::startNExpert(const Mat_<uchar> &image, const Rect &bb)
{
    const std::vector< Mat_<uchar> > &negExamplesForNN = nExpert->getNegativeExamples(image, bb, nnPositive, "NN negative");
    nnClassifier->integrateNegativeExamples(negExamplesForNN);

    const std::vector< Mat_<uchar> > &negExamplesForFern = nExpert->getNegativeExamples(image, bb, fernsPositive, "ferns negative");
    fernClassifier->integrateNegativeExamples(negExamplesForFern);
}

void CascadeClassifier::addPositiveExamples(const std::vector<Mat_<uchar> > &examples)
{
    varianceClassifier->integratePositiveExamples(examples);
    fernClassifier->integratePositiveExamples(examples);
    nnClassifier->integratePositiveExamples(examples);
}

void CascadeClassifier::addNegativeExamples(const std::vector<Mat_<uchar> > &examples)
{
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

std::vector< std::pair<Rect, double> > CascadeClassifier::prepareFinalResult(const Mat_<uchar> &image) const
{
    std::vector< std::pair<Rect, double> > finalResult;

    for(size_t index = 0; index < hypothesis.size(); ++index)
    {
        if(answers[index])
        {
            const double confidence = nnClassifier->calcConfidence(image(hypothesis[index].bb));
            finalResult.push_back(std::make_pair(hypothesis[index].bb, confidence));
            /*CV_Assert(confidence >= 0.5);*/
        }
    }

    std::sort(finalResult.begin(), finalResult.end(), greater);

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

            CV_Assert(_x.size() == _y.size());
            CV_Assert(_y.size() == _width.size());
            CV_Assert(_width.size() == _height.size());


//            Mat_<uchar> copy; debugOutput.copyTo(copy);
//            for(size_t index = 0; index < _x.size(); ++index)
//                rectangle(copy, Rect(_x[index], _y[index], _width[index], _height[index]), Scalar::all(0));

            Rect rect;

            const size_t middle = _x.size() / 2;

            if(_x.size() % 2  == 0)
                rect = Rect2d( (_x[middle] + _x[middle - 1]) * 0.5,
                             (_y[middle] + _y[middle - 1]) * 0.5,
                        (_width[middle] + _width[middle - 1]) * 0.5,
                        (_height[middle] + _height[middle - 1]) * 0.5);
            else
                rect = Rect( _x[middle], _y[middle], _width[middle],_height[middle]);

//            rectangle(copy, rect, Scalar::all(255));

//            imshow("group rects", copy);
//            waitKey();

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

    /////////////////////////////experimental////////////////////////////
//    Mat mirror = Mat::eye(3, 3, CV_32F);
//    mirror.at<float>(0,0) = -1.f;
//    Mat shift = Mat::eye(3, 3, CV_32F);
//    shift.at<float>(0,2) = bb.width / 2;
//    shift.at<float>(1,2) = bb.height / 2;
//    Mat result = shift * mirror * shift.inv();
    /////////////////////////////experimental////////////////////////////


    std::vector< Mat_<uchar> > positiveExamples;

//    if(isRectOK(bb))
//        positiveExamples.push_back(image(bb));

    std::vector<Rect> nClosestRects = generateClosestN(bb, /*numberOfsurroundBbs*/1);

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

            //warpedOOI *= rng.uniform(0.8, 1.2);

            //Mat_<uchar> mirrored;
            //warpAffine(warpedOOI, mirrored, result(Rect(0,0,3,2)), warpedOOI.size());

            positiveExamples.push_back(warpedOOI);
            //positiveExamples.push_back(mirrored);


//            Mat_<uchar> mirroredBrighter = mirrored * 1.1;
//            Mat_<uchar> mirroredDarker = mirrored * 0.9;

//            positiveExamples.push_back(mirroredBrighter);
//            positiveExamples.push_back(mirroredDarker);


//            Mat_<uchar> warpedOOIBrighter = warpedOOI * 1.1;
//            Mat_<uchar> warpedOOIDarker = warpedOOI * 0.9;

//            positiveExamples.push_back(warpedOOIBrighter);
//            positiveExamples.push_back(warpedOOIDarker);

        }

        positiveExamples.push_back(image(*positiveRect).clone());
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

Mat_<uchar> CascadeClassifier::PExpert::getWarped(const Mat_<uchar> &originalFrame, const Rect &bb, float shiftX, float shiftY, float scale, float rotation)
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

    const Mat resultTransform = /*cv::Mat::eye(3,3, CV_32F);*/rotationShiftTransform * rotationTransform * rotationShiftTransform.inv() * scaleTransform * shiftTransform;

    Mat_<uchar> dst;
    warpAffine(originalFrame, dst, resultTransform(cv::Rect(0,0,3,2)), dst.size());

    return dst(bb);
}

//#define DEBUG_OUTPUT2
std::vector<Mat_<uchar> > CascadeClassifier::NExpert::getNegativeExamples(const Mat_<uchar> &image,
                                                                          const Rect &object,
                                                                          const std::vector<Rect> &detectedObjects,
                                                                          std::string capture)
{

#ifdef DEBUG_OUTPUT2
    Mat copy; cvtColor(image, copy, CV_GRAY2BGR);
#endif
    std::vector< Mat_<uchar> > negativeExamples;

    for(size_t i = 0; i < detectedObjects.size(); ++i)
    {
        const Rect &actDetectedObject = detectedObjects[i];

        if(overlap(actDetectedObject, object) < .5 && VarianceClassifier::variance(image(actDetectedObject)) > 0)
        {
            negativeExamples.push_back(image(actDetectedObject).clone());
#ifdef DEBUG_OUTPUT2
            rectangle(copy, actDetectedObject, cv::Scalar(0, 165, 255));
#endif
        }
#ifdef DEBUG_OUTPUT2
        else
        {
            rectangle(copy, actDetectedObject, cv::Scalar(255, 165, 0));
        }
#endif
    }

#ifdef DEBUG_OUTPUT2
    rectangle(copy, object, cv::Scalar(165, 0, 255));
    imshow(capture, copy);

//    if(capture == "NN negative" && !negativeExamples.empty())
//        waitKey();
#endif

    return negativeExamples;
}

}
}
