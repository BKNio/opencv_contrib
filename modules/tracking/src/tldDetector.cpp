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

#include "tldDetector.hpp"

//#define SHOW_POSITIVE_EXAMPLE
//#define SHOW_ADDITIONAL_POSITIVE
//#define SHOW_ADDITIONAL_NEGATIVE

namespace cv
{
namespace tld
{

tldCascadeClassifier::tldCascadeClassifier(const Mat_<uchar> &zeroFrame, const Rect &bb, int numberOfMeasurements, int numberOfFerns, int maxNumberOfExamples):
    minimalBBSize(15, 15), standardPatchSize(15, 15),originalBBSize(bb.size()),
    frameSize(zeroFrame.size()), scaleStep(1.2f), hypothesis(generateHypothesis(frameSize, originalBBSize, minimalBBSize, scaleStep))
{

    if(bb.width < minimalBBSize.width || bb.height < minimalBBSize.height)
        CV_Error(Error::StsBadArg, "Initial bounding box is too small");

    const Mat_<uchar> ooi = zeroFrame(bb);

    varianceClassifier = makePtr<tldVarianceClassifier>(ooi);
    fernClassifier = makePtr<tldFernClassifier>(numberOfMeasurements, numberOfFerns);
    nnClassifier = makePtr<tldNNClassifier>(maxNumberOfExamples, standardPatchSize);

    addPositiveExample(ooi);
#ifdef SHOW_POSITIVE_EXAMPLE
    imshow("positive example ooi", ooi);
    waitKey();
#endif


    addSyntheticPositive(zeroFrame, bb, 10, 20);

    std::vector<Rect> surroundingRects = generateSurroundingRects(bb, 10);
    for(std::vector<Rect>::const_iterator negatievRect = surroundingRects.begin(); negatievRect != surroundingRects.end(); ++negatievRect)
    {
        addPositiveExample(zeroFrame(*negatievRect));
#ifdef SHOW_ADDITIONAL_NEGATIVE
        imshow("additional negative", zeroFrame(*negatievRect));
        waitKey();
#endif
    }

}

std::vector<Rect> tldCascadeClassifier::detect(const Mat_<uchar> &scaledImage) const
{
    timeval varianceStart, fernStart, nnStart, nnStop;

    answers.clear();
    gettimeofday(&varianceStart, NULL);
    varianceClassifier->isObjects(hypothesis, scaledImage, answers);
    gettimeofday(&fernStart, NULL);
    Mat_<uchar> blurred;
    GaussianBlur(scaledImage, blurred, Size(3,3), 0.);
    fernClassifier->isObjects(hypothesis, blurred, answers);
    gettimeofday(&nnStart, NULL);
//    nnClassifier->isObjects(hypothesis, scaledImage, answers);
    gettimeofday(&nnStop, NULL);

    return prepareFinalResult();

    /*std::cout << "var time " << fernStart.tv_sec - varianceStart.tv_sec + double(fernStart.tv_usec - varianceStart.tv_usec) / 1e6 << std::endl;
    std::cout << "fern time " << nnStart.tv_sec - fernStart.tv_sec + double(nnStart.tv_usec - fernStart.tv_usec) / 1e6 << std::endl;
    std::cout << "nn time " << nnStop.tv_sec - nnStart.tv_sec + double(nnStop.tv_usec - nnStart.tv_usec) / 1e6 << std::endl;*/
}

void tldCascadeClassifier::addSyntheticPositive(const Mat_<uchar> &image, const Rect bb, int numberOfsurroundBbs, int numberOfSyntheticWarped)
{
    const float shiftRangePercent = .01f;
    const float scaleRange = .01f;
    const float angleRangeDegree = 10.f;

    std::vector<Rect> NClosestRects = generateClosestN(bb, numberOfsurroundBbs);
    for(std::vector<Rect>::const_iterator positiveRect = NClosestRects.begin(); positiveRect != NClosestRects.end(); ++positiveRect)
    {
        for(int syntheticWarp = 0; syntheticWarp < numberOfSyntheticWarped; ++syntheticWarp)
        {
            const Mat_<uchar> &warpedOOI = randomWarp(image, *positiveRect, shiftRangePercent, scaleRange, angleRangeDegree);
            addPositiveExample(warpedOOI);
#ifdef SHOW_ADDITIONAL_POSITIVE
            imshow("additional positive", warpedOOI);
            imshow("additional bb", image(*positiveRect));
            waitKey();
#endif
        }
    }
}

void tldCascadeClassifier::addPositiveExample(const Mat_<uchar> &example)
{
    fernClassifier->integratePositiveExample(example);
    nnClassifier->integratePositiveExample(example);
}

void tldCascadeClassifier::addNegativeExample(const Mat_<uchar> &example)
{
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
        Size currentBBSize(bbSize.width * correctedScale, bbSize.height * correctedScale);

        if(currentBBSize.width < minimalBBSize.width || currentBBSize.height < minimalBBSize.height)
            break;
        addScanGrid(frameSize, currentBBSize, minimalBBSize, hypothesis);

//        {
//            for(std::vector<Hypothesis>::const_iterator it = hypothesis.begin(); it < hypothesis.end(); ++it)
//            {
//                Mat_<uchar> copy(frameSize, 0);
//                rectangle(copy, it->bb, Scalar::all(255));
//                imshow("copy", copy);
//                waitKey(1);
//            }
//            hypothesis.clear();
//        }

        correctedScale /= scaleStep;
    }
    return hypothesis;
}

std::vector<Rect> tldCascadeClassifier::generateClosestN(const Rect &bBox, size_t N) const
{
    return generateAndSelectRects(bBox, N, 0.99f, 0.5f);
}

std::vector<Rect> tldCascadeClassifier::generateSurroundingRects(const Rect &bBox, size_t N) const
{
    return generateAndSelectRects(bBox, N, 0.2f, 0.01f);
}

std::vector<Rect> tldCascadeClassifier::generateAndSelectRects(const Rect &bBox, int n, float rangeStart, float rangeEnd) const
{
    CV_Assert(n > 0);

    std::vector<Rect> ret; ret.reserve(n);

    const float dx = float(bBox.width) / 10;
    const float dy = float(bBox.height) / 10;

    const Point tl = bBox.tl();

    std::multimap<float, Rect> storage;

    for(int stepX = -n; stepX <= n; ++stepX)
    {
        for(int stepY = -n; stepY <= n; ++stepY)
        {
            const Rect actRect(Point(tl.x + dx*stepX, tl.y + dy*stepY), bBox.size());
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
                ret.push_back(closestIt->second);

        if(ret.size() == size_t(n))
            break;
    }

    return ret;
}

std::vector<Rect> tldCascadeClassifier::prepareFinalResult() const
{
    std::vector<Rect> rects;

    CV_Assert(hypothesis.size() == answers.size());
    for(size_t index = 0; index < hypothesis.size(); ++index)
    {
        if(answers[index])
        {
            for(size_t innerIndex = 0; innerIndex < rects.size(); ++innerIndex)
            {
                cv::Rect intersect = rects[innerIndex] & hypothesis[index];
                if(float(intersect.area()) / (rects[innerIndex].area() + hypothesis[index].area() - intersect.area()) > .5f)
                {

                }
            }
        }
    }

    std::vector<Rect> result;

    for(size_t resultsIndex = 0; resultsIndex < numberOfPrecedents.size(); ++resultsIndex)
    {

        const Point cSumPoint = sumPoints[resultsIndex];
        const Size cSumSize = sumSize[resultsIndex];
        const int cNumberOfPrecedents = numberOfPrecedents[resultsIndex];

        Point tl(cSumPoint.x / cNumberOfPrecedents, cSumPoint.y / cNumberOfPrecedents);
        Size rectSize(cSumSize.width / cNumberOfPrecedents, cSumSize.height / cNumberOfPrecedents);
        result.push_back(Rect(tl, rectSize));
    }

    return rects;
}

bool tldCascadeClassifier::isRectOK(const Rect &rect) const
{
    return rect.tl().x >= 0 && rect.tl().y >= 0 && rect.br().x <= frameSize.width && rect.br().y <= frameSize.height;
}

Mat_<uchar> tldCascadeClassifier::randomWarp(const Mat_<uchar> &originalFrame, Rect bb, float shiftRangePercent, float scaleRangePercent, float rotationRangeDegrees)
{
    const float shiftRangeX = bb.width * shiftRangePercent;
    const float shiftRangeY = bb.height * shiftRangePercent;

    Mat shiftTransform = cv::Mat::eye(3, 3, CV_32F);
    shiftTransform.at<float>(0,2) = rng.uniform(-shiftRangeX, shiftRangeX);
    shiftTransform.at<float>(1,2) = rng.uniform(-shiftRangeY, shiftRangeY);

    Mat scaleTransform = cv::Mat::eye(3, 3, CV_32F);
    scaleTransform.at<float>(0,0) = 1 - rng.uniform(-scaleRangePercent, scaleRangePercent);
    scaleTransform.at<float>(1,1) = 1 - rng.uniform(-scaleRangePercent, scaleRangePercent);

    Mat rotationShiftTransform = cv::Mat::eye(3, 3, CV_32F);
    rotationShiftTransform.at<float>(0,2) = bb.tl().x + float(bb.width) / 2;
    rotationShiftTransform.at<float>(1,2) = bb.tl().y + float(bb.height) / 2;

    const float rotationRangeRad = (rotationRangeDegrees * CV_PI) / 180.f;
    const float angle = rng.uniform(-rotationRangeRad, rotationRangeRad);

    Mat rotationTransform = cv::Mat::eye(3, 3, CV_32F);
    rotationTransform.at<float>(0,0) = rotationTransform.at<float>(1,1) = std::cos(angle);
    rotationTransform.at<float>(0,1) = std::sin(angle);
    rotationTransform.at<float>(1,0) = - rotationTransform.at<float>(0,1);

    const Mat resultTransform = rotationShiftTransform * rotationTransform * rotationShiftTransform.inv() * scaleTransform * shiftTransform;

    Mat_<uchar> dst;
    warpAffine(originalFrame, dst, resultTransform(cv::Rect(0,0,3,2)), dst.size());

    return dst(bb);
}

void tldCascadeClassifier::addScanGrid(const Size frameSize, const Size bbSize, const Size minimalBBSize, std::vector<Hypothesis> &hypothesis)
{
    CV_Assert(bbSize.width >= minimalBBSize.width && bbSize.height >= minimalBBSize.height);

    const int dx = bbSize.width / 10;
    const int dy = bbSize.height / 10;

    CV_Assert(dx > 0 && dy > 0);

    for(int currentX = 0; currentX < frameSize.width - bbSize.width - dx; currentX += dx)
        for(int currentY = 0; currentY < frameSize.height - bbSize.height - dy; currentY += dy)
            hypothesis.push_back(Hypothesis(currentX, currentY, bbSize));
}

//void tldCascadeClassifier::detect(const Mat_<uchar>& img, std::vector<Response>& responses)
//{
//    Mat_<uchar> standardPatch(STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE);
//    Mat tmp;
//    int dx = initSize.width / 10, dy = initSize.height / 10;
//    Size2d size = img.size();
//    double scale = 1.0;
//    int scaleID;
//    std::vector <Mat> resized_imgs, blurred_imgs;
//    std::vector <Point> varBuffer, ensBuffer;
//    std::vector <int> varScaleIDs, ensScaleIDs;

//    scaleID = 0;
//    resized_imgs.push_back(img);
//    blurred_imgs.push_back(imgBlurred);

//    do
//    {
//        /////////////////////////////////////////////////
//        //Mat bigVarPoints; resized_imgs[scaleID].copyTo(bigVarPoints);
//        /////////////////////////////////////////////////

//        Mat_<double> intImgP, intImgP2;

//        computeIntegralImages(resized_imgs[scaleID], intImgP, intImgP2);

//        for (int i = 0, imax = cvFloor((0.0 + resized_imgs[scaleID].cols - initSize.width) / dx); i < imax; i++)
//        {
//            for (int j = 0, jmax = cvFloor((0.0 + resized_imgs[scaleID].rows - initSize.height) / dy); j < jmax; j++)
//            {
//                if (!patchVariance(intImgP, intImgP2, Point(dx * i, dy * j), initSize))
//                    continue;

//                varBuffer.push_back(Point(dx * i, dy * j));
//                varScaleIDs.push_back(scaleID);

//                ///////////////////////////////////////////////////////
//                //circle(bigVarPoints, *(varBuffer.end() - 1) + Point(initSize.width / 2, initSize.height / 2), 1, cv::Scalar::all(0));
//                ///////////////////////////////////////////////////////
//            }
//        }
//        scaleID++;
//        size.width /= SCALE_STEP;
//        size.height /= SCALE_STEP;
//        scale *= SCALE_STEP;
//        resize(img, tmp, size, 0, 0, DOWNSCALE_MODE);
//        resized_imgs.push_back(tmp.clone());

//        GaussianBlur(resized_imgs[scaleID], tmp, GaussBlurKernelSize, .0f);
//        blurred_imgs.push_back(tmp.clone());

//        ///////////////////////////////////////////////////////
//        //imshow("big variance", bigVarPoints);
//        //waitKey();
//        ///////////////////////////////////////////////////////

//    } while (size.width >= initSize.width && size.height >= initSize.height);

//    for (int i = 0; i < (int)varBuffer.size(); i++)
//    {
//        prepareClassifiers(static_cast<int> (blurred_imgs[varScaleIDs[i]].step[0]));
//        if (ensembleClassifierNum(&blurred_imgs[varScaleIDs[i]].at<uchar>(varBuffer[i].y, varBuffer[i].x)) <= ENSEMBLE_THRESHOLD)
//            continue;
//        ensBuffer.push_back(varBuffer[i]);
//        ensScaleIDs.push_back(varScaleIDs[i]);

//        //////////////////////////////////////////////////////
//        //Mat ensembleOutPut; blurred_imgs[varScaleIDs[i]].copyTo(ensembleOutPut);
//        //rectangle(ensembleOutPut, Rect(varBuffer[i], initSize), Scalar::all(0));
//        //imshow("ensembleOutPut", ensembleOutPut);
//        //waitKey();
//        //////////////////////////////////////////////////////

//    }


//    for (int i = 0; i < (int)ensBuffer.size(); i++)
//    {
//        resample(resized_imgs[ensScaleIDs[i]], Rect2d(ensBuffer[i], initSize), standardPatch);

//        double srValue = Sr(standardPatch);

//        if(srValue > srValue)
//        {
//            Response response;
//            response.confidence = srValue;
//            double curScale = pow(SCALE_STEP, ensScaleIDs[i]);
//            response.bb = Rect2d(ensBuffer[i].x*curScale, ensBuffer[i].y*curScale, initSize.width * curScale, initSize.height * curScale);
//            responses.push_back(response);
//        }
    //    }
//}

}
}
