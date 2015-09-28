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

#include "tldDetector.hpp"

namespace cv
{
namespace tld
{

CascadeClassifier::CascadeClassifier(int preMeasure, int preFerns, Size preFernPathSize,
                                           int numberOfMeasurements, int numberOfFerns, Size fernPatchSize,
                                           int numberOfExamples, Size examplePatchSize, double actGroupTheta):
    minimalBBSize(20, 20), scaleStep(1.2f), isInited(false), groupRectanglesTheta(actGroupTheta)
{
    varianceClassifier = makePtr<VarianceClassifier>();
    preFernClassifier = makePtr<FernClassifier>(preMeasure, preFerns, preFernPathSize);
    fernClassifier = makePtr<FernClassifier>(numberOfMeasurements, numberOfFerns, fernPatchSize);
    nnClassifier = makePtr<NNClassifier>(numberOfExamples, examplePatchSize);

}

void CascadeClassifier::init(const Mat_<uchar> &zeroFrame, const Rect &bb, const std::vector< Mat_<uchar> > &examples)
{
    if(bb.width < minimalBBSize.width || bb.height < minimalBBSize.height)
        CV_Error(Error::StsBadArg, "Initial bounding box is too small");

    originalBBSize = bb.size();
    frameSize = zeroFrame.size();
    hypothesis = generateHypothesis(frameSize, originalBBSize, minimalBBSize, scaleStep);

    addPositiveExamples(examples);

    isInited = true;
}


//#define TIME_MEASURE
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

#ifdef TIME_MEASURE
    gettimeofday(&fernStart, NULL);
#endif

    preFernClassifier->isObjects(hypothesis, scaledImage, answers);
    fernClassifier->isObjects(hypothesis, scaledImage, answers);

#ifdef TIME_MEASURE
    gettimeofday(&nnStart, NULL);
#endif

    nnClassifier->isObjects(hypothesis, scaledImage, answers);

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

void CascadeClassifier::addPositiveExamples(const std::vector<Mat_<uchar> > &examples)
{
    varianceClassifier->integratePositiveExamples(examples);
    preFernClassifier->integratePositiveExamples(examples);
    fernClassifier->integratePositiveExamples(examples);
    nnClassifier->integratePositiveExamples(examples);
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
    myGroupRectangles(groupedRects, groupRectanglesTheta);
    for(std::vector<Rect>::const_iterator groupedRectsIt = groupedRects.begin(); groupedRectsIt != groupedRects.end(); ++groupedRectsIt)
        tResults.push_back(std::make_pair(*groupedRectsIt, nnClassifier->calcConfidence(image(*groupedRectsIt))));
#endif

    std::sort(tResults.begin(), tResults.end(), greater);

    std::vector<Rect> finalResult;
    std::transform(tResults.begin(), tResults.end(), std::back_inserter(finalResult), std::ptr_fun(strip));

    return finalResult;
}

void CascadeClassifier::myGroupRectangles(std::vector<Rect> &rectList, double eps) const
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
        float s = 1.f/rweights[i];
        rrects[i] = Rect(saturate_cast<int>(r.x * s),
             saturate_cast<int>(r.y * s),
             saturate_cast<int>(r.width * s),
             saturate_cast<int>(r.height * s));
    }

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

}
}
