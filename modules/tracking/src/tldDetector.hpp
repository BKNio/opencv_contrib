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

#ifndef OPENCV_TLD_DETECTOR
#define OPENCV_TLD_DETECTOR

#include "precomp.hpp"
#include "tldEnsembleClassifier.hpp"
#include "tldUtils.hpp"

namespace cv
{
namespace tld
{

class CV_EXPORTS_W tldCascadeClassifier
{
public:
    struct Response
    {
        Rect bb;
        float confidence;
    };

public:
    tldCascadeClassifier(const Mat_<uchar> &originalImage, const Rect &bb, int maxNumberOfExamples, int numberOfMeasurements, int numberOfFerns, Size patchSize, int preMeasure, int preFerns, double actThreshold);

    std::vector<std::pair<Rect, double> > detect(const Mat_<uchar> &scaledImage) const;

    void addSyntheticPositive(const Mat_<uchar> &image, const Rect bb, int numberOfsurroundBbs, int numberOfSyntheticWarped);

    void addPositiveExample(const Mat_<uchar> &example);
    void addNegativeExample(const Mat_<uchar> &example);

    static inline bool greater(const std::pair<Rect, double> &item1, const std::pair<Rect, double> &item2)
    {
        return item1.second > item2.second;
    }


/*private:*/
    Ptr<tldVarianceClassifier> varianceClassifier;
    Ptr<tldFernClassifier> preFernClassifier;
    Ptr<tldFernClassifier> fernClassifier;
    Ptr<tldNNClassifier> nnClassifier;
    mutable RNG rng;

    const Size minimalBBSize;
    const Size standardPatchSize;
    const Size originalBBSize;
    const Size frameSize;
    const double scaleStep;
    mutable std::vector<Hypothesis> hypothesis;
    mutable std::vector<bool> answers;

/*private:*/
    std::vector<Rect> generateClosestN(const Rect &bBox, size_t N) const;
    std::vector<Rect> generateSurroundingRects(const Rect &bBox, size_t N) const;
    std::vector<Rect> generateAndSelectRects(const Rect &bBox, int n, float rangeStart, float rangeEnd) const;
    std::vector< std::pair<Rect, double> > prepareFinalResult(const Mat_<uchar> &image) const;

    bool isRectOK(const cv::Rect &rect) const;

    Mat_<uchar> randomWarp(const Mat_<uchar> &originalFrame, Rect bb, float shiftRangePercent, float scaleRangePercent, float rotationRangeDegrees);
    static std::vector<Hypothesis> generateHypothesis(const Size frameSize, const Size bbSize, const Size minimalBBSize, double scaleStep);
    static void addScanGrid(const Size frameSize, const Size bbSize, const Size minimalBBSize, std::vector<Hypothesis> &hypothesis);
};

}
}

#endif
