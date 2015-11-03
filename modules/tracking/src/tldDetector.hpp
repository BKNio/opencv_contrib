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

#include <map>

#include "precomp.hpp"
#include "tldEnsembleClassifier.hpp"
#include "tldUtils.hpp"

namespace cv
{
namespace tld
{

class CV_EXPORTS_W CascadeClassifier
{

public:
    CascadeClassifier(int numberOfMeasurements, int numberOfFerns, Size fernPatchSize,
                         int numberOfExamples, Size examplePatchSize,
                         int actPositiveExampleNumbers, int actWrappedExamplesNumber);

    void init(const Mat_<uchar> &zeroFrame, const Rect &bb);

    std::vector<std::pair<Rect, double> > detect(const Mat_<uchar> &scaledImage) const;

    void startPExpert(const Mat_<uchar> &image, const Rect &bb);
    void startNExpert(const Mat_<uchar> &image, const Rect &bb);

    void addPositiveExamples(const std::vector< Mat_<uchar> > &examples);
    void addNegativeExamples(const std::vector<Mat_<uchar> > &examples);

private:

    class PExpert
    {
    public:
        PExpert(Size _frameSize, Size _dstSize) : frameSize(_frameSize), dstSize(_dstSize) {}
        std::vector<Mat_<uchar> > generatePositiveExamples(const Mat_<uchar> &image, const Rect &bb, int, int numberOfSyntheticWarped);

    private:
        RNG rng;
        const Size frameSize;
        const Size dstSize;

    private:
        bool isRectOK(const cv::Rect &rect) const;
        std::vector<float> generateRandomValues(float range, int quantity);
        Mat_<uchar> getWarped(const Mat_<uchar> &originalFrame, const Rect &bb, float shiftX, float shiftY, float scale, float rotation);
    };

    class NExpert
    {
    public:
        NExpert(Size _dstSize) : dstSize(_dstSize) {}
        std::vector< Mat_<uchar> > getNegativeExamples(const Mat_<uchar> &image, const Rect &object, const std::vector<Rect> &detectedObjects, std::string capture);

    private:
        const Size dstSize;
    };

public:
/*private:*/
    Ptr<VarianceClassifier> varianceClassifier;
    Ptr<FernClassifier> fernClassifier;
    Ptr<NNClassifier> nnClassifier;


private:
    Ptr<PExpert> pExpert;
    Ptr<NExpert> nExpert;

    mutable RNG rng;
    mutable std::map<double, Mat_<uchar> > scaledStorage;

    const Size minimalBBSize;
    const Size patchSize;
    const double scaleStep;
    const int positiveExampleNumbers, wrappedExamplesNumber;

    bool isInited;
    Size originalBBSize;
    Size frameSize;
    std::vector<Hypothesis> hypothesis;
    mutable std::vector<Answers> answers;

    std::vector<std::pair<Rect, double> > prepareFinalResult(const Mat_<uchar> &image) const;
    static std::vector<Hypothesis> generateHypothesis(const Size frameSize, const Size bbSize, const Size minimalBBSize, double scaleStep);
    static void addScanGrid(const Size frameSize, const Size bbSize, const Size minimalBBSize, std::vector<Hypothesis> &hypothesis, double scale);
    static bool removePredicate(const std::pair<Rect, double> item, const std::vector< std::pair<Rect, double> > &storage);
    static bool containPredicate(const std::pair<Rect, double> item, const std::pair<Rect, double> &refItem);

    mutable std::vector<Rect> fernsPositive, nnPositive;

    static Mat_<uchar> debugOutput;
};

}
}

#endif
