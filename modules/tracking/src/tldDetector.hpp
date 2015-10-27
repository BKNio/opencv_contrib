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

class CV_EXPORTS_W CascadeClassifier
{
public:
    struct Response
    {
        Rect bb;
        float confidence;
    };

public:
    CascadeClassifier(int preFernMeasurements, int preFerns, Size preFernPatchSize,
                         int numberOfMeasurements, int numberOfFerns, Size fernPatchSize,
                         int numberOfExamples, Size examplePatchSize,
                         int actPositiveExampleNumbers, int actWrappedExamplesNumber, double actGroupTheta);

    void init(const Mat_<uchar> &zeroFrame, const Rect &bb);

    std::vector<std::pair<Rect, double> > detect(const Mat_<uchar> &scaledImage) const;

    void startPExpert(const Mat_<uchar> &image, const Rect &bb);
    void startNExpert(const Mat_<uchar> &image, const Rect &bb);

    void addPositiveExamples(const std::vector< Mat_<uchar> > &examples);
    void addNegativeExamples(const std::vector<Mat_<uchar> > &examples);

    static inline bool greater(const std::pair<Rect, double> &item1, const std::pair<Rect, double> &item2) { return item1.second > item2.second; }
    static inline Rect strip(const std::pair<Rect, double> &item) { return item.first; }

private:

    class PExpert
    {
    public:
        PExpert(Size actFrameSize) : frameSize(actFrameSize) {}
        std::vector<Mat_<uchar> > generatePositiveExamples(const Mat_<uchar> &image, const Rect &bb, int numberOfsurroundBbs, int numberOfSyntheticWarped);
        bool isRectOK(const cv::Rect &rect) const;

    private:
        RNG rng;
        const Size frameSize;

    private:
        std::vector<Rect> generateClosestN(const Rect &bBox, int n);
        std::vector<float> generateRandomValues(float range, int quantity);
        Mat_<uchar> getWarped(const Mat_<uchar> &originalFrame, const Rect &bb, float shiftX, float shiftY, float scale, float rotation);
    };

    class NExpert
    {
    public:
        NExpert() {}
        std::vector< Mat_<uchar> > getNegativeExamples(const Mat_<uchar> &image, const Rect &object, const std::vector<Rect> &detectedObjects, std::string capture);
    };

public:
/*private:*/
    Ptr<VarianceClassifier> varianceClassifier;
    Ptr<FernClassifier> preFernClassifier;
    Ptr<FernClassifier> fernClassifier;
    Ptr<NNClassifier> nnClassifier;


private:
    Ptr<PExpert> pExpert;
    Ptr<NExpert> nExpert;

    mutable RNG rng;

    const Size minimalBBSize;
    const Size standardPatchSize;
    const double scaleStep;
    const double groupRectanglesTheta;
    const int positiveExampleNumbers, wrappedExamplesNumber;

    bool isInited;
    Size originalBBSize;
    Size frameSize;
    std::vector<Hypothesis> hypothesis;

    mutable std::vector<bool> answers;

    std::vector<std::pair<Rect, double> > prepareFinalResult(const Mat_<uchar> &image) const;
    void myGroupRectangles(std::vector<Rect>& rectList, double eps) const;
    static std::vector<Hypothesis> generateHypothesis(const Size frameSize, const Size bbSize, const Size minimalBBSize, double scaleStep);
    static void addScanGrid(const Size frameSize, const Size bbSize, const Size minimalBBSize, std::vector<Hypothesis> &hypothesis, double scale);

    bool isObject(const Mat_<uchar> &candidate) const;

    static bool isObjectPredicate(const CascadeClassifier *pCascadeClassifier, const Mat_<uchar> candidate);
    mutable std::vector<Rect> fernsPositive, nnPositive;

    static Mat_<uchar> debugOutput;
};

}
}

#endif
