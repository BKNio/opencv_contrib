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

#ifndef TLD_ENSEMBLE
#define TLD_ENSEMBLE

#include "precomp.hpp"

#include <list>
#include <vector>

namespace cv
{

namespace tld
{

struct Hypothesis
{
    Hypothesis() : bb() {}
    Hypothesis(int x, int y, Size size, double actScale) : bb(Point(x,y), size), scale(actScale) {}
    Rect bb;
    double scale;
};

struct Answers
{
    Answers() : confidence(-1.) {}
    void operator = (bool isObject) { confidence = isObject ? 1. : -1.; }
    operator bool() const {return confidence > 0.;}

    double confidence;
};

class CV_EXPORTS_W VarianceClassifier
{
public:
    VarianceClassifier(double actLowCoeff = 0.5, double actHighCoeff = 2.);
    void isObjects(const std::vector<Hypothesis> &hypothesis, const Mat_<uchar> &image, std::vector<Answers> &answers) const;
    void integratePositiveExamples(const std::vector< Mat_<uchar> > &examples);

    static double variance(const Mat_<uchar>& img);

    ~VarianceClassifier() {}

private:
    double actVariance;
    const double lowCoeff, hightCoeff;
    mutable Mat_<double> integral, integralSq;

private:
    bool isObject(const Rect &bb) const;
    static double variance(const Mat_<double>& sum, const Mat_<double>& sumSq, const Rect &bb);

};

class CV_EXPORTS_W FernClassifier
{
public:
    FernClassifier(int numberOfMeasurementsPerFern, int reqNumberOfFerns, Size actNormilizedPatchSize, double actThreshold = 0.5);

    void isObjects(const std::vector<Hypothesis> &hypothesis, const Mat_<uchar> &image, std::vector<Answers> &answers) const;

    void integratePositiveExamples(const std::vector< Mat_<uchar> > &examples);
    void integrateNegativeExamples(const std::vector< Mat_<uchar> > &examples);


    ~FernClassifier() {}

private:

    const Size patchSize;
    const double threshold;
    const int minSqDist;

    typedef std::vector< std::vector<std::pair<Point, Point> > > Ferns;
    Ferns ferns;

    typedef std::vector< std::vector<Point> > Precedents;
    Precedents precedents;

    RNG rng;

private:
    bool isObject(const Mat_<uchar> &image) const;
    double getProbability(const Mat_<uchar> &image) const;
    int code(const Mat_<uchar> &image, const Ferns::value_type &fern) const;
    void integrateExample(const Mat_<uchar> &image, bool isPositive);


    void saveFern(const std::string &path) const;
    static void compareFerns(const std::string &pathToFern1, const std::string &pathToFern2);
    std::vector<Mat> outputFerns(const Size &displaySize) const;
};


class CV_EXPORTS_W NNClassifier
{
public:
    NNClassifier(size_t actMaxNumberOfExamples, Size actNormilizedPatchSize, double actTheta = 0.5);

    void isObjects(const std::vector<Hypothesis> &hypothesis, const Mat_<uchar> &images, std::vector<Answers> &answers) const;

    void integratePositiveExamples(const std::vector< Mat_<uchar> > &examples);
    void integrateNegativeExamples(const std::vector< Mat_<uchar> > &examples);

    double calcConfidenceTracker(const Mat_<uchar> &image) const;

    ~NNClassifier() {}

private:
    const double theta;
    const size_t maxNumberOfExamples;
    const Size normilizedPatchSize;
    Mat_<uchar> normilizedPatch;

    typedef std::list<Mat_<uchar> > ExampleStorage;
    ExampleStorage positiveExamples, negativeExamples;
    RNG rng;

private:
    bool isObject(const Mat_<uchar> &image) const;

    double Sr(const Mat_<uchar>& patch) const;
    double Sc(const Mat_<uchar>& patch, bool isForTracker = false) const;
    void addExample(const Mat_<uchar> &example, std::list<Mat_<uchar> > &storage);
    static float NCC(const Mat_<uchar>& patch1, const Mat_<uchar>& patch2);


    std::pair<cv::Mat, cv::Mat> outputModel(int positiveMark = -1, int negativeMark = -1) const;
    std::pair<Mat, Mat> getModelWDecisionMarks(const Mat_<uchar> &image, double previousConf);
    double debugSr(const Mat_<uchar> &patch, int &positiveDecisitionExample, int &negativeDecisionExample);

};

}
}

#endif
