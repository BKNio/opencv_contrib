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

#include <list>
#include <vector>

#include "precomp.hpp"

//#define FERN_DEBUG

namespace cv
{
namespace tld
{

struct Hypothesis
{
    Hypothesis() : bb(), scaleId(-1), confidence(-1.){}
    Rect bb;
    int scaleId;
    double confidence;
};

class tldIClassifier
{
public:
    virtual void isObjects(const std::vector<Hypothesis> &hypothesis, const std::vector<Mat_<uchar> > &scaledImages, std::vector<bool> &answers) const = 0;
    virtual void integratePositiveExample(const Mat_<uchar> &image) = 0;
    virtual void integrateNegativeExample(const Mat_<uchar> &image) = 0;

    virtual ~tldIClassifier() {}

};

class CV_EXPORTS_W tldVarianceClassifier : tldIClassifier
{
public:
    tldVarianceClassifier(const Mat_<uchar> &originalImage, const Rect &bb, double actThreshold = 0.5);
    void isObjects(const std::vector<Hypothesis> &hypothesis, const std::vector<Mat_<uchar> > &scaledImages, std::vector<bool> &answers) const;
    void integratePositiveExample(const Mat_<uchar> &) {}
    void integrateNegativeExample(const Mat_<uchar> &) {}

    ~tldVarianceClassifier() {}

private:
    const double originalVariance;
    const double threshold;

private:
    bool isObject(const Rect &bb, const Mat_<double> &sum, const Mat_<double> &sumSq) const;

    static double variance(const Mat_<uchar>& img);
    static double variance(const Mat_<double>& sum, const Mat_<double>& sumSq, const Rect &bb);

};

//#define FERN_DEBUG
#define USE_BLUR
class CV_EXPORTS_W tldFernClassifier : public tldIClassifier
{
public:
    tldFernClassifier(int numberOfMeasurementsPerFern = 13, int reqNumberOfFerns = 100, Size actNormilizedPatchSize = Size(15, 15));

    void isObjects(const std::vector<Hypothesis> &hypothesis, const std::vector<Mat_<uchar> > &scaledImages, std::vector<bool> &answers) const;

    void integratePositiveExample(const Mat_<uchar> &image);
    void integrateNegativeExample(const Mat_<uchar> &image);

    std::vector<Mat> outputFerns(const Size &displaySize) const;

    ~tldFernClassifier() {}

/*private:*/
    const Size normilizedPatchSize;
    /*const int numberOfFerns, numberOfMeasurements;*/
    const double threshold;

    typedef std::vector<std::vector<std::pair<Point, Point> > > Ferns;
    Ferns ferns;

    typedef std::vector<std::vector<Point2i> > Precedents;
    Precedents precedents;

/*private:*/
public:
    bool isObject(const Mat_<uchar> &object) const;
    double getProbability(const Mat_<uchar> &image) const;
    int code(const Mat_<uchar> &image, const Ferns::value_type &fern) const;
    void integrateExample(const Mat_<uchar> &image, bool isPositive);
    static uchar getPixelVale(const Mat_<uchar> &image, const Point2f point);

#ifdef FERN_DEBUG
public:
    mutable cv::Mat debugOutput;
    mutable std::pair<uchar, uchar> vals;
#endif
};

class CV_EXPORTS_W tldNNClassifier : public tldIClassifier
{
public:
    tldNNClassifier(size_t actMaxNumberOfExamples = 500, Size actNormilizedPatchSize = Size(15, 15), double actTheta = 0.5);

    void isObjects(const std::vector<Hypothesis> &hypothesis, const std::vector<Mat_<uchar> > &scaledImages, std::vector<bool> &answers) const;

    void integratePositiveExample(const Mat_<uchar> &example) { addExample(example, positiveExamples); }
    void integrateNegativeExample(const Mat_<uchar> &example) { addExample(example, negativeExamples); }

    std::pair<cv::Mat, cv::Mat> outputModel() const;

    ~tldNNClassifier() {}

private:
    const double theta;
    const size_t maxNumberOfExamples;
    const Size normilizedPatchSize;
    Mat_<uchar> normilizedPatch;

    typedef std::list<Mat_<uchar> > ExampleStorage;
    ExampleStorage positiveExamples, negativeExamples;
    RNG rng;

private:
    bool isObject(const Mat_<uchar> &object) const;
    double Sr(const Mat_<uchar>& patch) const;
    double Sc(const Mat_<uchar>& patch) const;
    void addExample(const Mat_<uchar> &example, std::list<Mat_<uchar> > &storage);
public:
    static float NCC(const Mat_<uchar>& patch1, const Mat_<uchar>& patch2);
};

}
}
