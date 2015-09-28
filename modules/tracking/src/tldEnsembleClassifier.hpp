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

#include <list>
#include <vector>

#include "precomp.hpp"

//#define FERN_DEBUG
//#define FERN_PROFILE

namespace cv
{
namespace tld
{

struct Hypothesis
{
    Hypothesis() : bb(), confidence(-1.){}
    Hypothesis(int x, int y, Size size, double actScale) : bb(Point(x,y), size), confidence(-1.), scale(actScale) {}
    Rect bb;
    double confidence;
    double scale;
};

class tldIClassifier
{
public:
    virtual void isObjects(const std::vector<Hypothesis> &hypothesis, const Mat_<uchar> &image, std::vector<bool> &answers) const = 0;
    virtual void integratePositiveExample(const Mat_<uchar> &image) = 0;
    virtual void integrateNegativeExample(const Mat_<uchar> &image) = 0;

    virtual ~tldIClassifier() {}

};

class CV_EXPORTS_W tldVarianceClassifier : tldIClassifier
{
public:
    tldVarianceClassifier(const Mat_<uchar> &originalImage, double actThreshold = 0.5);
    void isObjects(const std::vector<Hypothesis> &hypothesis, const Mat_<uchar> &image, std::vector<bool> &answers) const;
    void integratePositiveExample(const Mat_<uchar> &) {}
    void integrateNegativeExample(const Mat_<uchar> &) {}

    ~tldVarianceClassifier() {}

/*private:*/
    const double originalVariance;
    const double coefficient;
    const double threshold;

/*private:*/
    bool isObject(const Rect &bb, const Mat_<double> &sum, const Mat_<double> &sumSq) const;

    static double variance(const Mat_<uchar>& img);
    static double variance(const Mat_<double>& sum, const Mat_<double>& sumSq, const Rect &bb);

};

//#define FERN_DEBUG
//#define USE_BLUR
//#define FERN_PROFILE
class CV_EXPORTS_W tldFernClassifier : public tldIClassifier
{
public:
    tldFernClassifier(int numberOfMeasurementsPerFern, int reqNumberOfFerns, Size actNormilizedPatchSize = Size(15, 15), double actThreshold = 0.5);

    void isObjects(const std::vector<Hypothesis> &hypothesis, const Mat_<uchar> &image, std::vector<bool> &answers) const;

    void integratePositiveExample(const Mat_<uchar> &image);
    void integrateNegativeExample(const Mat_<uchar> &image);

    std::vector<Mat> outputFerns(const Size &displaySize) const;

    ~tldFernClassifier() {}

/*private:*/
    const Size normilizedPatchSize;
    /*const int numberOfFerns, numberOfMeasurements;*/
    const double threshold;
    const int minSqDist;

    typedef std::vector<std::vector<std::pair<Point, Point> > > Ferns;
    Ferns ferns;
    //Ferns::value_type measurements;

    typedef std::vector<std::vector<Point_<unsigned long> > > Precedents;
    Precedents precedents;

/*private:*/
public:
    bool isObject(const Mat_<uchar> &image) const;
    double getProbability(const Mat_<uchar> &image) const;
    int code(const Mat_<uchar> &image, const Ferns::value_type &fern) const;
    void integrateExample(const Mat_<uchar> &image, bool isPositive);
    static uchar getPixelVale(const Mat_<uchar> &image, const Point2f point);

#ifdef FERN_DEBUG
public:
    mutable cv::Mat debugOutput;
    mutable std::pair<uchar, uchar> vals;
#endif

#ifdef FERN_PROFILE
    mutable double codeTime;
    mutable double acsessTime;
    mutable double calcTime;
#endif

};
//#define NNDEBUG
class CV_EXPORTS_W tldNNClassifier : public tldIClassifier
{
public:
    tldNNClassifier(size_t actMaxNumberOfExamples = 500, Size actNormilizedPatchSize = Size(15, 15), double actTheta = 0.5);

    void isObjects(const std::vector<Hypothesis> &hypothesis, const Mat_<uchar> &images, std::vector<bool> &answers) const;

    void integratePositiveExample(const Mat_<uchar> &example) { addExample(example, positiveExamples); }
    void integrateNegativeExample(const Mat_<uchar> &example) { addExample(example, negativeExamples); }

    std::pair<cv::Mat, cv::Mat> outputModel() const;

    ~tldNNClassifier() {}

//private:
    const double theta;
    const size_t maxNumberOfExamples;
    const Size normilizedPatchSize;
    Mat_<uchar> normilizedPatch;

    typedef std::list<Mat_<uchar> > ExampleStorage;
    ExampleStorage positiveExamples, negativeExamples;
    RNG rng;

public:
/*private:*/
    bool isObject(const Mat_<uchar> &image) const;
    double calcConfidence(const Mat_<uchar> &image) const;
    double Sr(const Mat_<uchar>& patch) const;
    double Sc(const Mat_<uchar>& patch) const;
    void addExample(const Mat_<uchar> &example, std::list<Mat_<uchar> > &storage);
public:
    static float NCC(const Mat_<uchar>& patch1, const Mat_<uchar>& patch2);


#ifdef NNDEBUG
public:
    mutable ExampleStorage::const_iterator positive, negative;
    Mat_<uchar> outputPrecedents(const Mat_<uchar> &object)
    {
        Mat_<uchar> resizeObject;

        if(!object.empty())
            resize(object, resizeObject, normilizedPatchSize);
        Mat_<uchar> precedents(cv::Size(3*normilizedPatchSize.width, normilizedPatchSize.height), 0u);

        positive->copyTo(precedents(Rect(Point(), normilizedPatchSize)));
        negative->copyTo(precedents(Rect(Point(normilizedPatchSize.width,0), normilizedPatchSize)));
        if(!object.empty())
            resizeObject.copyTo(precedents(Rect(Point(2*normilizedPatchSize.width,0), normilizedPatchSize)));

        return precedents;
    }

#endif


};

}
}

#endif
