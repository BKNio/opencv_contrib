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

#include <list>

#include "precomp.hpp"
#include "opencl_kernels_tracking.hpp"
#include "tldEnsembleClassifier.hpp"
#include "tldUtils.hpp"
#include <opencv2/core.hpp>

namespace cv
{
namespace tld
{
const int STANDARD_PATCH_SIZE = 15;
const int NEG_EXAMPLES_IN_INIT_MODEL = 300;
const int MAX_EXAMPLES_IN_MODEL = 500;
const int MEASURES_PER_CLASSIFIER = 13;
const int GRIDSIZE = STANDARD_PATCH_SIZE;
const int DOWNSCALE_MODE = cv::INTER_LINEAR;
const double THETA_NN = 0.6;
const double CORE_THRESHOLD = 0.5;
const double SCALE_STEP = 1.2;
const double ENSEMBLE_THRESHOLD = 0.5;
const double VARIANCE_THRESHOLD = 0.5;
const double NEXPERT_THRESHOLD = 0.2;

static const cv::Size GaussBlurKernelSize(3, 3);

class TLDDetector
{

public:
    struct Response
    {
        Rect bb;
        float confidence;
    };

public:
    TLDDetector(const Mat &image, const Rect &bb, size_t actMaxNumberOfExamples, size_t actNumberOfFerns);
    ~TLDDetector(){}

    double ensembleClassifierNum(const uchar* data);
    void prepareClassifiers(int rowstep);

    double Sr(const Mat_<uchar>& patch);
    double Sc(const Mat_<uchar>& patch);

    void detect(const Mat& img, const Mat& imgBlurred, std::vector<Response>& patches, Size initSize);

    bool patchVariance(Mat_<double>& intImgP, Mat_<double>& intImgP2, Point pt, Size size);
    void computeIntegralImages(const Mat& img, Mat_<double>& intImgP, Mat_<double>& intImgP2){ integral(img, intImgP, intImgP2, CV_64F); }

    void addPositiveExample(const Mat_<uchar> &example) { addExample(example, positiveExamples); }
    void addNegativeExample(const Mat_<uchar> &example) { addExample(example, negativeExamples); }

    ////////////////////////////////////////////////
    static void printRect(cv::Mat &image, const Rect2d rect);
    static void outputScanningGrid(const cv::Mat &image, const std::vector<cv::Rect2d> &scanGrid);
    ////////////////////////////////////////////////

private:
    void addExample(const Mat_<uchar> &example, std::list<Mat_<uchar> > &storage);

private:

    Ptr<tldFernClassifier> ensebmler;

    const double originalVariance;
    const size_t maxNumberOfExamples;

    RNG rng;
    std::list<Mat_<uchar> > positiveExamples, negativeExamples;
    std::vector<tldFernClassifier> classifiers;

};
}
}

#endif
