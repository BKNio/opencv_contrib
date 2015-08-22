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


class tldNNClassifier
{
public:
    tldNNClassifier(size_t actMaxNumberOfExamples, Size actPatchSize) : maxNumberOfExamples(actMaxNumberOfExamples), patchSize(actPatchSize){}

    ~tldNNClassifier(){}

    void addPositiveExample(const Mat_<uchar> &example) { addExample(example, positiveExamples); }
    void addNegativeExample(const Mat_<uchar> &example) { addExample(example, negativeExamples); }

    double Sr(const Mat_<uchar>& patch) const;
    double Sc(const Mat_<uchar>& patch) const;

private:
    const size_t maxNumberOfExamples;
    const Size patchSize;

    std::list<Mat_<uchar> > positiveExamples, negativeExamples;
    RNG rng;

private:
    void addExample(const Mat_<uchar> &example, std::list<Mat_<uchar> > &storage);
    static double NCC(const Mat_<uchar>& patch1, const Mat_<uchar>& patch2);
};

class tldDetector
{
public:
    struct Response
    {
        Rect bb;
        float confidence;
    };

public:
    tldDetector(const Mat &originalImage, const Rect &bb, int actMaxNumberOfExamples, int numberOfFerns, int numberOfMeasurements);
    ~tldDetector(){}

    void detect(const Mat_<uchar> &img, std::vector<Response>& responses);

    void addPositiveExample(const Mat_<uchar> &example);
    void addNegativeExample(const Mat_<uchar> &example);

private:
    Ptr<tldFernClassifier> fernClassifier;
    Ptr<tldNNClassifier> nnClassifier;

    const double originalVariance;

private:
    static double variance(const Mat_<uchar>& img);
    static double variance(const Mat_<double>& intImgP, const Mat_<double>& intImgP2, Point pt, Size size);

};

}
}

#endif
