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

#include "tldEnsembleClassifier.hpp"

namespace cv
{
namespace tld
{


TLDEnsembleClassifier::TLDEnsembleClassifier(const Rect &roi, size_t actNumberOfFerns, size_t actNumberOfMeasurements):
    originalSize(roi.size()), numberOfFerns(actNumberOfFerns), numberOfMeasurements(actNumberOfMeasurements)
{

    CV_Assert(originalSize.area() * (originalSize.width + originalSize.height) >= numberOfFens * measurePerClassifier); //is it enough measurements

    std::vector<Vec4b> originalMeasurements;

    originalMeasurements.reserve(originalSize.area() * (originalSize.width + originalSize.height));

    for(int i = 0; i < originalSize.width; ++i) //generating all possible horizontal and vertical pixel comprations
    {
        for(int j = 0; j < originalSize.height; ++j)
        {
            Vec4b measure;
            measure.val[0] = i;
            measure.val[1] = j;

            for(int kk = 0; kk < originalSize.width; ++kk)
            {
                if(kk == i)
                    continue;

                measure.val[2] = kk;
                measure.val[3] = j;
                originalMeasurements.push_back(measure);

            }

            for(int kk = 0; kk < originalSize.height; ++kk)
            {
                if(kk == j)
                    continue;

                measure.val[2] = i;
                measure.val[3] = kk;
                originalMeasurements.push_back(measure);
            }

        }
    }

    std::random_shuffle(originalMeasurements.begin(), originalMeasurements.end());

    measurements.assign(numberOfFerns, std::vector<Vec4b>());

    std::vector<Vec4b>::iterator originalMeasurementsIt = originalMeasurements.begin();
    for(size_t i = 0; i < numberOfFerns; ++i)
    {
        measurements[i].assign(originalMeasurementsIt, originalMeasurementsIt + numberOfMeasurements);
        originalMeasurementsIt += numberOfMeasurements;
    }

}

double TLDEnsembleClassifier::getProbability(const Mat_<uchar> &image) const
{
    int position = code(image);
    int posNum = posAndNeg[position].x, negNum = posAndNeg[position].y;

    if (posNum == 0 && negNum == 0)
        return 0.0;
    else
        return double(posNum) / (posNum + negNum);
}

void TLDEnsembleClassifier::integratePositiveExample(const Mat_<uchar> &image)
{

}

void TLDEnsembleClassifier::integrateNegativeExample(const Mat_<uchar> &image)
{

}

void TLDEnsembleClassifier::integrateExample(const Mat_<uchar> &image, bool isPositive)
{

}

int TLDEnsembleClassifier::code(const Mat_<uchar> &image) const
{
    int position = 0;
    for (size_t i = 0; i < measurements.size(); i++)
    {
        position <<= 1;
        if (*(image + rowstep * measurements[i].val[2] + measurements[i].val[0]) <
                *(image + rowstep * measurements[i].val[3] + measurements[i].val[1]))
        {
            position++;
        }
    }
    return position;
}

void TLDEnsembleClassifier::integrate(const Mat_<uchar>& patch, bool isPositive)
{
    int position = code(patch.data, (int)patch.step[0]);
    if (isPositive)
        posAndNeg[position].x++;
    else
        posAndNeg[position].y++;
}

double TLDEnsembleClassifier::posteriorProbabilityFast(const uchar* data) const
{
    int position = codeFast(data);
    double posNum = (double)posAndNeg[position].x, negNum = (double)posAndNeg[position].y;
    if (posNum == 0.0 && negNum == 0.0)
        return 0.0;
    else
        return posNum / (posNum + negNum);
}

void TLDEnsembleClassifier::printClassifier(const Size &displaySize, const Size &internalSize, const std::vector<TLDEnsembleClassifier> &classifiers)
{

    static RNG rng;

    const Mat black(displaySize, CV_8UC3, Scalar::all(0));

    for(std::vector<TLDEnsembleClassifier>::const_iterator classifier = classifiers.begin(); classifier != classifiers.end(); ++classifier )
    {
        Mat copyBlack; black.copyTo(copyBlack);

        for(std::vector<Vec4b>::const_iterator measure = classifier->measurements.begin(); measure != classifier->measurements.end(); ++measure)
        {
            Scalar color(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));

            Point p1(measure->operator[](0) * (double(displaySize.width) / internalSize.width), measure->operator[](1) * (double(displaySize.height) / internalSize.height) );
            Point p2(measure->operator[](2) * (double(displaySize.width) / internalSize.width), measure->operator[](3) * (double(displaySize.height) / internalSize.height) );
                    line(copyBlack, p1, p2, color, 4);
        }

        imshow("printClassifier", copyBlack);
        waitKey();
    }
}

}
}
