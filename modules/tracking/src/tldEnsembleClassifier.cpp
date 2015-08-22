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


tldFernClassifier::tldFernClassifier(const Size &initialSize, int actNumberOfFerns, int actNumberOfMeasurements):
    originalSize(initialSize), numberOfFerns(actNumberOfFerns), numberOfMeasurements(actNumberOfMeasurements),
    ferns(actNumberOfFerns), precedents(actNumberOfFerns)
{
    CV_Assert(originalSize.area() * (originalSize.width + originalSize.height) >= numberOfFerns * numberOfMeasurements); //is it enough measurements

    Ferns::value_type measurements;

    measurements.reserve(originalSize.area() * (originalSize.width + originalSize.height));

    for(int i = 0; i < originalSize.width; ++i) //generating all possible horizontal and vertical pixel comprations
    {
        for(int j = 0; j < originalSize.height; ++j)
        {
            Point firstPoint(i,j);

            for(int kk = 0; kk < originalSize.width; ++kk)
            {
                if(kk == i)
                    continue;

                measurements.push_back(std::make_pair(firstPoint, Point(kk, j)));
            }

            for(int kk = 0; kk < originalSize.height; ++kk)
            {
                if(kk == j)
                    continue;

                measurements.push_back(std::make_pair(firstPoint, Point(i, kk)));
            }

        }
    }

    std::random_shuffle(measurements.begin(), measurements.end());

    Precedents::value_type emptyPrecedents(1 << numberOfMeasurements, Point());

    Ferns::value_type::const_iterator originalMeasurementsIt = measurements.begin();
    for(size_t i = 0; i < size_t(numberOfFerns); ++i)
    {
        ferns[i].assign(originalMeasurementsIt, originalMeasurementsIt + numberOfMeasurements);
        originalMeasurementsIt += numberOfMeasurements;

        precedents[i] = emptyPrecedents;
    }

}

double tldFernClassifier::getProbability(const Mat_<uchar> &image) const
{
    CV_Assert(image.size() == originalSize);

    double accumProbability = 0.;
    for(size_t i = 0; i < size_t(numberOfFerns); ++i)
    {
        int position = code(image, ferns[i]);

        int posNum = precedents[i][position].x, negNum = precedents[i][position].y;

        if (posNum != 0 || negNum != 0)
            accumProbability += double(posNum) / (posNum + negNum);
    }

    return accumProbability / numberOfFerns;
}

int tldFernClassifier::code(const Mat_<uchar> &image, const Ferns::value_type &fern) const
{
    int position = 0;
    for(Ferns::value_type::const_iterator measureIt = fern.begin(); measureIt != fern.end(); ++measureIt)
    {
        position <<= 1;
        if(image.at<uchar>(measureIt->first) < image.at<uchar>(measureIt->second))
            position++;
    }
    return position;
}

void tldFernClassifier::integrateExample(const Mat_<uchar> &image, bool isPositive)
{
    for(size_t i = 0; i < ferns.size(); ++i)
    {
        int position = code(image, ferns[i]);

        if(isPositive)
            precedents[i][position].x++;
        else
            precedents[i][position].y++;
    }
}

std::vector<Mat> tldFernClassifier::outputFerns(const Size &displaySize) const
{
    RNG rng;

    double scaleW = double(displaySize.width) / originalSize.width;
    double scaleH = double(displaySize.height) / originalSize.height;

    const Mat black(displaySize, CV_8UC3, Scalar::all(0));

    std::vector<Mat> fernsImages;
    fernsImages.reserve(numberOfFerns);

    for(Ferns::const_iterator fernIt = ferns.begin(); fernIt != ferns.end(); ++fernIt)
    {
        Mat copyBlack; black.copyTo(copyBlack);
        for(Ferns::value_type::const_iterator measureIt = fernIt->begin(); measureIt != fernIt->end(); ++measureIt)
        {
            Scalar color(rng.uniform(20,255), rng.uniform(20,255), rng.uniform(20,255));

            Point p1(cvRound(measureIt->first.x * scaleW), cvRound(measureIt->first.y * scaleH));
            Point p2(cvRound(measureIt->second.x * scaleW), cvRound(measureIt->second.y * scaleH));
            line(copyBlack, p1, p2, color, 4);
        }

        fernsImages.push_back(copyBlack);

    }

    return fernsImages;
}

}
}
