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

#include "tldDetector.hpp"

namespace cv
{
namespace tld
{


double tldNNClassifier::Sr(const Mat_<uchar> &patch) const
{
    double splus = 0., sminus = 0.;
    for(std::list<Mat_<uchar> >::const_iterator it = positiveExamples.begin(); it != positiveExamples.end(); ++it)
        splus = std::max(splus, 0.5 * (NCC(*it, patch) + 1.0));

    for(std::list<Mat_<uchar> >::const_iterator it = positiveExamples.begin(); it != negativeExamples.end(); ++it)
        sminus = std::max(sminus, 0.5 * (NCC(*it, patch) + 1.0));

    if (splus + sminus == 0.0)
        return 0.0;

    return splus / (sminus + splus);
}

double tldNNClassifier::Sc(const Mat_<uchar> &patch) const
{
    double splus = 0., sminus = 0.;

    size_t mediana = positiveExamples.size() / 2 + positiveExamples.size() % 2;

    std::list<Mat_<uchar> >::const_iterator end = positiveExamples.begin();
    for(size_t i = 0; i < mediana; ++i) ++end;

    for(std::list<Mat_<uchar> >::const_iterator it = positiveExamples.begin(); it != end; ++it)
        splus = std::max(splus, 0.5 * (NCC(*it, patch) + 1.0));

    for(std::list<Mat_<uchar> >::const_iterator it = positiveExamples.begin(); it != negativeExamples.end(); ++it)
        sminus = std::max(sminus, 0.5 * (NCC(*it, patch) + 1.0));

    if (splus + sminus == 0.0)
        return 0.0;

    return splus / (sminus + splus);
}

void tldNNClassifier::addExample(const Mat_<uchar> &example, std::list<Mat_<uchar> > &storage)
{
    CV_Assert(patchSize == example.size());
    CV_Assert(storage.size() <= maxNumberOfExamples);

    if(storage.size() == maxNumberOfExamples)
    {
        int randomIndex = rng.uniform(0, int(maxNumberOfExamples));
        std::list<Mat_<uchar> >::iterator it = storage.begin();
        for(int i = 0; i < randomIndex; ++i)
            ++it;

        storage.erase(it);
    }

    storage.push_back(example.clone());
}

double tldNNClassifier::NCC(const Mat_<uchar> &patch1, const Mat_<uchar> &patch2)
{
    CV_Assert(patch1.size() == patch2.size());

    int N = patch1.size().area();

    double s1 = 0., s2 = 0., n1 = 0., n2 = 0., prod = 0.;
    for( int i = 0; i < patch1.rows; i++ )
    {
        for( int j = 0; j < patch1.cols; j++ )
        {
            int p1 = patch1(i, j), p2 = patch2(i, j);
            s1 += p1; s2 += p2;
            n1 += (p1 * p1); n2 += (p2 * p2);
            prod += (p1 * p2);
        }
    }

    double sq1 = sqrt(n1/N - (s1/N)*(s1/N)), sq2 = sqrt(n2/N - (s2/N)*(s2/N));

    double ares = (prod + s1 * s2 / (N * N)) / (sq1 * sq2);

    return ares / N;
}

tldDetector::tldDetector(const Mat &originalImage, const Rect &bb, int maxNumberOfExamples, int numberOfFerns, int numberOfMeasurements):
    originalVariance(variance(originalImage(bb)))
{
    if(bb.width < minimalBBSize.width || bb.height < minimalBBSize.height)
        CV_Error(Error::StsBadArg, "Initial bounding box is too small");

    nnClassifier = makePtr<tldNNClassifier>(maxNumberOfExamples, standardPath);
    fernClassifier = makePtr<tldFernClassifier>(bb.size(), numberOfFerns, numberOfMeasurements);
}

double tldDetector::variance(const Mat_<uchar>& img)
{
    double p = 0, p2 = 0;
    for( int i = 0; i < img.rows; i++ )
    {
        for( int j = 0; j < img.cols; j++ )
        {
            p += img.at<uchar>(i, j);
            p2 += img.at<uchar>(i, j) * img.at<uchar>(i, j);
        }
    }
    p /= (img.cols * img.rows);
    p2 /= (img.cols * img.rows);
    return p2 - p * p;
}

double tldDetector::variance(const Mat_<double>& intImgP, const Mat_<double>& intImgP2, Point pt, Size size)
{
    int x = (pt.x), y = (pt.y), width = (size.width), height = (size.height);

    CV_Assert(0 <= x && (x + width) < intImgP.cols && (x + width) < intImgP2.cols);
    CV_Assert(0 <= y && (y + height) < intImgP.rows && (y + height) < intImgP2.rows);

    double p = 0, p2 = 0;
    double A, B, C, D;

    A = intImgP(y, x);
    B = intImgP(y, x + width);
    C = intImgP(y + height, x);
    D = intImgP(y + height, x + width);
    p = (A + D - B - C) / (width * height);

    A = intImgP2(y, x);
    B = intImgP2(y, x + width);
    C = intImgP2(y + height, x);
    D = intImgP2(y + height, x + width);
    p2 = (A + D - B - C) / (width * height);

    return (p2 - p * p);
}

void tldDetector::detect(const Mat_<uchar>& img, std::vector<Response>& responses)
{
//    Mat_<uchar> standardPatch(STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE);
//    Mat tmp;
//    int dx = initSize.width / 10, dy = initSize.height / 10;
//    Size2d size = img.size();
//    double scale = 1.0;
//    int scaleID;
//    std::vector <Mat> resized_imgs, blurred_imgs;
//    std::vector <Point> varBuffer, ensBuffer;
//    std::vector <int> varScaleIDs, ensScaleIDs;

//    scaleID = 0;
//    resized_imgs.push_back(img);
//    blurred_imgs.push_back(imgBlurred);

//    do
//    {
//        /////////////////////////////////////////////////
//        //Mat bigVarPoints; resized_imgs[scaleID].copyTo(bigVarPoints);
//        /////////////////////////////////////////////////

//        Mat_<double> intImgP, intImgP2;

//        computeIntegralImages(resized_imgs[scaleID], intImgP, intImgP2);

//        for (int i = 0, imax = cvFloor((0.0 + resized_imgs[scaleID].cols - initSize.width) / dx); i < imax; i++)
//        {
//            for (int j = 0, jmax = cvFloor((0.0 + resized_imgs[scaleID].rows - initSize.height) / dy); j < jmax; j++)
//            {
//                if (!patchVariance(intImgP, intImgP2, Point(dx * i, dy * j), initSize))
//                    continue;

//                varBuffer.push_back(Point(dx * i, dy * j));
//                varScaleIDs.push_back(scaleID);

//                ///////////////////////////////////////////////////////
//                //circle(bigVarPoints, *(varBuffer.end() - 1) + Point(initSize.width / 2, initSize.height / 2), 1, cv::Scalar::all(0));
//                ///////////////////////////////////////////////////////
//            }
//        }
//        scaleID++;
//        size.width /= SCALE_STEP;
//        size.height /= SCALE_STEP;
//        scale *= SCALE_STEP;
//        resize(img, tmp, size, 0, 0, DOWNSCALE_MODE);
//        resized_imgs.push_back(tmp.clone());

//        GaussianBlur(resized_imgs[scaleID], tmp, GaussBlurKernelSize, .0f);
//        blurred_imgs.push_back(tmp.clone());

//        ///////////////////////////////////////////////////////
//        //imshow("big variance", bigVarPoints);
//        //waitKey();
//        ///////////////////////////////////////////////////////

//    } while (size.width >= initSize.width && size.height >= initSize.height);

//    for (int i = 0; i < (int)varBuffer.size(); i++)
//    {
//        prepareClassifiers(static_cast<int> (blurred_imgs[varScaleIDs[i]].step[0]));
//        if (ensembleClassifierNum(&blurred_imgs[varScaleIDs[i]].at<uchar>(varBuffer[i].y, varBuffer[i].x)) <= ENSEMBLE_THRESHOLD)
//            continue;
//        ensBuffer.push_back(varBuffer[i]);
//        ensScaleIDs.push_back(varScaleIDs[i]);

//        //////////////////////////////////////////////////////
//        //Mat ensembleOutPut; blurred_imgs[varScaleIDs[i]].copyTo(ensembleOutPut);
//        //rectangle(ensembleOutPut, Rect(varBuffer[i], initSize), Scalar::all(0));
//        //imshow("ensembleOutPut", ensembleOutPut);
//        //waitKey();
//        //////////////////////////////////////////////////////

//    }


//    for (int i = 0; i < (int)ensBuffer.size(); i++)
//    {
//        resample(resized_imgs[ensScaleIDs[i]], Rect2d(ensBuffer[i], initSize), standardPatch);

//        double srValue = Sr(standardPatch);

//        if(srValue > srValue)
//        {
//            Response response;
//            response.confidence = srValue;
//            double curScale = pow(SCALE_STEP, ensScaleIDs[i]);
//            response.bb = Rect2d(ensBuffer[i].x*curScale, ensBuffer[i].y*curScale, initSize.width * curScale, initSize.height * curScale);
//            responses.push_back(response);
//        }
    //    }
}

void tldDetector::addPositiveExample(const Mat_<uchar> &example)
{

}

void tldDetector::addNegativeExample(const Mat_<uchar> &example)
{

}

}
}
