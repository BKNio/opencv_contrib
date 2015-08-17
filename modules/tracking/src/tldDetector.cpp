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

TLDDetector::TLDDetector(const Mat &originalImage, const Rect &bb, int actMaxNumberOfExamples, int numberOfFerns, int numberOfMeasurements):
    maxNumberOfExamples(actMaxNumberOfExamples)
{


    Mat_<uchar> image;

    if(originalImage.type() == CV_8UC3)
        cvtColor(originalImage, image, CV_BGR2GRAY);
    else if(originalImage.type() == CV_8U)
        image = originalImage;
    else
        CV_Error(Error::StsBadArg, "Image should be grayscale or RGB");

    if(bb.height < 20 || bb.width < 20)
        CV_Error(Error::StsBadArg, "Minimal height and width should be greater or equal to 20");

    fernClassifier = makePtr<tldFernClassifier>(bb.size(), numberOfFerns, numberOfMeasurements);

    std::vector<Rect> scanGrid;
    generateScanGrid(originalImage.size(), bb.size(), scanGrid);
    ///////////////////////////
    //TLDDetector::outputScanningGrid(image, scanGrid);
    ///////////////////////////

    std::vector<Rect> closest;
    getClosestN(scanGrid, bb, 10, closest);
    ///////////////////////////
    //TLDDetector::outputScanningGrid(image, closest);
    ///////////////////////////

    fernClassifier->printClassifiers(cv::Size(256, 256));

    ///////////////////////////
    //TLDEnsembleClassifier::printClassifier(cv::Size(256, 256), minSize_, detector->classifiers);
    ///////////////////////////

//    Mat_<uchar> warpedPatch(minSize_);

//    for (int i = 0; i < (int)closest.size(); i++)
//    {
//        for (int j = 0; j < 20; j++)
//        {
//            Point2f center;
//            Size2f size;

//            center.x = (float)(closest[i].x + closest[i].width * (0.5 + rng.uniform(-0.01, 0.01)));
//            center.y = (float)(closest[i].y + closest[i].height * (0.5 + rng.uniform(-0.01, 0.01)));

//            size.width = (float)(closest[i].width * rng.uniform((double)0.99, (double)1.01));
//            size.height = (float)(closest[i].height * rng.uniform((double)0.99, (double)1.01));

//            float angle = (float)rng.uniform(-10.0, 10.0);

//            resample(originalImage, RotatedRect(center, size, angle), warpedPatch);
//            GaussianBlur(warpedPatch, blurredWapedPatch, GaussBlurKernelSize, 0.0);
//            for (int k = 0; k < (int)detector->classifiers.size(); k++)
//                detector->classifiers[k].integrate(blurredWapedPatch, true);

//            Mat_<uchar> stdPatch(STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE);
//            resample(originalImage, RotatedRect(center, size, angle), stdPatch);
//            pushIntoModel(stdPatch, true);
//            /////////////////////////////////////////////////////
//            //imshow("blurredWapedPatch", blurredWapedPatch);
//            //imshow("stdPatch", stdPatch);
//            //waitKey();
//            /////////////////////////////////////////////////////

//        }
//    }

//    TLDDetector::generateScanGrid(originalImage.rows, originalImage.cols, minSize_, scanGrid, true);
//    std::vector<int> indices;
//    indices.reserve(NEG_EXAMPLES_IN_INIT_MODEL);
//    while ((int)indices.size() < NEG_EXAMPLES_IN_INIT_MODEL)
//    {
//        int i = rng.uniform((int)0, (int)scanGrid.size());
//        if (std::find(indices.begin(), indices.end(), i) == indices.end() && overlap(boundingBox, scanGrid[i]) < NEXPERT_THRESHOLD)
//        {
//            indices.push_back(i);
//            Mat_<uchar> standardPatch(STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE);
//            resample(originalImage, scanGrid[i], standardPatch);
//            pushIntoModel(standardPatch, false);

//            resample(originalImage, scanGrid[i], warpedPatch);
//            for (int k = 0; k < (int)detector->classifiers.size(); k++)
//                detector->classifiers[k].integrate(warpedPatch, false);
//        }
//    }
}

void TLDDetector::prepareClassifiers(int rowstep)
{
//    for(std::vector<tldFernClassifier>::iterator it = classifiers.begin(); it != classifiers.end(); ++it)
//        it->prepareClassifier(rowstep);
}

double TLDDetector::ensembleClassifierNum(const uchar* data)
{
//    double p = 0.;
//    for(std::vector<tldFernClassifier>::iterator it = classifiers.begin(); it != classifiers.end(); ++it)
//        p += it->posteriorProbabilityFast(data);

//    p /= classifiers.size();
//    return p;

    return 0.;
}

double TLDDetector::Sr(const Mat_<uchar>& patch)
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

double TLDDetector::Sc(const Mat_<uchar>& patch)
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

void TLDDetector::detect(const Mat& img, const Mat& imgBlurred, std::vector<Response>& patches, Size initSize)
{
    patches.clear();
    Mat_<uchar> standardPatch(STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE);
    Mat tmp;
    int dx = initSize.width / 10, dy = initSize.height / 10;
    Size2d size = img.size();
    double scale = 1.0;
    int scaleID;
    std::vector <Mat> resized_imgs, blurred_imgs;
    std::vector <Point> varBuffer, ensBuffer;
    std::vector <int> varScaleIDs, ensScaleIDs;

    scaleID = 0;
    resized_imgs.push_back(img);
    blurred_imgs.push_back(imgBlurred);

    do
    {
        /////////////////////////////////////////////////
        //Mat bigVarPoints; resized_imgs[scaleID].copyTo(bigVarPoints);
        /////////////////////////////////////////////////

        Mat_<double> intImgP, intImgP2;

        computeIntegralImages(resized_imgs[scaleID], intImgP, intImgP2);

        for (int i = 0, imax = cvFloor((0.0 + resized_imgs[scaleID].cols - initSize.width) / dx); i < imax; i++)
        {
            for (int j = 0, jmax = cvFloor((0.0 + resized_imgs[scaleID].rows - initSize.height) / dy); j < jmax; j++)
            {
                if (!patchVariance(intImgP, intImgP2, Point(dx * i, dy * j), initSize))
                    continue;

                varBuffer.push_back(Point(dx * i, dy * j));
                varScaleIDs.push_back(scaleID);

                ///////////////////////////////////////////////////////
                //circle(bigVarPoints, *(varBuffer.end() - 1) + Point(initSize.width / 2, initSize.height / 2), 1, cv::Scalar::all(0));
                ///////////////////////////////////////////////////////
            }
        }
        scaleID++;
        size.width /= SCALE_STEP;
        size.height /= SCALE_STEP;
        scale *= SCALE_STEP;
        resize(img, tmp, size, 0, 0, DOWNSCALE_MODE);
        resized_imgs.push_back(tmp.clone());

        GaussianBlur(resized_imgs[scaleID], tmp, GaussBlurKernelSize, .0f);
        blurred_imgs.push_back(tmp.clone());

        ///////////////////////////////////////////////////////
        //imshow("big variance", bigVarPoints);
        //waitKey();
        ///////////////////////////////////////////////////////

    } while (size.width >= initSize.width && size.height >= initSize.height);

    for (int i = 0; i < (int)varBuffer.size(); i++)
    {
        prepareClassifiers(static_cast<int> (blurred_imgs[varScaleIDs[i]].step[0]));
        if (ensembleClassifierNum(&blurred_imgs[varScaleIDs[i]].at<uchar>(varBuffer[i].y, varBuffer[i].x)) <= ENSEMBLE_THRESHOLD)
            continue;
        ensBuffer.push_back(varBuffer[i]);
        ensScaleIDs.push_back(varScaleIDs[i]);

        //////////////////////////////////////////////////////
        //Mat ensembleOutPut; blurred_imgs[varScaleIDs[i]].copyTo(ensembleOutPut);
        //rectangle(ensembleOutPut, Rect(varBuffer[i], initSize), Scalar::all(0));
        //imshow("ensembleOutPut", ensembleOutPut);
        //waitKey();
        //////////////////////////////////////////////////////

    }


    for (int i = 0; i < (int)ensBuffer.size(); i++)
    {
        resample(resized_imgs[ensScaleIDs[i]], Rect2d(ensBuffer[i], initSize), standardPatch);

        double srValue = Sr(standardPatch);

        if(srValue > srValue)
        {
            Response response;
            response.confidence = srValue;
            double curScale = pow(SCALE_STEP, ensScaleIDs[i]);
            response.bb = Rect2d(ensBuffer[i].x*curScale, ensBuffer[i].y*curScale, initSize.width * curScale, initSize.height * curScale);
            patches.push_back(response);
        }
    }
}

void TLDDetector::printRect(Mat &image, const Rect rect)
{
    rectangle(image, rect, Scalar::all(255));
}

void TLDDetector::outputScanningGrid(const Mat &image, const std::vector<Rect> &scanGrid)
{
    cv::Mat imageCopy; image.copyTo(imageCopy);

    std::vector<Rect> copyScanGrid(scanGrid);

    if(copyScanGrid.size() > 100)
    {
        std::random_shuffle(copyScanGrid.begin(), copyScanGrid.end());
        std::for_each(copyScanGrid.begin(), copyScanGrid.begin() + 100, std::bind1st(std::ptr_fun(printRect), imageCopy));
    }
    else
    {
        std::for_each(copyScanGrid.begin(), copyScanGrid.end(), std::bind1st(std::ptr_fun(printRect), imageCopy));
    }


    cv::imshow("outputScanningGrid",imageCopy);
    cv::waitKey();
}


// Computes the variance of subimage given by box, with the help of two integral
// images intImgP and intImgP2 (sum of squares), which should be also provided.
bool TLDDetector::patchVariance(Mat_<double>& intImgP, Mat_<double>& intImgP2, Point pt, Size size)
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

    return ((p2 - p * p) > VARIANCE_THRESHOLD * originalVariance);
}

void TLDDetector::addPositiveExample(const Mat_<uchar> &example)
{
    addExample(example, positiveExamples); fernClassifier->integratePositiveExample(example);
}

void TLDDetector::addNegativeExample(const Mat_<uchar> &example)
{
    addExample(example, negativeExamples); fernClassifier->integrateNegativeExample(example);
}

void TLDDetector::addExample(const Mat_<uchar> &example, std::list<Mat_<uchar> > &storage)
{
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

}
}
