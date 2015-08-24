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

#include "tldModel.hpp"

namespace cv
{
namespace tld
{
TrackerTLDModel::TrackerTLDModel(TrackerTLD::Params params, const Mat& image, const Rect &boundingBox):
    minSize_(boundingBox.size()), params_(params), boundingBox_(boundingBox)
{
    detector = makePtr<tldCascadeClassifier>(image, boundingBox, 500, 50, 13);

    std::vector<Rect> scanGrid;
    //generateScanGrid(originalImage.size(), bb.size(), scanGrid);
    ///////////////////////////
    //TLDDetector::outputScanningGrid(image, scanGrid);
    ///////////////////////////

    std::vector<Rect> closest;
    //getClosestN(scanGrid, bb, 10, closest);
    ///////////////////////////
    //TLDDetector::outputScanningGrid(image, closest);
    ///////////////////////////



//    Mat_<uchar> warpedPatch(bb.size()), blurredWarpedPatch(bb.size);

//    for (size_t i = 0; i < closest.size(); i++)
//    {
//        for (size_t j = 0; j < 20; j++)
//        {
//            Point2f center;
//            center.x = closest[i].x + closest[i].width * (0.5 + rng.uniform(-0.01, 0.01));
//            center.y = closest[i].y + closest[i].height * (0.5 + rng.uniform(-0.01, 0.01));

//            Size2f size;
//            size.width = closest[i].width * rng.uniform(0.99, 1.01);
//            size.height = closest[i].height * rng.uniform(0.99, 1.01);

//            float angle = rng.uniform(-10.0, 10.0);

//            resample(originalImage, RotatedRect(center, size, angle), warpedPatch);
//            addExample(warpedPatch, positiveExamples);

//            GaussianBlur(warpedPatch, blurredWarpedPatch, GaussBlurKernelSize, 0.0);
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

//void TLDDetector::outputScanningGrid(const Mat &image, const std::vector<Rect> &scanGrid)
//{
//    cv::Mat imageCopy; image.copyTo(imageCopy);

//    std::vector<Rect> copyScanGrid(scanGrid);

//    if(copyScanGrid.size() > 100)
//    {
//        std::random_shuffle(copyScanGrid.begin(), copyScanGrid.end());
//        std::for_each(copyScanGrid.begin(), copyScanGrid.begin() + 100, std::bind1st(std::ptr_fun(printRect), imageCopy));
//    }
//    else
//    {
//        std::for_each(copyScanGrid.begin(), copyScanGrid.end(), std::bind1st(std::ptr_fun(printRect), imageCopy));
//    }


//    cv::imshow("outputScanningGrid",imageCopy);
//    cv::waitKey();
//}

//void tldDetector::printRect(Mat &image, const Rect rect)
//{
//    rectangle(image, rect, Scalar::all(255));
//}

void TrackerTLDModel::integrateRelabeled(Mat& img, Mat& imgBlurred, const std::vector<tldCascadeClassifier::Response>& patches)
{
//    Mat_<uchar> standardPatch(STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE), blurredPatch(minSize_);
//    int positiveIntoModel = 0, negativeIntoModel = 0, positiveIntoEnsemble = 0, negativeIntoEnsemble = 0;
//    for (int k = 0; k < (int)patches.size(); k++)
//    {
//        if (patches[k].shouldBeIntegrated)
//        {
//            resample(img, patches[k].rect, standardPatch);
//            if (patches[k].isObject)
//            {
//                positiveIntoModel++;
//                pushIntoModel(standardPatch, true);
//            }
//            else
//            {
//                negativeIntoModel++;
//                pushIntoModel(standardPatch, false);
//            }
//        }

//#ifdef CLOSED_LOOP
//        if (patches[k].shouldBeIntegrated || !patches[k].isPositive)
//#else
//        if (patches[k].shouldBeIntegrated)
//#endif
//        {
//            resample(imgBlurred, patches[k].rect, blurredPatch);
//            if (patches[k].isObject)
//                positiveIntoEnsemble++;
//            else
//                negativeIntoEnsemble++;
//            for (int i = 0; i < (int)detector->classifiers.size(); i++)
//                detector->classifiers[i].integrate(blurredPatch, patches[k].isObject);
//        }
//    }
}

void TrackerTLDModel::integrateAdditional(const std::vector<Mat_<uchar> >& eForModel, const std::vector<Mat_<uchar> >& eForEnsemble, bool isPositive)
{
//    int positiveIntoModel = 0, negativeIntoModel = 0, positiveIntoEnsemble = 0, negativeIntoEnsemble = 0;
//    if ((int)eForModel.size() == 0) return;

//    //int64 e1, e2;
//    //double t;
//    //e1 = getTickCount();
//    for (int k = 0; k < (int)eForModel.size(); k++)
//    {
//        double sr = detector->Sr(eForModel[k]);
//        if ((sr > THETA_NN) != isPositive)
//        {
//            if (isPositive)
//            {
//                positiveIntoModel++;
//                pushIntoModel(eForModel[k], true);
//            }
//            else
//            {
//                negativeIntoModel++;
//                pushIntoModel(eForModel[k], false);
//            }
//        }
//        double p = 0;
//        for (int i = 0; i < (int)detector->classifiers.size(); i++)
//            p += detector->classifiers[i].posteriorProbability(eForEnsemble[k].data, (int)eForEnsemble[k].step[0]);
//        p /= detector->classifiers.size();
//        if ((p > ENSEMBLE_THRESHOLD) != isPositive)
//        {
//            if (isPositive)
//                positiveIntoEnsemble++;
//            else
//                negativeIntoEnsemble++;
//            for (int i = 0; i < (int)detector->classifiers.size(); i++)
//                detector->classifiers[i].integrate(eForEnsemble[k], isPositive);
//        }
//    }
}

void TrackerTLDModel::ocl_integrateAdditional(const std::vector<Mat_<uchar> >& eForModel, const std::vector<Mat_<uchar> >& eForEnsemble, bool isPositive)
{
//    int positiveIntoModel = 0, negativeIntoModel = 0, positiveIntoEnsemble = 0, negativeIntoEnsemble = 0;
//    if ((int)eForModel.size() == 0) return;

//    //int64 e1, e2;
//    //double t;
//    //e1 = getTickCount();

//    //Prepare batch of patches
//    int numOfPatches = (int)eForModel.size();
//    Mat_<uchar> stdPatches(numOfPatches, 225);
//    double *resultSr = new double[numOfPatches];
//    double *resultSc = new double[numOfPatches];
//    uchar *patchesData = stdPatches.data;
//    for (int i = 0; i < numOfPatches; i++)
//    {
//        uchar *stdPatchData = eForModel[i].data;
//        for (int j = 0; j < 225; j++)
//            patchesData[225 * i + j] = stdPatchData[j];
//    }

//    //Calculate Sr and Sc batches
//    detector->ocl_batchSrSc(stdPatches, resultSr, resultSc, numOfPatches);

//    for (int k = 0; k < (int)eForModel.size(); k++)
//    {
//        double sr = resultSr[k];
//        if ((sr > THETA_NN) != isPositive)
//        {
//            if (isPositive)
//            {
//                positiveIntoModel++;
//                pushIntoModel(eForModel[k], true);
//            }
//            else
//            {
//                negativeIntoModel++;
//                pushIntoModel(eForModel[k], false);
//            }
//        }
//        double p = 0;
//        for (int i = 0; i < (int)detector->classifiers.size(); i++)
//            p += detector->classifiers[i].posteriorProbability(eForEnsemble[k].data, (int)eForEnsemble[k].step[0]);
//        p /= detector->classifiers.size();
//        if ((p > ENSEMBLE_THRESHOLD) != isPositive)
//        {
//            if (isPositive)
//                positiveIntoEnsemble++;
//            else
//                negativeIntoEnsemble++;
//            for (int i = 0; i < (int)detector->classifiers.size(); i++)
//                detector->classifiers[i].integrate(eForEnsemble[k], isPositive);
//        }
//    }

}

//Push the patch to the model
void TrackerTLDModel::pushIntoModel(const Mat_<uchar>& example, bool isPositive)
{

//    int &proxyN = isPositive ? timeStampPositiveNext : timeStampNegativeNext;
//    std::vector<int> &proxyT = isPositive ? timeStampsPositive : timeStampsNegative;
//    Mat &model = isPositive ? posExp : negExp;

//    CV_Assert(int(proxyT.size()) <= 500);

//    if()

//    if (isPositive)
//    {
//        if (posNum < 500)
//        {
//            uchar *patchPtr = example.data;
//            uchar *modelPtr = posExp.data;

//            for (int i = 0; i < STANDARD_PATCH_SIZE*STANDARD_PATCH_SIZE; i++)
//                modelPtr[posNum*STANDARD_PATCH_SIZE*STANDARD_PATCH_SIZE + i] = patchPtr[i];

//            posNum++;
//        }

//        //proxyV = &positiveExamples;
//        proxyN = &timeStampPositiveNext;
//        proxyT = &timeStampsPositive;
//    }
//    else
//    {
//        if (negNum < 500)
//        {
//            uchar *patchPtr = example.data;
//            uchar *modelPtr = negExp.data;

//            for (int i = 0; i < STANDARD_PATCH_SIZE*STANDARD_PATCH_SIZE; i++)
//                modelPtr[negNum*STANDARD_PATCH_SIZE*STANDARD_PATCH_SIZE + i] = patchPtr[i];

//            negNum++;
//        }

//        //proxyV = &negativeExamples;
//        proxyN = &timeStampNegativeNext;
//        proxyT = &timeStampsNegative;
//    }

//    if ((int)proxyV->size() < MAX_EXAMPLES_IN_MODEL)
//    {
//        proxyV->push_back(example);
//        proxyT->push_back(*proxyN);
//    }
//    else
//    {
//        int index = rng.uniform((int)0, (int)proxyV->size());
//        (*proxyV)[index] = example;
//        (*proxyT)[index] = (*proxyN);
//    }

//    (*proxyN)++;
}
}
}
