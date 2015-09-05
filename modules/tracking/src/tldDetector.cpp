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

tldCascadeClassifier::tldCascadeClassifier(const Mat_<uchar> &originalImage, const Rect &bb, int maxNumberOfExamples, int numberOfFerns, int numberOfMeasurements):
    originalBBSize(bb.size()), frameSize(originalImage.size()), scaleStep(1.2f)
{

    /*if(bb.width < minimalBBSize.width || bb.height < minimalBBSize.height)
        CV_Error(Error::StsBadArg, "Initial bounding box is too small");*/

    varianceClassifier = makePtr<tldVarianceClassifier>(originalImage, bb);
    fernClassifier = makePtr<tldFernClassifier>(numberOfMeasurements, numberOfFerns);
    nnClassifier = makePtr<tldNNClassifier>(maxNumberOfExamples, standardPath);
}

void tldCascadeClassifier::isObjects(const std::vector<Hypothesis> &hypothesis, const Mat_<uchar> &scaledImage, std::vector<bool> &answers) const
{
    varianceClassifier->isObjects(hypothesis, scaledImage, answers);
    fernClassifier->isObjects(hypothesis, scaledImage, answers);
    nnClassifier->isObjects(hypothesis, scaledImage, answers);
}

void tldCascadeClassifier::addPositiveExample(const Mat_<uchar> &example)
{
    fernClassifier->integratePositiveExample(example);
    nnClassifier->integratePositiveExample(example);
}

void tldCascadeClassifier::addNegativeExample(const Mat_<uchar> &example)
{
    fernClassifier->integrateNegativeExample(example);
    nnClassifier->integrateNegativeExample(example);
}

std::vector<Hypothesis> tldCascadeClassifier::generateHypothesis() const
{
    std::vector<Hypothesis> hypothesis;

    const double scaleX = frameSize.width / originalBBSize.width;
    const double scaleY = frameSize.height / originalBBSize.height;

    const double scale = std::min(scaleX, scaleY);

    const double power =log(scale) / log(scaleStep);
    double correctedScale = pow(scaleStep, power);

    CV_Assert(int(originalBBSize.width * correctedScale) <= frameSize.width && int(originalBBSize.height * correctedScale) <= frameSize.height);

    for(;;)
    {
        Size correntBBSize(originalBBSize.width * correctedScale, originalBBSize.height * correctedScale);

        if(correntBBSize.width < minimalBBSize.width || correntBBSize.height < minimalBBSize.height)
            break;
        addScanGrid(correntBBSize, frameSize, hypothesis);

        /*{
            for(std::vector<Hypothesis>::const_iterator it = hypothesis.begin(); it != hypothesis.end(); ++it)
            {
                Mat copy; image.copyTo(copy);
                rectangle(copy, it->bb, Scalar::all(255));
                imshow("copy", copy);
                waitKey(1);
            }
            hypothesis.clear();
        }*/

        correctedScale /= scaleStep;
    }
    return hypothesis;
}

void tldCascadeClassifier::addScanGrid(const Size bbSize, const Size imageSize , std::vector<Hypothesis> &hypothesis)
{
    CV_Assert(bbSize.width >= 20 && bbSize.height >= 20);

    const int dx = bbSize.width / 10;
    const int dy = bbSize.height / 10;

    for(int currentX = 0; currentX < imageSize.width - bbSize.width - dx; currentX += dx)
        for(int currentY = 0; currentY < imageSize.height - bbSize.height - dy; currentY += dy)
            hypothesis.push_back(Hypothesis(currentX, currentY, bbSize));
}

//void tldCascadeClassifier::detect(const Mat_<uchar>& img, std::vector<Response>& responses)
//{
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
//}

}
}
