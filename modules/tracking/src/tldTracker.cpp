/*///////////////////////////////////////////////////////////////////////////////////////
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

#include "tldTracker.hpp"
#include <algorithm>
#include <sys/time.h>

namespace cv
{

Ptr<TrackerTLD> TrackerTLD::createTracker(const TrackerTLD::Params &parameters)
{
    return makePtr<tld::TrackerTLDImpl>(parameters);
}

namespace tld
{


TrackerTLDImpl::TrackerTLDImpl(const TrackerTLD::Params &parameters)
{
    isInit = false;
    medianFlow = TrackerMedianFlow::createTracker();
    cascadeClassifier = makePtr<CascadeClassifier>(parameters.numberOfMeasurements, parameters.numberOfFerns, parameters.fernPatchSize,
                parameters.numberOfExamples, parameters.examplePatchSize,
                parameters.numberOfPositiveExamples, parameters.numberOfWarpedPositiveExamples);

}

void TrackerTLDImpl::read(const cv::FileNode& fn)
{
    params.read(fn);
}

void TrackerTLDImpl::write(cv::FileStorage& fs) const
{
    params.write(fs);
}

bool TrackerTLDImpl::initImpl(const Mat& image, const Rect2d& boundingBox)
{

    roi = cv::Rect2d(cv::Point2d(), image.size());
    model = makePtr<TrackerTLDModel>();
    Mat actImage;

    isTrackerOK = true;

    if(image.type() == CV_8UC3)
        cvtColor(image, actImage, COLOR_BGR2GRAY);
    else if(image.type() == CV_8U)
        image.copyTo(actImage);
    else
        CV_Error(Error::StsBadArg, "wrong input image type: should be CV_8UC3 or CV_8U");

    medianFlow->init(actImage, boundingBox);
    cascadeClassifier->init(actImage, boundingBox);

    integrator = makePtr<Integrator>();
    integrator->cascadeClassifier = cascadeClassifier;

    return true;

}

#define DEBUG_OUTPUT

bool TrackerTLDImpl::updateImpl(const Mat& image, Rect2d& boundingBox)
{
    Mat_<uchar> actImage;

    if(image.type() == CV_8UC3)
        cvtColor(image, actImage, COLOR_BGR2GRAY);
    else if(image.type() == CV_8U)
        image.copyTo(actImage);
    else
        CV_Error(Error::StsBadArg, "wrong input image type: should be CV_8UC3 or CV_8U");

    const std::vector< std::pair<Rect, double> > &detections = cascadeClassifier->detect(actImage);

    //timeval start, stop;
    //gettimeofday(&start, NULL);

    Rect2d objectFromTracker;
    double trackerConfidence = -1.;
    if(isTrackerOK)
        isTrackerOK = medianFlow->update(actImage, objectFromTracker);

    if(isTrackerOK && roi.contains(objectFromTracker.tl()) && roi.contains(objectFromTracker.br()))
    {
        trackerConfidence = cascadeClassifier->nnClassifier->calcConfidenceTracker(actImage(objectFromTracker));
        std::cout << "main tracker conf " << trackerConfidence << " tracker object " << objectFromTracker << std::endl;
    }

#ifdef DEBUG_OUTPUT
    Mat debugOutput; cvtColor(actImage, debugOutput, COLOR_GRAY2BGR);
#endif


    const Integrator::IntegratorResult integratorResult = integrator->getObjectToTrainFrom
                (actImage,
                 std::make_pair(objectFromTracker, trackerConfidence),
                 detections);

    if(integratorResult.objectToResetTracker.area() > 0)
    {
        std::cout << "reset main tracker " << integratorResult.objectToResetTracker << std::endl;
        medianFlow = TrackerMedianFlow::createTracker();
        isTrackerOK = medianFlow->init(actImage, integratorResult.objectToResetTracker);

    }

    if(integratorResult.objectToTrain.area())
    {
        cascadeClassifier->startPExpert(actImage, integratorResult.objectToTrain);
        cascadeClassifier->startNExpert(actImage, integratorResult.objectToTrain);

        rectangle(debugOutput, integratorResult.objectToTrain, Scalar(169, 0, 255), 2);
    }

    if(integratorResult.objectToResetTracker.area() > 0)
    {
        rectangle(debugOutput, integratorResult.objectToResetTracker, Scalar(0, 255, 255), 2);
    } else if(isTrackerOK)
    {
        rectangle(debugOutput, objectFromTracker, Scalar(0,255,0), 1);
    }


    boundingBox = integratorResult.objectToOutput;

    if(integratorResult.ObjectDetector.area() > 0)
        rectangle(debugOutput, integratorResult.ObjectDetector, Scalar(255, 0, 0), 1);

    static int counter = 0;
    imshow("debugoutput", debugOutput);

    std::stringstream ss; ss << "/tmp/debug/" << counter++ << ".png";
    imwrite(ss.str(), debugOutput);

    //gettimeofday(&stop, NULL);

    //std::cout << "merge " << std::fixed << stop.tv_sec - start.tv_sec + double(stop.tv_usec - start.tv_usec) / 1e6 << std::endl;

    return true;
}

const Integrator::IntegratorResult Integrator::getObjectToTrainFrom(const Mat_<uchar> &frame,
                                                       const std::pair<Rect2d, double> &objectFromTracker,
                                                       const std::vector<std::pair<Rect, double> > &objectsFromDetector)
{
    std::cout << "_________________________" << std::endl;

    Rect objectToTrain, objectToPresent, detectorObject;
    Rect2d objectToResetMainTracker;

    if(!objectsFromDetector.empty())
    {
        std::vector< std::pair<Rect, double> >::const_iterator maxElemnt = std::max_element(objectsFromDetector.begin(),
                         objectsFromDetector.end(), sortDetections);

        detectorObject = maxElemnt->first;

        /*-----------------------------------------------------------------*/

       /* int minArea = maxElemnt->first.area();
        Rect minAreaRect = maxElemnt->first;
        for(std::vector< std::pair<Rect, double> >::const_iterator it = objectsFromDetector.begin(); it != objectsFromDetector.end(); ++it)
        {
            int currOverlap = it->first.area();

            if(currOverlap < minArea)
            {
                minArea = currOverlap;
                minAreaRect = it->first;
            }
        }

        if(detectorObject == minAreaRect && objectsFromDetector.size() > 1)
         {
            std::cout << "***********************************************" << std::endl;

            for(std::vector< std::pair<Rect, double> >::const_iterator it = objectsFromDetector.begin(); it != objectsFromDetector.end(); ++it)
            {
                std::cout << it->first.area() << " ";
            }

            std::cout << std::endl << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>MIN<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
            waitKey(300);
        }*/


        /*-----------------------------------------------------------------*/

        std::cout << "object from detector conf " << maxElemnt->second << std::endl;

        if(objectFromTracker.first.area() > 0)
        {

            std::cout << "object from tracker conf " << objectFromTracker.second << std::endl;

            if(maxElemnt->second > objectFromTracker.second && maxElemnt->second > 0.51)
            {
                std::cout << "reseting main tracker, no train " << std::endl;
                objectToResetMainTracker = maxElemnt->first;
                preparing = 0;
            }
            else if(overlap(objectFromTracker.first, detectorObject) > 0.85 && objectFromTracker.second > 0.51 && maxElemnt->second > 0.51)
            {
                if(preparing >= 0 && objectFromTracker.first.area() > 900 )
                {
                    std::cout << "good overlap and conf training and ready to train" << std::endl;
                    objectToTrain = objectFromTracker.first;
                }
                else
                {
                    std::cout << "not ready to train or object is too small" << preparing << std::endl;
                }
                ++preparing;
            }

            objectToPresent = objectFromTracker.second > maxElemnt->second ? Rect(objectFromTracker.first) : maxElemnt->first;
        }
        else if(maxElemnt->second > 0.51)
        {
            //////
//            const double scaleFactorX = double(15) / detectorObject.width;
//            const double scaleFactorY = double(15) / detectorObject.height;

//            Mat_<uchar> resized;
//            resize(frame, resized, Size(), scaleFactorX, scaleFactorY);

//            const Point newPoint(cvRound(detectorObject.x * scaleFactorX), cvRound(detectorObject.y * scaleFactorY));
//            const Rect newRect(newPoint, Size(15,15));

//            std::pair<Mat, Mat> model = cascadeClassifier->nnClassifier->getModelWDecisionMarks(resized(newRect), maxElemnt->second);
//            imshow("positive", model.first);
//            imshow("negative", model.second);
            //////

            std::cout << "reset main tracker from detector" << std::endl;
            objectToResetMainTracker = maxElemnt->first;
            objectToPresent = maxElemnt->first;
            preparing = 0;
        }
    }
    else if(objectFromTracker.first.area() > 0 && objectFromTracker.second > 0.51)
    {
        std::cout << "object from tracker conf " << objectFromTracker.second << std::endl;
        objectToPresent = objectFromTracker.first;

        if(objectFromTracker.second > 0.52)
        {
            objectToTrain = objectFromTracker.first;
            std::cout << " >>> train from tracker <<<" << std::endl;
        }
    }


    return IntegratorResult(objectToTrain, objectToResetMainTracker,objectToPresent, detectorObject);
}

bool Integrator::sortDetections(const std::pair<Rect, double> &candidate1, const std::pair<Rect, double> &candidate2)
{
    return candidate1.second < candidate2.second;
}

}

}
