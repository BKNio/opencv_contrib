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
    cascadeClassifier =  Ptr<CascadeClassifier>(
                new CascadeClassifier(
                parameters.preFernMeasurements, parameters.preFerns, parameters.preFernPatchSize,
                parameters.numberOfMeasurements, parameters.numberOfFerns, parameters.fernPatchSize,
                parameters.numberOfExamples, parameters.examplePatchSize,
                parameters.numberOfPositiveExamples, parameters.numberOfWarpedPositiveExamples,
                parameters.groupRectanglesTheta
                )
                );

    integrator = makePtr<Integrator>(4);
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

    return true;

}

#define DEBUG_OUTPUT

bool TrackerTLDImpl::updateImpl(const Mat& image, Rect2d& boundingBox)
{
    Mat actImage;

    if(image.type() == CV_8UC3)
        cvtColor(image, actImage, COLOR_BGR2GRAY);
    else if(image.type() == CV_8U)
        image.copyTo(actImage);
    else
        CV_Error(Error::StsBadArg, "wrong input image type: should be CV_8UC3 or CV_8U");

    std::vector<Rect> detections = cascadeClassifier->detect(actImage);
    Rect objectFromDetector;
    double detectorConfidence = -1.;
    if(!detections.empty())
    {
        objectFromDetector = detections.front();
        detectorConfidence = cascadeClassifier->nnClassifier->calcConfidence(image(objectFromDetector));
    }

    Rect2d objectFromTracker;
    double trackerConfidence = -1.;
    if(isTrackerOK)
        isTrackerOK = medianFlow->update(actImage, objectFromTracker);

    if(isTrackerOK && roi.contains(objectFromTracker.tl()) && roi.contains(objectFromTracker.br()))
        trackerConfidence = cascadeClassifier->nnClassifier->calcConfidence(image(objectFromTracker));

#ifdef DEBUG_OUTPUT
    Mat debugOutput; cvtColor(actImage, debugOutput, COLOR_GRAY2BGR);
    if(!detections.empty())
        rectangle(debugOutput, detections.front(), Scalar(255,0,0), 2);
    if(isTrackerOK)
        rectangle(debugOutput, objectFromTracker, Scalar(0, 255, 0), 2);
#endif



    std::pair<Rect, Rect> integratorResults = integrator->getObjectToTrainFrom
                (actImage,
                 std::make_pair(objectFromTracker, trackerConfidence),
                 std::make_pair(objectFromDetector, detectorConfidence)
                 );

    if(integratorResults.first.area())
    {
        cascadeClassifier->startPExpert(actImage, integratorResults.first);
        cascadeClassifier->startNExpert(actImage, integratorResults.first, detections);

        if(!isTrackerOK)
        {
            medianFlow = TrackerMedianFlow::createTracker();
            isTrackerOK = medianFlow->init(actImage, integratorResults.first);
        }

        rectangle(debugOutput, integratorResults.first, Scalar(169, 0, 255), 2);
    }

    boundingBox = integratorResults.second;

    static int counter = 0;
    imshow("debugoutput", debugOutput);

    std::stringstream ss; ss << "/tmp/debug/" << counter++ << ".png";
    imwrite(ss.str(), debugOutput);

    return true;
}

std::pair<Rect, Rect> Integrator::getObjectToTrainFrom(const Mat_<uchar> &frame, const std::pair<Rect, double> &objectFromTracker, const std::pair<Rect, double> &objectFromDetector)
{
    const bool isTrackerPresent = objectFromTracker.first.area() > 0;
    const bool isDetectorPresent = objectFromDetector.first.area() > 0;

    const Rect &trackerObject = objectFromTracker.first;
    const Rect &detectorObject = objectFromDetector.first;

    const double trackerConfidence = objectFromTracker.second;
    const double detectorConfidence = objectFromDetector.second;

    std::string info;
    if(detectorConfidence > trackerConfidence)
        info = "detecor better";
    else
        info = "tracker better";

    std::cout << info << " detector conf " << detectorConfidence << " trackerConfidence " << trackerConfidence << std::endl;

    Rect objectToTrain, objectToOutput;

    if(isDetectorPresent && isTrackerPresent)
    {
        if(overlap(trackerObject, detectorObject) > 0.85)
        {
            objectToTrain = trackerObject;
            objectToOutput = averageRects(trackerObject, detectorObject);
        }
        else
        {
            objectToOutput = trackerConfidence >= detectorConfidence ? trackerObject : detectorObject;
        }
    }
    else if(isDetectorPresent && detectorConfidence > 0.5)
    {
        objectToOutput = detectorObject;
    }
    else if(isTrackerPresent && detectorConfidence > 0.5)
    {
        objectToOutput = trackerObject;
    }

    if(!isTrackerPresent)
    {
        //std::cout << "reinit tracker..." << std::endl;

        std::for_each(candidates.begin(), candidates.end(), std::bind2nd(std::ptr_fun(updateCandidate), frame));

        candidates.erase(std::remove_if(candidates.begin(), candidates.end(), std::ptr_fun(selectCandidateForRemove)), candidates.end());

        if(isDetectorPresent)
        {
            const std::vector<Candidate>::iterator position =
                    std::find_if(candidates.begin(), candidates.end(), std::bind2nd(std::ptr_fun(selectCandidateForInc), detectorObject));

            if(position == candidates.end())
            {
                candidates.push_back(Candidate(frame, detectorObject));
            }
            else
            {
                position->rects.push_back(detectorObject);
                if(++position->hints.at<int>() > numberOfConfirmations)
                {
                    Rect ret = myGroupRectangles(position->rects);
                    candidates.clear();

                    objectToOutput = ret;
                    objectToTrain = ret;
                }
            }

        }
    }

    return std::make_pair(objectToTrain, objectToOutput);
}

void Integrator::updateCandidate(Candidate candidate, Mat_<uchar> &frame)
{
    Rect2d trakingObject;

    CV_Assert(candidate.hints.at<int>() >= 0);

    if(!candidate.medianFlow->update(frame, trakingObject))
        candidate.hints.at<int>() = -1;
    else
        candidate.prevRect = trakingObject;

}

bool Integrator::selectCandidateForInc(Candidate candidate, const Rect &bb)
{
    return overlap(candidate.prevRect, bb) > 0.5;
}

bool Integrator::selectCandidateForRemove(const Candidate candidate)
{
    return candidate.hints.at<int>() == -1;
}

Rect Integrator::averageRects(const Rect &item1, const Rect &item2)
{
    const Point tl((item1.x + item2.x) / 2, (item1.y + item2.y) / 2);
    const Size size((item1.width + item2.width) / 2, (item1.height + item2.height) / 2);
    return Rect(tl, size);
}

Integrator::Candidate::Candidate(const Mat_<uchar> &frame, Rect bb) : hints(1, 1, 0)
{
    medianFlow = TrackerMedianFlow::createTracker();

    if(!medianFlow->init(frame, bb))
        hints.at<int>() = -1;

    prevRect = bb;
}

}

}
