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

Rect Integrator::roi = Rect();
Ptr<NNClassifier> Integrator::nnClassifier = Ptr<NNClassifier>();
Mat Integrator::copy = Mat();


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

    integrator = makePtr<Integrator>(cascadeClassifier->nnClassifier, roi);

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


    Rect trackerReset;
    std::pair<Rect, Rect> integratorResults = integrator->getObjectToTrainFrom
                (actImage,
                 std::make_pair(objectFromTracker, trackerConfidence),
                 std::make_pair(objectFromDetector, detectorConfidence), trackerReset
                 );

    if(trackerReset.area() > 0)
    {
        //CV_Assert(trackerReset.area() > 0);

        std::cout << ">>>>restart tracker<<<<" << std::endl;

        medianFlow = TrackerMedianFlow::createTracker();
        isTrackerOK = medianFlow->init(actImage, trackerReset);
    }

    if(integratorResults.first.area())
    {
        cascadeClassifier->startPExpert(actImage, integratorResults.first);
        cascadeClassifier->startNExpert(actImage, integratorResults.first, detections);

        rectangle(debugOutput, integratorResults.first, Scalar(169, 0, 255), 2);
    }

    boundingBox = integratorResults.second;

    static int counter = 0;
    imshow("debugoutput", debugOutput);

    std::stringstream ss; ss << "/tmp/debug/" << counter++ << ".png";
    imwrite(ss.str(), debugOutput);

    return true;
}

std::pair<Rect, Rect> Integrator::getObjectToTrainFrom(const Mat_<uchar> &frame,
                                                       const std::pair<Rect, double> &objectFromTracker,
                                                       const std::pair<Rect, double> &objectFromDetector,
                                                       Rect &trackerReset)
{
    trackerReset = Rect();

    cvtColor(frame, copy, CV_GRAY2BGR);

    const bool isTrackerPresent = objectFromTracker.first.area() > 0;
    const bool isDetectorPresent = objectFromDetector.first.area() > 0;

    const Rect &trackerObject = objectFromTracker.first;
    const Rect &detectorObject = objectFromDetector.first;

    const double trackerConfidence = objectFromTracker.second;
    const double detectorConfidence = objectFromDetector.second;

    std::for_each(candidates.begin(), candidates.end(), std::bind2nd(std::ptr_fun(updateCandidates), frame));
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(), std::ptr_fun(selectCandidateForRemove)), candidates.end());

    std::sort(candidates.begin(), candidates.end(), sortPredicate);

    if(candidates.size() > 5)
    {
       candidates.erase(candidates.begin() + 5, candidates.end());
       CV_Assert(candidates.size() == 5);
    }

    const bool isIntegratorPresent = !candidates.empty();
    const Rect integratorObject = isIntegratorPresent ? candidates.front().prevRect : Rect2d();
    const double integratorConfidence = isIntegratorPresent ? candidates.front().confidence.at<double>() : -1.;

    /*if(isDetectorPresent)
    {
        std::vector<Candidate>::iterator position = std::find_if(candidates.begin(),
                                                                 candidates.end(),
                                                                 std::bind2nd(std::ptr_fun(overlapPredicate), detectorObject));

        if(position == candidates.end())
        {
            candidates.push_back(Candidate(frame, detectorObject));
            //std::cout << "integrator: add new object to track" << std::endl;
        }
        else
        {
            //std::cout << "integrator: object is allready tracking" << std::endl;
        }
    }*/

    Rect mergedTrackerObject;
    double mergedTrackerConfidence = -1.;
    bool isMergedTrackerPresent = true;

    if(isIntegratorPresent && isTrackerPresent)
    {
        if(integratorConfidence > trackerConfidence)
        {
            std::cout << "integrator's tracker better ";

            trackerReset = integratorObject;
            mergedTrackerObject = integratorObject;
            mergedTrackerConfidence = integratorConfidence;


            std::swap(candidates.front(), candidates.back());
            candidates.erase(candidates.end() - 1, candidates.end());

            rectangle(copy, trackerReset, Scalar(0,0,255));

        }
        else
        {
            std::cout << "main tracker better ";

            mergedTrackerObject = trackerObject;
            mergedTrackerConfidence = trackerConfidence;
        }

        std::cout << " integrator's tracker: " << integratorConfidence << " main tracker " << trackerConfidence << std::endl;
    }
    else if(isTrackerPresent)
    {
        mergedTrackerObject = trackerObject;
        mergedTrackerConfidence = trackerConfidence;
    }
    else if(isIntegratorPresent)
    {
        mergedTrackerObject = integratorObject;
        mergedTrackerConfidence = integratorConfidence;
        trackerReset = integratorObject;
    }
    else
        isMergedTrackerPresent = false;

    Rect objectToTrain, objectToOutput;

    std::string info;
    if(detectorConfidence > mergedTrackerConfidence)
    {
        info = "detector better";

        std::vector<Candidate>::iterator position = std::find_if(candidates.begin(),
                                                                 candidates.end(),
                                                                 std::bind2nd(std::ptr_fun(overlapPredicate), mergedTrackerConfidence));

        if(position == candidates.end())
        {
            candidates.push_back(Candidate(frame, detectorObject));
            //std::cout << "integrator: add new object to track" << std::endl;
        }
        //else
        //{
            //std::cout << "integrator: object is allready tracking" << std::endl;
        //}

        //candidates.push_back(Candidate(frame, detectorObject));
    }
    else
    {
        info = "tracker  better";
    }

    std::cout << info << " detector conf " << detectorConfidence << " trackerConfidence " << mergedTrackerConfidence << std::endl;


    if(isDetectorPresent && isMergedTrackerPresent)
    {
        if(overlap(mergedTrackerObject, detectorObject) > 0.85 && trackerReset.area() == 0)
        {
            objectToTrain = mergedTrackerObject;
            objectToOutput = averageRects(mergedTrackerObject, detectorObject);
        }
        else
        {
            objectToOutput = mergedTrackerConfidence >= detectorConfidence ? mergedTrackerObject : detectorObject;
        }
    }
    else if(isDetectorPresent && detectorConfidence > 0.5)
    {
        objectToOutput = detectorObject;
    }
    else if(isMergedTrackerPresent && mergedTrackerConfidence > 0.5)
    {
        objectToOutput = mergedTrackerObject;
    }


    imshow("integrator", copy);
    waitKey(1);
    return std::make_pair(objectToTrain, objectToOutput);
}

void Integrator::updateCandidates(Candidate candidate, Mat_<uchar> &frame)
{
    Rect2d trakingObject;

    CV_Assert(candidate.confidence.at<double>(0) >= 0.);
    CV_Assert(roi.area() != 0);

    if(!candidate.medianFlow->update(frame, trakingObject))
        candidate.confidence.at<double>(0) = -1.;
    else
    {
        if(roi.contains(trakingObject.tl()) && roi.contains(trakingObject.br()))
        {
            candidate.prevRect = trakingObject;
            candidate.confidence.at<double>(0) = nnClassifier->calcConfidence(frame(candidate.prevRect));
            rectangle(copy, trakingObject, Scalar(255, 0, 0));
        }
        else
        {
            candidate.confidence.at<double>(0) = -1.;
            rectangle(copy, trakingObject, Scalar(0, 0, 255));
        }
    }


}

void Integrator::updateCandidatesConfidence(Integrator::Candidate candidate, Mat_<uchar> &frame)
{
    CV_Assert(0);
    CV_Assert(candidate.confidence.at<double>(0) != -1.);
    CV_Assert(!nnClassifier.empty());
    candidate.confidence.at<double>() = nnClassifier->calcConfidence(frame(candidate.prevRect));
}

bool Integrator::sortPredicate(const Integrator::Candidate &candidate1, const Integrator::Candidate &candidate2)
{
    return candidate1.confidence.at<double>(0) > candidate2.confidence.at<double>(0);
}

bool Integrator::overlapPredicate(Candidate candidate, const Rect &bb)
{
    return overlap(candidate.prevRect, bb) > 0.85;
}

bool Integrator::selectCandidateForRemove(const Candidate candidate)
{
    return candidate.confidence.at<double>() < 0.5;
}

Rect Integrator::averageRects(const Rect &item1, const Rect &item2)
{
    const Point tl((item1.x + item2.x) / 2, (item1.y + item2.y) / 2);
    const Size size((item1.width + item2.width) / 2, (item1.height + item2.height) / 2);
    return Rect(tl, size);
}

Integrator::Candidate::Candidate(const Mat_<uchar> &frame, Rect bb) : confidence(1, 1, 0.), hints(1, 1, 0)
{
    medianFlow = TrackerMedianFlow::createTracker();

    if(!medianFlow->init(frame, bb))
        confidence.at<int>() = -1;

    prevRect = bb;
}

}

}
