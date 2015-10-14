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

    const std::vector< std::pair<Rect, double> > &detections = cascadeClassifier->detect(actImage);

    Rect2d objectFromTracker;
    double trackerConfidence = -1.;
    if(isTrackerOK)
        isTrackerOK = medianFlow->update(actImage, objectFromTracker);

    if(isTrackerOK && roi.contains(objectFromTracker.tl()) && roi.contains(objectFromTracker.br()))
        trackerConfidence = cascadeClassifier->nnClassifier->calcConfidence(actImage(objectFromTracker));

#ifdef DEBUG_OUTPUT
    Mat debugOutput; cvtColor(actImage, debugOutput, COLOR_GRAY2BGR);

    //for(std::vector< std::pair<Rect, double> >::const_iterator it = detections.begin(); it != detections.end(); ++it)
    //    rectangle(debugOutput, it->first, Scalar(255,0,0), 1);

    if(isTrackerOK)
        rectangle(debugOutput, objectFromTracker, Scalar(0, 255, 0), 1);
#endif


    const Integrator::IntegratorResult integratorResult = integrator->getObjectToTrainFrom
                (actImage,
                 std::make_pair(objectFromTracker, trackerConfidence),
                 detections);

    if(integratorResult.objectToResetTracker.area() > 0)
    {
        std::cout << "restarting tracker..." << std::endl;

        medianFlow = TrackerMedianFlow::createTracker();
        isTrackerOK = medianFlow->init(actImage, integratorResult.objectToResetTracker);
        rectangle(debugOutput, integratorResult.objectToResetTracker, Scalar(0,255, 255));
    }

    if(integratorResult.objectToTrain.area())
    {
        cascadeClassifier->startPExpert(actImage, integratorResult.objectToTrain);
        cascadeClassifier->startNExpert(actImage, integratorResult.objectToTrain);

        rectangle(debugOutput, integratorResult.objectToTrain, Scalar(169, 0, 255), 2);
    }

    boundingBox = integratorResult.objectToOutput;

    if(boundingBox.area() > 0)
        rectangle(debugOutput, boundingBox, Scalar(255, 0, 0), 1);

    static int counter = 0;
    imshow("debugoutput", debugOutput);

    std::stringstream ss; ss << "/tmp/debug/" << counter++ << ".png";
    imwrite(ss.str(), debugOutput);

    return true;
}

const Integrator::IntegratorResult Integrator::getObjectToTrainFrom(const Mat_<uchar> &frame,
                                                       const std::pair<Rect, double> &objectFromTracker,
                                                       const std::vector<std::pair<Rect, double> > &objectsFromDetector)
{
    std::cout << "---------------------------------------------" << std::endl;

    cvtColor(frame, copy, CV_GRAY2BGR);


    std::for_each(candidates.begin(), candidates.end(), std::bind2nd(std::ptr_fun(updateCandidates), frame));
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(), std::ptr_fun(selectCandidateForRemove)), candidates.end());

    std::sort(candidates.begin(), candidates.end(), sortPredicate);

    if(candidates.size() > maxCandidatesSize)
    {
       candidates.erase(candidates.begin() + maxCandidatesSize, candidates.end());
       CV_Assert(candidates.size() == maxCandidatesSize);
    }

    const bool isIntegratorPresent = !candidates.empty();
    const Rect2d integratorObject = isIntegratorPresent ? candidates.front()->prevRect : Rect2d();
    const double integratorConfidence = isIntegratorPresent ? candidates.front()->confidence : -1.;

    const bool isTrackerPresent = objectFromTracker.first.area() > 0;
    const Rect2d &trackerObject = objectFromTracker.first;
    const double trackerConfidence = objectFromTracker.second;

    Rect grouped;

    if(!objectsFromDetector.empty())
    {
        Mat toGroupDebug; cvtColor(frame, toGroupDebug, CV_GRAY2BGR);

        const double value = objectsFromDetector.front().second;
        std::vector<Rect> toGroup;
        for(std::vector< std::pair<Rect, double> >::const_iterator it = objectsFromDetector.begin(); it != objectsFromDetector.end(); ++it)
        {
            if(std::fabs(it->second - value) <= value * 0.07)
            {
                toGroup.push_back(it->first);
                rectangle(toGroupDebug, it->first, Scalar(0, 255, 0));
            }
        }

        grouped = myGroupRectangles(toGroup);

        rectangle(toGroupDebug, grouped, Scalar(0, 0, 255));

        imshow("groupedDebug", toGroupDebug);
    }

    const Rect &detectorObject = grouped;
    const bool isDetectorPresent = detectorObject.area() > 0;
    const double detectorConfidence = isDetectorPresent ?nnClassifier->calcConfidence(frame(detectorObject)) : -1.;


//    const Rect &detectorObject = objectsFromDetector.front().first;
//    const bool isDetectorPresent = detectorObject.area() > 0;
//    const double detectorConfidence = objectsFromDetector.front().second;

    Rect mergedTrackerObject;
    double mergedTrackerConfidence = -1.;
    bool isMergedTrackerPresent = true;

    Rect objectToResetMainTracker;

    if(isIntegratorPresent && isTrackerPresent)
    {
        if(integratorConfidence > trackerConfidence)
        {
            std::cout << "integrator's tracker better ";

            objectToResetMainTracker = integratorObject;
            mergedTrackerObject = integratorObject;
            mergedTrackerConfidence = integratorConfidence;

            std::swap(candidates.front(), candidates.back());
            candidates.erase(candidates.end() - 1, candidates.end());

            rectangle(copy, objectToResetMainTracker, Scalar(0,255,255));
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
        objectToResetMainTracker = integratorObject;

        std::cout << "using integrators tracker" << std::endl;
    }
    else
        isMergedTrackerPresent = false;


    Rect2d tMaxOverlapObject;
    double tMaxOverlap;
    double tMaxOverlapConf;

    if(isMergedTrackerPresent)
    {
        for(std::vector< std::pair<Rect, double> >::const_iterator it = objectsFromDetector.begin(); it != objectsFromDetector.end(); ++it)
        {
            const double currOverlap = overlap(mergedTrackerObject, it->first);
            if(currOverlap > tMaxOverlap)
            {
                tMaxOverlap =currOverlap;
                tMaxOverlapObject = it->first;
                tMaxOverlapConf = it->second;
            }
        }
    }

//    const bool isMaxOverlapObject = tMaxOverlapObject.area() > 0 && tMaxOverlap > 0.85;
//    const Rect2d maxOverlapObject = isMaxOverlapObject ? tMaxOverlapObject : Rect2d();
//    const double maxOverlapObjectConfidence = isMaxOverlapObject ? tMaxOverlapConf : -1.;

    Rect objectToTrain, objectToPresent;

    std::string info;
    if(detectorConfidence > mergedTrackerConfidence)
    {
        info = "detector better";

        std::vector< Ptr<Candidate> >::iterator position = std::find_if(candidates.begin(),
                                                                 candidates.end(),
                                                                 std::bind2nd(std::ptr_fun(overlapPredicate), detectorObject));

        CV_Assert(detectorConfidence == nnClassifier->calcConfidence(frame(detectorObject)));

        if(position == candidates.end())
        {
            candidates.push_back(makePtr<Candidate>(frame, detectorObject));
            std::cout << "added new integrator's tracker" << std::endl;
            rectangle(copy, detectorObject, Scalar(0,255,0));
        }
        else
        {
            std::cout << "exist good overlap do not add new one" << std::endl;
            rectangle(copy, detectorObject, Scalar(0,165,255));
            rectangle(copy, (*position)->prevRect, Scalar(0,165,255));
        }

    }
    else
        info = "tracker  better";

    std::cout << info << " detector conf " << detectorConfidence << " trackerConfidence " << mergedTrackerConfidence << std::endl;

    if(isDetectorPresent && isMergedTrackerPresent)
    {
        if(overlap(mergedTrackerObject, detectorObject) > 0.85 && objectToResetMainTracker.area() == 0 && mergedTrackerConfidence > 0.51 && detectorConfidence > 0.51)
        {
            objectToTrain = mergedTrackerObject;
            std::cout << "object to train getObjectsToTrain " << mergedTrackerObject << std::endl;
            objectToPresent = averageRects(mergedTrackerObject, detectorObject);
        }
        else
        {
            objectToPresent = mergedTrackerConfidence >= detectorConfidence ? mergedTrackerObject : detectorObject;
        }
    }
    else if(isDetectorPresent && detectorConfidence > 0.51)
    {
        objectToPresent = detectorObject;
    }
    else if(isMergedTrackerPresent && mergedTrackerConfidence > 0.51)
    {
        objectToPresent = mergedTrackerObject;
    }

    imshow("integrator", copy);
    return IntegratorResult(objectToTrain, objectToResetMainTracker,objectToPresent);
}

void Integrator::updateCandidates(Ptr<Candidate> candidate, Mat_<uchar> &frame)
{
    Rect2d trakingObject;

    CV_Assert(candidate->confidence >= 0.);
    CV_Assert(roi.area() != 0);

    if(!candidate->medianFlow->update(frame, trakingObject))
        candidate->confidence = -1.;
    else
    {
        if(roi.contains(trakingObject.tl()) && roi.contains(trakingObject.br()))
        {
            candidate->prevRect = trakingObject;
            candidate->confidence = nnClassifier->calcConfidence(frame(candidate->prevRect));
            rectangle(copy, candidate->prevRect, Scalar(255, 0, 0), 2);
        }
        else
        {
            candidate->confidence = -1.;
            rectangle(copy, trakingObject, Scalar(0, 0, 255), 2);
        }
    }


}

//void Integrator::updateCandidatesConfidence(Ptr<Candidate> candidate, Mat_<uchar> &frame)
//{
//    CV_Assert(0);
//    CV_Assert(candidate.confidence.at<double>(0) != -1.);
//    CV_Assert(!nnClassifier.empty());
//    candidate.confidence.at<double>() = nnClassifier->calcConfidence(frame(candidate.prevRect));
//}

bool Integrator::sortPredicate(const Ptr<Candidate> &candidate1, const Ptr<Candidate> &candidate2)
{
    return candidate1->confidence > candidate2->confidence;
}

bool Integrator::overlapPredicate(const Ptr<Candidate> candidate, const Rect &bb)
{
    return overlap(candidate->prevRect, bb) > 0.85;
}

bool Integrator::selectCandidateForRemove(const Ptr<Candidate> candidate)
{
    return candidate->confidence < 0.5;
}

Rect Integrator::averageRects(const Rect &item1, const Rect &item2)
{
    const Point tl((item1.x + item2.x) / 2, (item1.y + item2.y) / 2);
    const Size size((item1.width + item2.width) / 2, (item1.height + item2.height) / 2);
    return Rect(tl, size);
}

Integrator::Candidate::Candidate(const Mat_<uchar> &frame, Rect bb) : confidence(0.)
{
    medianFlow = TrackerMedianFlow::createTracker();

    if(!medianFlow->init(frame, bb))
        confidence = -1;

    prevRect = bb;
}

}

}
