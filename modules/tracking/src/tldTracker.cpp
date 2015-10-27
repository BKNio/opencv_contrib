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

//    FernClassifier::compareFerns("/tmp/2000.xml", "/tmp/1000.xml");
//    exit(0);


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
    Mat_<uchar> actImage;

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
    {
        trackerConfidence = cascadeClassifier->nnClassifier->calcConfidenceTracker(actImage(objectFromTracker));
        //std::cout << "main tracker conf " << trackerConfidence << " tracker object " << objectFromTracker << std::endl;
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
        //std::cout << "reset main tracker " << integratorResult.objectToResetTracker << std::endl;
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

    return true;
}

const Integrator::IntegratorResult Integrator::getObjectToTrainFrom(const Mat_<uchar> &frame,
                                                       const std::pair<Rect2d, double> &objectFromTracker,
                                                       const std::vector<std::pair<Rect, double> > &objectsFromDetector)
{
    /*bool lockTrain = false;

    std::cout << "__________________________" << std::endl;

    cvtColor(frame, copy, CV_GRAY2BGR);

    std::for_each(candidates.begin(), candidates.end(), std::bind2nd(std::ptr_fun(updateCandidates), frame));
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(), std::ptr_fun(selectCandidateForRemove)), candidates.end());
    std::for_each(candidates.begin(), candidates.end(), std::bind2nd(std::ptr_fun(incrementHints), objectsFromDetector));

    std::vector< Ptr<Candidate> > readyCandidates;

    for(std::vector< Ptr<Candidate> >::iterator it = candidates.begin(); it != candidates.end(); ++it)
        if(it->operator ->()->hints >= 3)
            readyCandidates.push_back(*it);

    std::sort(readyCandidates.begin(), readyCandidates.end(), sortPredicateConf);

    bool tIsIntegratorPresent = false;

    if(!readyCandidates.empty())
    {
        tIsIntegratorPresent = true;
        std::cout << "maxPos conf " << readyCandidates.front()->confidence << " hints " << readyCandidates.front()->hints << std::endl;
    }

    const bool isIntegratorPresent = tIsIntegratorPresent;
    const Rect2d integratorObject = isIntegratorPresent ? readyCandidates.front()->prevRect : Rect2d();
    const double integratorConfidence = isIntegratorPresent ? readyCandidates.front()->confidence : -1.;

    const bool isTrackerPresent = objectFromTracker.first.area() > 0;
    const Rect2d &trackerObject = objectFromTracker.first;
    const double trackerConfidence = objectFromTracker.second;

    Rect grouped;

    if(!objectsFromDetector.empty())
    {
        std::vector< std::pair<Rect, double> >::const_iterator maxElemnt = std::max_element(objectsFromDetector.begin(),
                         objectsFromDetector.end(), sortDetections);

       CV_Assert(maxElemnt != objectsFromDetector.end());

       grouped = maxElemnt->first;

       std::pair<Mat, Mat> model = nnClassifier->getModelWDecisionMarks(frame(grouped), maxElemnt->second);

       imshow("positive model", model.first);
       imshow("negative model", model.second);
    }

    const Rect &detectorObject = grouped;
    const bool isDetectorPresent = detectorObject.area() > 0;
    const double detectorConfidence = isDetectorPresent ? nnClassifier->calcConfidence(frame(detectorObject)) : -1.;

    Rect mergedTrackerObject;
    double mergedTrackerConfidence = -1.;
    bool isMergedTrackerPresent = true;

    Rect2d objectToResetMainTracker;

    if(isIntegratorPresent && isTrackerPresent)
    {
        if(integratorConfidence - trackerConfidence > 0.05 * trackerConfidence)
        {
            std::cout << "integrator's tracker better ";
            objectToResetMainTracker = integratorObject;
            mergedTrackerObject = integratorObject;
            mergedTrackerConfidence = integratorConfidence;

            CV_Assert(!readyCandidates.empty());

            std::vector< Ptr<Candidate> >::iterator removePos = std::remove(candidates.begin(), candidates.end(), readyCandidates.front());

            CV_Assert(removePos != candidates.end());

            candidates.erase(removePos, candidates.end());

            rectangle(copy, objectToResetMainTracker, Scalar::all(255),2);
        }
        else
        {
            std::cout << "main tracker better ";

            mergedTrackerObject = trackerObject;
            mergedTrackerConfidence = trackerConfidence;
        }

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
            candidates.push_back(makePtr<Candidate>(frame, detectorObject, detectorConfidence));
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
        if(overlap(mergedTrackerObject, detectorObject) > 0.85 && mergedTrackerConfidence > 0.5 && detectorConfidence > 0.5)
        {
            if(candidates.empty())
            {
                objectToTrain = mergedTrackerObject;
            }
            else
            {
                std::cout << "candidates not empty cannot train" << std::endl;
            }

            objectToPresent = averageRects(mergedTrackerObject, detectorObject);
        }
        else
        {
            objectToPresent = mergedTrackerConfidence >= detectorConfidence ? mergedTrackerObject : detectorObject;
        }
    }
    else if(isDetectorPresent && detectorConfidence > 0.5)
    {
        objectToPresent = detectorObject;
    }
    else if(isMergedTrackerPresent && mergedTrackerConfidence > 0.5)
    {
        objectToPresent = mergedTrackerObject;
    }

    imshow("integrator", copy);*/


    std::cout << "_________________________" << std::endl;

    Rect objectToTrain, objectToPresent, detectorObject;
    Rect2d objectToResetMainTracker;

    if(!objectsFromDetector.empty())
    {
        std::vector< std::pair<Rect, double> >::const_iterator maxElemnt = std::max_element(objectsFromDetector.begin(),
                         objectsFromDetector.end(), sortDetections);


        detectorObject = maxElemnt->first;

//        std::pair<Mat, Mat> model = nnClassifier->getModelWDecisionMarks(frame(detectorObject), maxElemnt->second);

//        imshow("positive model", model.first);
//        imshow("negative model", model.second);

        std::cout << "object from detector conf " << maxElemnt->second << std::endl;

        if(objectFromTracker.first.area() > 0)
        {
            //std::vector<double> overlaps;
            //std::transform(objectsFromDetector.begin(), objectsFromDetector.end(), std::back_inserter(overlaps),
                           //std::bind2nd(std::ptr_fun(overlapIncPredicate), objectFromTracker.first));

            //std::vector<double>::iterator maxOverlap = std::max_element(overlaps.begin(), overlaps.end());

            //const std::pair<Rect, double> bestOverlap = objectsFromDetector[std::distance(overlaps.begin(), maxOverlap)];

            //CV_Assert(overlap(objectFromTracker.first, bestOverlap.first) == *maxOverlap);
            //CV_Assert(bestOverlap.second > 0.5);

            std::cout << "object from tracker conf " << objectFromTracker.second << std::endl;

            if(maxElemnt->second > objectFromTracker.second && maxElemnt->second > 0.51)
            {
                std::cout << "reseting main tracker, no train " << std::endl;
                objectToResetMainTracker = maxElemnt->first;
                preparing = 0;
            }
            else if(overlap(objectFromTracker.first, detectorObject) > 0.85 && objectFromTracker.second > 0.51 && maxElemnt->second > 0.51)
            {
                if(preparing >= 0)
                {
                    std::cout << "good overlap and conf training and ready to train" << std::endl;
                    objectToTrain = objectFromTracker.first;
                }
                else
                {
                    std::cout << "not ready to train " << preparing << std::endl;
                }
                ++preparing;
            }

            objectToPresent = objectFromTracker.second > maxElemnt->second ? Rect(objectFromTracker.first) : maxElemnt->first;
        }
        else if(maxElemnt->second > 0.51)
        {
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

void Integrator::updateCandidates(Ptr<Candidate> candidate, const Mat_<uchar> &frame)
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
            candidate->confidence = nnClassifier->calcConfidenceTracker(frame(candidate->prevRect));
            candidate->hints -= .1;
            rectangle(copy, candidate->prevRect, Scalar(255, 0, 0), 1);
        }
        else
        {
            candidate->confidence = -.1;
            rectangle(copy, trakingObject, Scalar(0, 0, 255), 1);
        }
    }


}

void Integrator::incrementHints(Ptr<Integrator::Candidate> candidate, const std::vector<std::pair<Rect, double> > &objectsFromDetector)
{
    std::vector<std::pair<Rect, double> >::const_iterator pos =
            std::find_if(objectsFromDetector.begin(), objectsFromDetector.end(), std::bind2nd(std::ptr_fun(overlapIncPredicate), candidate->prevRect));

    if(pos != objectsFromDetector.end())
        candidate->hints += 1.;

    std::stringstream ss;
    ss << "updated candidate conf " << candidate->confidence << " hints " << candidate->hints;
    std::cout << ss.str() << std::endl;
}

bool Integrator::sortPredicateConf(const Ptr<Candidate> &candidate1, const Ptr<Candidate> &candidate2)
{
    return candidate1->confidence > candidate2->confidence;
}

bool Integrator::sortDetections(const std::pair<Rect, double> &candidate1, const std::pair<Rect, double> &candidate2)
{
    return candidate1.second < candidate2.second;
}

bool Integrator::sortPredicateHints(const Ptr<Candidate> &candidate)
{
    return candidate->hints > 3.;
}

bool Integrator::overlapPredicate(const Ptr<Candidate> candidate, const Rect &bb)
{
    return overlap(candidate->prevRect, bb) > 0.9;
}

double Integrator::overlapIncPredicate(const std::pair<Rect, double> candidate, const Rect2d bb)
{
    return overlap(candidate.first, bb);
}

bool Integrator::selectCandidateForRemove(const Ptr<Candidate> candidate)
{
    bool ret = candidate->confidence < 0.5 || candidate->hints < -1.;


    ret |= candidate->hints > 20;

    if(ret)
    {
        std::stringstream ss;
        ss << "remove candidate conf " << candidate->confidence << " " << candidate->hints << std::endl;
        std::cout << ss.str() << std::endl;
    }


    return ret;
}

Rect Integrator::averageRects(const Rect &item1, const Rect &item2)
{
    const Point tl((item1.x + item2.x) / 2, (item1.y + item2.y) / 2);
    const Size size((item1.width + item2.width) / 2, (item1.height + item2.height) / 2);
    return Rect(tl, size);
}

Integrator::Candidate::Candidate(const Mat_<uchar> &frame, Rect bb, double _confidence) : confidence(_confidence)
{
    medianFlow = TrackerMedianFlow::createTracker();

    if(!medianFlow->init(frame, bb))
        confidence = -1;

    prevRect = bb;
    hints = 0;
}

}

}
