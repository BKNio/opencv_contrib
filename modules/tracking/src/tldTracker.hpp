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

#ifndef OPENCV_TLD_TRACKER
#define OPENCV_TLD_TRACKER

#include <algorithm>
#include <climits>
#include <map>
#include <numeric>


#include"precomp.hpp"
#include"opencv2/video/tracking.hpp"
#include"opencv2/imgproc.hpp"
#include"tldModel.hpp"

namespace cv
{

TrackerTLD::Params::Params()
{
    preFernMeasurements = 7;
    preFerns = 50;
    preFernPatchSize = Size(15, 15);

    numberOfMeasurements = 11;
    numberOfFerns = 100;
    fernPatchSize = Size(15, 15);

    numberOfExamples = 2000;
    examplePatchSize = Size(15, 15);

    numberOfInitPositiveExamples = 1;
    numberOfInitWarpedPositiveExamples = 200;

    numberOfPositiveExamples = 1;
    numberOfWarpedPositiveExamples = 100;

    groupRectanglesTheta = 0.15;
}

void TrackerTLD::Params::read(const cv::FileNode& fn)
{
    fn["preFernMeasurements"] >> preFernMeasurements;
    fn["preFerns"] >> preFerns;
    fn["preFernPatchSize"] >> preFernPatchSize;

    fn["numberOfMeasurements"] >> numberOfMeasurements;
    fn["numberOfFerns"] >> numberOfFerns;
    fn["fernPatchSize"] >> fernPatchSize;

    fn["numberOfExamples"] >> numberOfExamples;
    fn["examplePatchSize"] >> examplePatchSize;

    fn["numberOfInitPositiveExamples"] >> numberOfInitPositiveExamples;
    fn["numberOfInitWarpedPositiveExamples"] >> numberOfInitWarpedPositiveExamples;

    fn["numberOfPositiveExamples"] >> numberOfPositiveExamples;
    fn["numberOfWarpedPositiveExamples"] >> numberOfWarpedPositiveExamples;

    fn["groupRectanglesTheta"] >> groupRectanglesTheta;
}

void TrackerTLD::Params::write(cv::FileStorage& fs) const
{
    fs << "preFernMeasurements" << preFernMeasurements;
    fs << "preFerns" << preFerns;
    fs << "preFernPatchSize" << preFernPatchSize;

    fs << "numberOfMeasurements" << numberOfMeasurements;
    fs << "numberOfFerns" << numberOfFerns;
    fs << "fernPatchSize" << fernPatchSize;

    fs << "numberOfExamples" << numberOfExamples;
    fs << "examplePatchSize" << examplePatchSize;

    fs << "numberOfInitPositiveExamples" << numberOfInitPositiveExamples;
    fs << "numberOfInitWarpedPositiveExamples" << numberOfInitWarpedPositiveExamples;

    fs << "numberOfPositiveExamples" << numberOfPositiveExamples;
    fs << "numberOfWarpedPositiveExamples" << numberOfWarpedPositiveExamples;

    fs << "groupRectanglesTheta" << groupRectanglesTheta;
}

namespace tld
{
//class TrackerProxy
//{
//public:
//    virtual bool init(const Mat& image, const Rect2d& boundingBox) = 0;
//    virtual bool update(const Mat& image, Rect2d& boundingBox) = 0;
//    virtual ~TrackerProxy(){}
//};

//template<class T, class Tparams>
//class TrackerProxyImpl : public TrackerProxy
//{
//public:
//    TrackerProxyImpl(Tparams params = Tparams()) :params_(params){}
//    bool init(const Mat& image, const Rect2d& boundingBox)
//    {
//        trackerPtr = T::createTracker();
//        return trackerPtr->init(image, boundingBox);
//    }
//    bool update(const Mat& image, Rect2d& boundingBox)
//    {
//        return trackerPtr->update(image, boundingBox);
//    }
//private:
//    Ptr<T> trackerPtr;
//    Tparams params_;
//    Rect2d boundingBox_;
//};


class Integrator
{
public:

    struct IntegratorResult
    {
        const Rect objectToTrain;
        const Rect2d objectToResetTracker;
        const Rect objectToOutput;
        const Rect ObjectDetector;

        IntegratorResult(Rect actObjectToTrain, Rect2d actObjectToResetTracker, Rect actObjectToOutput, Rect actObjectDetector) :
            objectToTrain(actObjectToTrain), objectToResetTracker(actObjectToResetTracker), objectToOutput(actObjectToOutput), ObjectDetector(actObjectDetector) {}
    };

    Integrator():  preparing(0) {}

    const IntegratorResult getObjectToTrainFrom(const Mat_<uchar> &/*frame*/,
                                               const std::pair<Rect2d, double> &objectFromTracker,
                                               const std::vector<std::pair<Rect, double> > &objectsFromDetector);

private:
    int preparing;

private:
    static bool sortDetections(const std::pair<Rect, double> &candidate1, const std::pair<Rect, double> &candidate2);

//    static void updateCandidates(Ptr<Candidate> candidate, const Mat_<uchar> &frame);
//    static void incrementHints(Ptr<Candidate> candidate, const std::vector< std::pair<Rect, double> > &objectsFromDetector);
//    static bool sortPredicateConf(const Ptr<Candidate> &candidate1, const Ptr<Candidate> &candidate2);


//    static bool sortPredicateHints(const Ptr<Candidate> &candidate1);

//    static bool overlapPredicate(const Ptr<Candidate> candidate, const Rect &bb);
//    static double overlapIncPredicate(const std::pair<Rect, double> candidate, const Rect2d bb);
//    static bool selectCandidateForRemove(const Ptr<Candidate> candidate);
//    static Rect averageRects(const Rect &item1, const Rect &item2);
};

class TrackerTLDImpl : public TrackerTLD
{
public:
    TrackerTLDImpl(const TrackerTLD::Params &parameters = TrackerTLD::Params());
    void read(const FileNode& fn);
    void write(FileStorage& fs) const;

private:
    TrackerTLD::Params params;
    bool isTrackerOK;

private:

    bool initImpl(const Mat& image, const Rect2d& boundingBox);
    bool updateImpl(const Mat& image, Rect2d& boundingBox);

    Ptr<TrackerMedianFlow> medianFlow;
    Ptr<CascadeClassifier> cascadeClassifier;
    Ptr<Integrator> integrator;

    Rect roi;

};

}
}

#endif
