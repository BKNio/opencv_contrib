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
    preFernMeasurements = 13;
    preFerns = 15;
    preFernPatchSize = Size(15, 15);

    numberOfMeasurements = 13;
    numberOfFerns = 100;
    fernPatchSize = Size(26, 26);

    numberOfExamples = 150;
    examplePatchSize = Size(15, 15);

    numberOfInitPositiveExamples = 13;
    numberOfInitWarpedPositiveExamples = 20;

    numberOfPositiveExamples = 5;
    numberOfWarpedPositiveExamples = 5;

    groupRectanglesTheta = 0.25;
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
class TrackerProxy
{
public:
    virtual bool init(const Mat& image, const Rect2d& boundingBox) = 0;
    virtual bool update(const Mat& image, Rect2d& boundingBox) = 0;
    virtual ~TrackerProxy(){}
};

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


class TrackerTLDImpl : public TrackerTLD
{
public:
    TrackerTLDImpl(const TrackerTLD::Params &parameters = TrackerTLD::Params());
    void read(const FileNode& fn);
    void write(FileStorage& fs) const;

protected:
    class PExpert
    {
    public:
        PExpert(Size actFrameSize) : frameSize(actFrameSize) {}
        std::vector<Mat_<uchar> > generatePositiveExamples(const Mat_<uchar> &image, const Rect &bb, int numberOfsurroundBbs, int numberOfSyntheticWarped);

    private:
        mutable RNG rng;
        const Size frameSize;

    private:
        bool isRectOK(const cv::Rect &rect) const;
        std::vector<Rect> generateClosestN(const Rect &bBox, int n) const;
        std::vector<float> generateRandomValues(float range, int quantity) const ;
        Mat_<uchar> getWarped(const Mat_<uchar> &originalFrame, Rect bb, float shiftX, float shiftY, float scale, float rotation);
    };

    class NExpert
    {
    public:
        NExpert() {}
        std::vector< Mat_<uchar> > getNegativeExamples(const Mat_<uchar> &image, const Rect &object, std::vector<Rect> &detectedObjects);
    };


private:
    TrackerTLD::Params params;
    bool isTrackerOK;
    Ptr<PExpert> pExpert;
    Ptr<NExpert> nExpert;

private:

    bool initImpl(const Mat& image, const Rect2d& boundingBox);
    bool updateImpl(const Mat& image, Rect2d& boundingBox);

    Ptr<TrackerMedianFlow> medianFlow;
    Ptr<CascadeClassifier> cascadeClassifier;

};

}
}

#endif
