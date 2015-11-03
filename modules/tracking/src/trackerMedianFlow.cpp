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

#include "precomp.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include <algorithm>
#include <limits.h>

namespace cv
{

#undef ALEX_DEBUG
#ifdef ALEX_DEBUG
#define dfprintf(x) fprintf x
#define dprintf(x) printf x
#else
#define dfprintf(x)
#define dprintf(x)
#endif

/*
 *  TrackerMedianFlow
 */
/*
 * TODO:
 * add "non-detected" answer in algo --> test it with 2 rects --> frame-by-frame debug in TLD --> test it!!
 * take all parameters out
 *              asessment framework
 *
 *
 * FIXME:
 * when patch is cut from image to compute NCC, there can be problem with size
 * optimize (allocation<-->reallocation)
 * optimize (remove vector.erase() calls)
 *       bring "out" all the parameters to TrackerMedianFlow::Param
 */

class TrackerMedianFlowImpl : public TrackerMedianFlow
{
public:
    TrackerMedianFlowImpl(TrackerMedianFlow::Params paramsIn):termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.3){params=paramsIn;isInit=false;}
    void read( const FileNode& fn );
    void write( FileStorage& fs ) const;
private:
    bool initImpl( const Mat& image, const Rect2d& boundingBox );
    bool updateImpl( const Mat& image, Rect2d& boundingBox );

    bool medianFlowImpl(Mat oldImage,Mat newImage,Rect2d& oldBox);
    Rect2d vote(const std::vector<Point2f>& oldPoints, const std::vector<Point2f>& newPoints, const Rect2d& oldRect, bool &isOK);

    template<typename T>
    T getMedian(const std::vector<T>& values);
    std::string type2str(int type);
    //void computeStatistics(std::vector<float>& data,int size=-1);
    void check_FB(const Mat& oldImage,const Mat& newImage,
                  const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status);
    void check_NCC(const Mat& oldImage,const Mat& newImage,
                   const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status);
    void checkDisplacement(const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status);

    inline float l2distance(Point2f p1,Point2f p2);

    static float NCC(const Mat_<uchar> &patch1, const Mat_<uchar> &patch2);

    TrackerMedianFlow::Params params;
    TermCriteria termcrit;
};

class TrackerMedianFlowModel : public TrackerModel
{
public:
    TrackerMedianFlowModel(TrackerMedianFlow::Params /*params*/){}
    Rect2d getBoundingBox(){return boundingBox_;}
    void setBoudingBox(Rect2d boundingBox){boundingBox_=boundingBox;}
    Mat getImage(){return image_;}
    void setImage(const Mat& image){image.copyTo(image_);}
protected:
    Rect2d boundingBox_;
    Mat image_;
    void modelEstimationImpl( const std::vector<Mat>& /*responses*/ ){}
    void modelUpdateImpl(){}
};

/*
 * Parameters
 */
TrackerMedianFlow::Params::Params()
{
    pointsInGrid = 10;
    pointsDx = 4.f;
    pointsDy = 4.f;
}

void TrackerMedianFlow::Params::read( const cv::FileNode& fn )
{
    pointsInGrid=fn["pointsInGrid"];
    pointsDx = fn["pointsDx"];
    pointsDy = fn["pointsDy"];

    CV_Assert(pointsDx > 0.f && pointsDy > 0.f);
}

void TrackerMedianFlow::Params::write( cv::FileStorage& fs ) const
{
    fs << "pointsInGrid" << pointsInGrid;
    fs << "pointsDx" << pointsDx;
    fs << "pointsDy" << pointsDy;
}

void TrackerMedianFlowImpl::read( const cv::FileNode& fn )
{
    params.read( fn );
}

void TrackerMedianFlowImpl::write( cv::FileStorage& fs ) const
{
    params.write( fs );
}

Ptr<TrackerMedianFlow> TrackerMedianFlow::createTracker(const TrackerMedianFlow::Params &parameters)
{
    return Ptr<TrackerMedianFlowImpl>(new TrackerMedianFlowImpl(parameters));
}

bool TrackerMedianFlowImpl::initImpl( const Mat& image, const Rect2d& boundingBox )
{
    model = Ptr<TrackerMedianFlowModel>(new TrackerMedianFlowModel(params));

    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setImage(image);

    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setBoudingBox(boundingBox);

    return true;
}

bool TrackerMedianFlowImpl::updateImpl( const Mat& image, Rect2d& boundingBox )
{
    Mat oldImage=((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->getImage();

    Rect2d oldBox=((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->getBoundingBox();

    if(!medianFlowImpl(oldImage,image,oldBox))
        return false;

    boundingBox=oldBox;

    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setImage(image);
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setBoudingBox(oldBox);
    return true;
}

std::string TrackerMedianFlowImpl::type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = (uchar)(1 + (type >> CV_CN_SHIFT));

    switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}
bool TrackerMedianFlowImpl::medianFlowImpl(Mat oldImage,Mat newImage,Rect2d& oldBox)
{
    std::vector<Point2f> pointsToTrackOld,pointsToTrackNew;

    Mat oldImage_gray,newImage_gray;

    if(oldImage.type() == CV_8UC3)
        cvtColor( oldImage, oldImage_gray, COLOR_BGR2GRAY );
    else if(oldImage.type() == CV_8U)
        oldImage.copyTo(oldImage_gray);
    else
        CV_Assert(0);


    if(newImage.type() == CV_8UC3)
        cvtColor( newImage, newImage_gray, COLOR_BGR2GRAY );
    else if(newImage.type() == CV_8U)
        newImage.copyTo(newImage_gray);
    else
        CV_Assert(0);

//    for(int i = 0; i < params.pointsInGrid; i++)
//    {
//        for(int j = 0; j<params.pointsInGrid; j++)
//        {
//            pointsToTrackOld.push_back(
//                        Point2f((float)(oldBox.x+((oldBox.width)/params.pointsInGrid)*j+.5*oldBox.width/params.pointsInGrid),
//                                (float)(oldBox.y+((oldBox.height)/params.pointsInGrid)*i+.5*oldBox.height/params.pointsInGrid))
//                        );
//        }
//    }


    for(int i = 1;; ++i)
    {
        if(i * params.pointsDx > oldBox.width - params.pointsDx)
            break;

        for(int j = 1;; ++j)
        {
            if(j * params.pointsDy > oldBox.height - params.pointsDy)
                break;

            Point2f point(oldBox.x + i * params.pointsDx, oldBox.y + j * params.pointsDy);

            CV_Assert(oldBox.contains(point));
            pointsToTrackOld.push_back(point);
        }
    }


    std::vector<uchar> status;
    calcOpticalFlowPyrLK(oldImage_gray, newImage_gray, pointsToTrackOld, pointsToTrackNew, status, noArray(), Size(7,7), 3, termcrit,0);

    {
        CV_Assert(pointsToTrackNew.size() == pointsToTrackOld.size());
        CV_Assert(pointsToTrackNew.size() == status.size());

        std::vector<Point2f> pointsToTrackOldCopy(pointsToTrackOld), pointsToTrackNewCopy(pointsToTrackNew);
        pointsToTrackOld.clear();
        pointsToTrackNew.clear();

        for(size_t index = 0; index < status.size(); ++index)
        {
            if(status[index])
            {
                pointsToTrackOld.push_back(pointsToTrackOldCopy[index]);
                pointsToTrackNew.push_back(pointsToTrackNewCopy[index]);
            }
        }
    }

    CV_Assert(pointsToTrackNew.size() == pointsToTrackOld.size());

    if(pointsToTrackNew.size() < 3)
        return false;

    std::vector<bool> filter_status;

    check_FB(oldImage_gray, newImage_gray, pointsToTrackOld, pointsToTrackNew, filter_status);
    /*{
        Mat_<uchar> newImageGrayCopy; newImage_gray.copyTo(newImageGrayCopy);
        for(std::vector<Point2f>::iterator point = pointsToTrackNew.begin(); point != pointsToTrackNew.end(); ++point)
        {
            if(filter_status[std::distance(pointsToTrackNew.begin(), point)])
                circle(newImageGrayCopy, *point, 2, Scalar::all(255));
            else
                circle(newImageGrayCopy, *point, 2, Scalar::all(0));
        }

        imshow("FB check", newImageGrayCopy);
    }*/

    check_NCC(oldImage_gray, newImage_gray, pointsToTrackOld, pointsToTrackNew, filter_status);
    /*{
        Mat_<uchar> newImageGrayCopy; newImage_gray.copyTo(newImageGrayCopy);
        for(std::vector<Point2f>::iterator point = pointsToTrackNew.begin(); point != pointsToTrackNew.end(); ++point)
        {
            if(filter_status[std::distance(pointsToTrackNew.begin(), point)])
                circle(newImageGrayCopy, *point, 2, Scalar::all(255));
            else
                circle(newImageGrayCopy, *point, 2, Scalar::all(0));
        }

        imshow("ncc check", newImageGrayCopy);
    }*/

    checkDisplacement(pointsToTrackOld, pointsToTrackNew, filter_status);
    /*{
        Mat_<uchar> newImageGrayCopy; newImage_gray.copyTo(newImageGrayCopy);
        for(std::vector<Point2f>::iterator point = pointsToTrackNew.begin(); point != pointsToTrackNew.end(); ++point)
        {
            if(filter_status[std::distance(pointsToTrackNew.begin(), point)])
                circle(newImageGrayCopy, *point, 2, Scalar::all(255));
            else
                circle(newImageGrayCopy, *point, 2, Scalar::all(0));
        }

        imshow("displacement check", newImageGrayCopy);
    }*/


    {
        CV_Assert(pointsToTrackNew.size() == pointsToTrackOld.size());
        CV_Assert(pointsToTrackNew.size() == filter_status.size());

        std::vector<Point2f> pointsToTrackOldCopy(pointsToTrackOld), pointsToTrackNewCopy(pointsToTrackNew);
        pointsToTrackOld.clear();
        pointsToTrackNew.clear();

        for(size_t index = 0; index < filter_status.size(); ++index)
        {
            if(filter_status[index])
            {
                pointsToTrackOld.push_back(pointsToTrackOldCopy[index]);
                pointsToTrackNew.push_back(pointsToTrackNewCopy[index]);
            }
        }
    }


    CV_Assert(pointsToTrackNew.size() == pointsToTrackOld.size());

    if(pointsToTrackOld.size() < 3)
        return std::cout << "too low points number" << std::endl, false;

    bool isOK = false;
    oldBox = vote(pointsToTrackOld,pointsToTrackNew, oldBox, isOK);

    if(!isOK)
        std::cout << "vote return error" << std::endl;

    return isOK;
}

Rect2d TrackerMedianFlowImpl::vote(const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,const Rect2d& oldRect, bool& isOK)
{
    CV_Assert(oldPoints.size() == newPoints.size());
    CV_Assert(oldPoints.size() >= 3);

    Rect2d newRect;
    Point2d newTl(oldRect.x, oldRect.y);

    std::vector<float> translationsX; translationsX.reserve(oldPoints.size());

    for(size_t index = 0; index < oldPoints.size(); ++index)
        translationsX.push_back(newPoints[index].x-oldPoints[index].x);

    const float xshift = getMedian(translationsX);

    if(xshift > 20.f)
        return std::cout << "vote: wrong shiftX" << std::endl, isOK = false, Rect2d();

    newTl.x += xshift;

    std::vector<float> translationsY; translationsY.reserve(oldPoints.size());
    for(size_t index = 0; index < oldPoints.size(); ++index)
        translationsY.push_back(newPoints[index].y-oldPoints[index].y);

    const float yshift = getMedian(translationsY);

    if(yshift > 20.f)
        return std::cout << "vote: wrong shiftY" << std::endl, isOK = false, Rect2d();

    newTl.y += yshift;

    std::vector<float> scales; scales.reserve(oldPoints.size() * oldPoints.size());
    for(size_t i = 0; i < oldPoints.size(); ++i)
    {
        for(size_t j = 0; j < i; ++j)
        {
            float nd = l2distance(newPoints[i], newPoints[j]);
            float od = l2distance(oldPoints[i], oldPoints[j]);

            CV_Assert(std::fabs(od) > 1e-2);
            scales.push_back(nd / od);
        }
    }

    float scale = getMedian(scales);

    if(scale > 1.5f || scale < 0.66f)
        return std::cout << "vote: wrong scale" << std::endl, isOK = false, Rect2d();

    CV_Assert(std::fabs(scale) > 1e-3);

    newRect.x = newTl.x;
    newRect.y = newTl.y;
    newRect.width = oldRect.width * scale;
    newRect.height = oldRect.height * scale;

    isOK = true;

    return newRect;
}

template<typename T>
T TrackerMedianFlowImpl::getMedian(const std::vector<T>& values)
{
    std::vector<T> copyValues(values);

    size_t size = values.size();

    std::sort(copyValues.begin(),copyValues.end());
    if(size % 2 == 0)
        return (copyValues[size / 2 - 1] + copyValues[size / 2]) / ( (T)2.0);
    else
        return copyValues[size / 2];

}

float TrackerMedianFlowImpl::l2distance(Point2f p1,Point2f p2)
{
    float dx=p1.x-p2.x, dy=p1.y-p2.y;
    return sqrt(dx*dx+dy*dy);
}
void TrackerMedianFlowImpl::check_FB(const Mat& oldImage,const Mat& newImage,
                                     const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status)
{

    CV_Assert(oldPoints.size() == newPoints.size());

    if(status.empty())
        status = std::vector<bool>(oldPoints.size(),true);

    std::vector<uchar> LKstatus;
    std::vector<Point2f> pointsToTrackReprojection;

    calcOpticalFlowPyrLK(newImage, oldImage, newPoints, pointsToTrackReprojection, LKstatus, noArray(),Size(7,7), 3, termcrit, 0);

    for(size_t index = 0; index < oldPoints.size(); ++index)
    {
        if(LKstatus[index])
        {
            if(l2distance(oldPoints[index], pointsToTrackReprojection[index]) > 1.)
                status[index] = 0;
            else
                status[index] = 1;
        }
        else
            status[index] = 0;
    }
}

float TrackerMedianFlowImpl::NCC(const Mat_<uchar> &patch1, const Mat_<uchar> &patch2)
{
    CV_Assert(patch1.size().area() > 0);
    CV_Assert(patch1.size() == patch2.size());

    const float N = patch1.size().area();

    float p1Sum = 0., p2Sum = 0., p1p2Sum = 0., p1SqSum = 0. , p2SqSum = 0.;

    for(int i = 0; i < patch1.rows; ++i)
    {
        for(int j = 0; j < patch1.cols; ++j)
        {
            const float p1 = patch1.at<uchar>(i,j);
            const float p2 = patch2.at<uchar>(i,j);

            p1Sum += p1;
            p2Sum += p2;

            p1p2Sum += p1*p2;

            p1SqSum += p1*p1;
            p2SqSum += p2*p2;

        }
    }

    const float p1Mean = p1Sum / N;
    const float p2Mean = p2Sum / N;

    const float p1Dev = p1SqSum / N- p1Mean * p1Mean;
    if(std::fabs(p1Dev) < 1e-3)
        return std::numeric_limits<float>::quiet_NaN();

    const float p2Dev = p2SqSum / N- p2Mean * p2Mean;
    if(std::fabs(p2Dev) < 1e-3)
        return std::numeric_limits<float>::quiet_NaN();


    return(p1p2Sum / N - p1Mean * p2Mean) / std::sqrt(p1Dev * p2Dev);
}

void TrackerMedianFlowImpl::check_NCC(const Mat& oldImage,const Mat& newImage,
                                      const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status)
{
    CV_Assert(oldPoints.size() == newPoints.size());
    CV_Assert(status.size() == newPoints.size());

    const Size patchSize(30,30);
    std::vector<float> ncc;
    Mat_<uchar> p1,p2;

    for (size_t i = 0; i < oldPoints.size(); i++)
    {
        if(status[i] == 0)
            continue;

        getRectSubPix(oldImage, patchSize, oldPoints[i], p1, CV_8U);
        getRectSubPix(newImage, patchSize, newPoints[i], p2, CV_8U);

        const float currentNCC = NCC(p1, p2);

        if(!cvIsNaN(currentNCC))
            ncc.push_back(currentNCC);
        else
            status[i] = 0;

    }

    if(ncc.empty())
        return;

    float median = getMedian(ncc);

    if(median < .8f)
    {
        std::cout << "bad NCC median" << median << std::endl;
        status = std::vector<bool>(status.size(), false);
    }

//    if(median == 1.f)
//    {
////        imshow("old image", oldImage);
////        imshow("new image", newImage);
////        waitKey();
//        std::cout << "median " << median << std::endl;
//    }

    size_t nccPos = 0;
    for(size_t i = 0; i < oldPoints.size(); i++)
        if(status[i])
            status[i] = status[i] & (ncc[nccPos++] > std::min(.9f, median));

    CV_Assert(nccPos = ncc.size());

}

void TrackerMedianFlowImpl::checkDisplacement(const std::vector<Point2f> &oldPoints, const std::vector<Point2f> &newPoints, std::vector<bool> &status)
{
    CV_Assert(oldPoints.size() == newPoints.size());
    CV_Assert(status.size() == oldPoints.size());

    std::vector<float> displacement; displacement.reserve(oldPoints.size());

    for(size_t index = 0; index < oldPoints.size(); ++index)
    {
        if(!status[index])
            continue;

        displacement.push_back(l2distance(oldPoints[index], newPoints[index]));
    }

    if(displacement.size() == 0)
        return;

    float medianDisplacement = getMedian(displacement);

    if(medianDisplacement > 25.f)
    {
        std::cout << "too big displacement" << std::endl;
        status = std::vector<bool>(status.size(), false);
        return;
    }

    size_t indexDisplacement = 0;
    for(size_t index = 0; index < oldPoints.size(); ++index)
    {
        if(!status[index])
            continue;

        if(displacement[indexDisplacement] > 1.1 * medianDisplacement || displacement[indexDisplacement] < 0.9 * medianDisplacement)
            status[index] = false;
        indexDisplacement++;
    }

    CV_Assert(indexDisplacement == displacement.size());
}

} /* namespace cv */
