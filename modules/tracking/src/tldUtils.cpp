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

#include "tldUtils.hpp"

#include <map>
#include <iterator>

namespace cv
{
namespace tld
{

//Debug functions and variables
Rect2d etalon(14.0, 110.0, 20.0, 20.0);

std::string type2str(const Mat& mat)
{
  int type = mat.type();
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
  r += (chans + '0');

  return r;
}

//Scale & Blur image using scale Indx
//double scaleAndBlur(const Mat& originalImg, int scale, Mat& scaledImg, Mat& blurredImg, Size GaussBlurKernelSize, double scaleStep)
//{
//    double dScale = 1.0;
//    for( int i = 0; i < scale; i++, dScale *= scaleStep );
//    Size2d size = originalImg.size();
//    size.height /= dScale; size.width /= dScale;
//    resize(originalImg, scaledImg, size);
//    GaussianBlur(scaledImg, blurredImg, GaussBlurKernelSize, 0.0);
//    return dScale;
//}

bool comparartor(Overlaps::value_type a, Overlaps::value_type b) { return a.second > b.second;}

//std::vector<Hypothesis> getClosestN(const std::vector<Hypothesis> &hypothesis, const Rect &bBox, size_t n, double maxOverlap)
//{
//    std::vector<Hypothesis> res; res.reserve(n);

//    if(n == 0)
//        return res;

//    if(n >= hypothesis.size())
//        return hypothesis;
//    else
//    {
//        Overlaps overlaps;
//        overlaps.reserve(hypothesis.size());

//        for(size_t i = 0; i < hypothesis.size(); ++i)
//            overlaps.push_back(std::make_pair(i, overlap(bBox, hypothesis[i].bb)));

//        std::sort(overlaps.begin(), overlaps.end(), std::ptr_fun(comparartor));

//        Overlaps::const_iterator it = overlaps.begin();
//        while(res.size() < n && it != overlaps.end())
//        {
//            if(it->second < maxOverlap)
//            {
//                double ratio = double(hypothesis[it->first].bb.area()) / bBox.area();
//                if( 0.8 < ratio && ratio < 1.4)
//                    res.push_back(hypothesis[it->first]);
//            }
//            ++it;
//        }

//    }

//    return res;
//}

int getMedian(const std::vector<int>& values, int size)
{
    if( size == -1 )
        size = (int)values.size();
    std::vector<int> copy(values.begin(), values.begin() + size);
    std::sort(copy.begin(), copy.end());
    if( size % 2 == 0 )
        return (copy[size / 2 - 1] + copy[size / 2]) / 2;
    else
        return copy[(size - 1) / 2];
}

double overlap(const Rect& r1, const Rect& r2)
{
    double a1 = r1.area(), a2 = r2.area(), a0 = (r1&r2).area();
    return a0 / (a1 + a2 - a0);
}

void resample(const Mat& img, const RotatedRect& r2, Mat_<uchar>& samples)
{
    Mat_<float> M(2, 3), R(2, 2), Si(2, 2), s(2, 1), o(2, 1);

    R(0, 0) = (float)cos(r2.angle * CV_PI / 180);
    R(0, 1) = (float)(-sin(r2.angle * CV_PI / 180));
    R(1, 0) = (float)sin(r2.angle * CV_PI / 180);
    R(1, 1) = (float)cos(r2.angle * CV_PI / 180);

    Si(0, 0) = (float)(samples.cols / r2.size.width);
    Si(0, 1) = 0.0f;

    Si(1, 0) = 0.0f;
    Si(1, 1) = (float)(samples.rows / r2.size.height);

    s(0, 0) = (float)samples.cols;
    s(1, 0) = (float)samples.rows;

    o(0, 0) = r2.center.x;
    o(1, 0) = r2.center.y;

    Mat_<float> A(2, 2), b(2, 1);

    A = Si * R;
    b = s / 2.0 - Si * R * o;

    A.copyTo(M.colRange(Range(0, 2)));
    b.copyTo(M.colRange(Range(2, 3)));

    warpAffine(img, samples, M, samples.size());
}

void resample(const Mat& img, const Rect2d& r2, Mat_<uchar>& samples)
{
    Mat_<float> M(2, 3);
    M(0, 0) = (float)(samples.cols / r2.width); M(0, 1) = 0.0f; M(0, 2) = (float)(-r2.x * samples.cols / r2.width);
    M(1, 0) = 0.0f; M(1, 1) = (float)(samples.rows / r2.height); M(1, 2) = (float)(-r2.y * samples.rows / r2.height);
    warpAffine(img, samples, M, samples.size());
}

std::pair<double, Rect2d> augmentedOverlap(const Rect2d rect, const Rect2d bb)
{
    return std::make_pair(overlap(rect,bb), bb);
}

class MYSimilarRects
{
public:
    MYSimilarRects(double /*_eps*/) /*: eps(_eps)*/ {}
    inline bool operator()(const Rect& r1, const Rect& r2) const
    {
        if(double(r1.area()) / r2.area() > 1.5 || double(r1.area()) / r2.area() < 0.66)
            return false;

        if(double(r1.width) / r2.width > 1.5 || double(r1.width) / r2.width < 0.66)
            return false;

        if(double(r1.height) / r2.height > 1.5 || double(r1.height) / r2.height < 0.66)
            return false;

        const double delta = (std::min(r1.width, r2.width) + std::min(r1.height, r2.height)) * 0.5;

        const Point tl1 = r1.tl();
        const Point tl2 = r2.tl();

        const Point br1 = r1.br();
        const Point br2 = r2.br();

        if(sqrt((tl1 - tl2).ddot(tl1 - tl2)) < delta && sqrt((br1 - br2).ddot(br1 - br2)) < delta)
            return true;

        if(sqrt((tl1 - br2).ddot(tl1 - br2)) < delta && overlap(r1, r2) > 0)
            return true;

        return false;
    }
};



//void generateScanGrid(const Size &imageSize, const Size &actBBSize, std::vector<Rect> &res)
//{
//    Size2d bbSize = actBBSize;

//    generateScanGridInternal(imageSize, bbSize, res);

//    bool isDownScaleDone = false;
//    bool isUpScaleDone = false;

//    double upScale = 1., downScale = 1.;

//    while(!isDownScaleDone || !isUpScaleDone)
//    {
//        if(!isDownScaleDone)
//        {
//            downScale /= SCALE_STEP;
//            Size2d downSize = bbSize * downScale;
//            if(downSize.height < 20 || downSize.width < 20)
//                isDownScaleDone = true;
//            else
//                generateScanGrid(imageSize, downSize, res);
//        }

//        if(!isUpScaleDone)
//        {
//            upScale *= SCALE_STEP;
//            Size2d upSize = bbSize * upScale;
//            if(upSize.height > imageSize.height / 2 || upSize.width > imageSize.width / 2)
//                isUpScaleDone = true;
//            else
//                generateScanGridInternal(imageSize, upSize, res);
//        }
//    }
//}

//void generateScanGridInternal(const Size &imageSize, const Size2d &bbSize, std::vector<Rect> &res)
//{
//    double h = bbSize.height, w = bbSize.width;
//    for (double x = 0; x + w + 1.0 <= imageSize.width; x += 0.1 * w)
//    {
//        for (double y = 0; y + h + 1.0 <= imageSize.height; y += 0.1 * h)
//            res.push_back(Rect(x, y, w, h));
//    }

//}

}
}
