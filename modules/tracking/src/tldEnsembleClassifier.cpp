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

#include "tldEnsembleClassifier.hpp"

namespace cv
{
namespace tld
{

/*                   tldVarianceClassifier                   */

tldVarianceClassifier::tldVarianceClassifier(const Mat_<uchar> &originalImage, const Rect &bb, double actThreshold) :
    originalVariance(variance(originalImage(bb))), threshold(actThreshold)
{}

void tldVarianceClassifier::isObjects(const std::vector<Hypothesis> &hypothesis, const Mat_<uchar> &image, std::vector<bool> &answers) const
{
    CV_Assert(answers.empty());
    answers.reserve(hypothesis.size());

    std::pair< Mat_<double>, Mat_<double> > integralScaledImages;

    cv::integral(image, integralScaledImages.first, integralScaledImages.second, CV_64F, CV_64F);

    for(std::vector<Hypothesis>::const_iterator it = hypothesis.begin(); it != hypothesis.end(); ++it)
        answers.push_back(isObject(it->bb, integralScaledImages.first, integralScaledImages.second));
}

bool tldVarianceClassifier::isObject(const Rect &bb, const Mat_<double> &sum, const Mat_<double> &sumSq) const
{
    return variance(sum, sumSq, bb) >= originalVariance * threshold;
}

double tldVarianceClassifier::variance(const Mat_<uchar>& img)
{
    double p = 0., p2 = 0.;
    for( int i = 0; i < img.rows; i++ )
    {
        for( int j = 0; j < img.cols; j++ )
        {
            p += img.at<uchar>(i, j);
            p2 += img.at<uchar>(i, j) * img.at<uchar>(i, j);
        }
    }
    p /= (img.cols * img.rows);
    p2 /= (img.cols * img.rows);
    return p2 - p * p;
}

double tldVarianceClassifier::variance(const Mat_<double> &sum, const Mat_<double> &sumSq, const Rect &bb)
{
    const Point pt = bb.tl();
    const Size size = bb.size();
    const int x = (pt.x), y = (pt.y), width = (size.width), height = (size.height);

    CV_Assert(0 <= x && (x + width) < sum.cols && (x + width) < sumSq.cols);
    CV_Assert(0 <= y && (y + height) < sum.rows && (y + height) < sumSq.rows);

    double p = 0, p2 = 0;
    double a, b, c, d;

    a = sum(y, x);
    b = sum(y, x + width);
    c = sum(y + height, x);
    d = sum(y + height, x + width);
    p = (a + d - b - c) / (width * height);

    a = sumSq(y, x);
    b = sumSq(y, x + width);
    c = sumSq(y + height, x);
    d = sumSq(y + height, x + width);
    p2 = (a + d - b - c) / (width * height);

    return (p2 - p * p);
}


/*                   tldFernClassifier                   */

tldFernClassifier::tldFernClassifier(int numberOfMeasurementsPerFern, int reqNumberOfFerns, Size actNormilizedPatchSize):
    normilizedPatchSize(actNormilizedPatchSize), threshold(0.5), minSqDist(9)
{
    Ferns::value_type measurements;

    const int shift = 1;
    for(int i = shift; i < normilizedPatchSize.width - shift; ++i)
    {
        for(int j = shift; j < normilizedPatchSize.height - shift; ++j)
        {
            Point firstPoint(i,j);

#if 1
            for(int kk = shift; kk < normilizedPatchSize.width - shift; ++kk)
            {
                const Point diff = Point(kk, j) - firstPoint;
                if(diff.dot(diff) < minSqDist)
                    continue;

                measurements.push_back(std::make_pair(firstPoint, Point(kk, j)));
            }

            for(int kk = shift; kk < normilizedPatchSize.height -shift; ++kk)
            {
                const Point diff = Point(i, kk) - firstPoint;
                if(diff.dot(diff) < minSqDist)
                    continue;

                measurements.push_back(std::make_pair(firstPoint, Point(i, kk)));
            }
#else
            for(int ii = /*i + 1*/ shift; ii < normilizedPatchSize.width - shift; ++ii)
            {
                for(int jj = /*j + 1*/ shift; jj < normilizedPatchSize.height - shift; ++jj)
                {
                    Point secondPoint(ii,jj);

                    const Point diff = firstPoint - secondPoint;
                    if(diff.dot(diff) < 4)
                        continue;

                    measurements.push_back(std::make_pair(firstPoint, secondPoint));
                }
            }
#endif


        }
    }

    const int actNumberOfFerns = reqNumberOfFerns > 0 ? reqNumberOfFerns : int(measurements.size()) / numberOfMeasurementsPerFern;

    if(int(measurements.size()) < reqNumberOfFerns * numberOfMeasurementsPerFern)
        CV_Error(cv::Error::StsBadArg, "Not enough measurements");

    std::random_shuffle(measurements.begin(), measurements.end());

    Precedents::value_type emptyPrecedents(1 << numberOfMeasurementsPerFern, Point());
    Ferns::value_type::const_iterator originalMeasurementsIt = measurements.begin();

    ferns = Ferns(actNumberOfFerns);
    precedents = Precedents(actNumberOfFerns);

    for(int i = 0; i < actNumberOfFerns; ++i)
    {
        ferns[i].assign(originalMeasurementsIt, originalMeasurementsIt + numberOfMeasurementsPerFern);
        originalMeasurementsIt += numberOfMeasurementsPerFern;

        precedents[i] = emptyPrecedents;
    }

}

void tldFernClassifier::isObjects(const std::vector<Hypothesis> &hypothesis, const Mat_<uchar> &images, std::vector<bool> &answers) const
{
    CV_Assert(hypothesis.size() == answers.size());

    for(size_t i = 0; i < hypothesis.size(); ++i)
        if(answers[i])
            answers[i] = isObject(images(hypothesis[i].bb));
}

void tldFernClassifier::integratePositiveExample(const Mat_<uchar> &image)
{
    integrateExample(image, true);
}

void tldFernClassifier::integrateNegativeExample(const Mat_<uchar> &image)
{
    integrateExample(image, false);
}

bool tldFernClassifier::isObject(const Mat_<uchar> &image) const
{
    return getProbability(image) > threshold;
}

double tldFernClassifier::getProbability(const Mat_<uchar> &image) const
{

#ifdef USE_BLUR
    Mat_<uchar> blurred;
    GaussianBlur(image, blurred, Size(3,3), 0.);
#endif

    double accumProbability = 0.;
    for(size_t i = 0; i < ferns.size(); ++i)
    {
#ifdef FERN_DEBUG
    debugOutput = Mat();
#endif
#ifdef USE_BLUR
        int position = code(blurred, ferns[i]);
#else
        int position = code(image, ferns[i]);
#endif
        int posNum = precedents[i][position].x, negNum = precedents[i][position].y;

        if (posNum != 0 || negNum != 0)
            accumProbability += double(posNum) / (posNum + negNum);
#ifdef FERN_DEBUG
    imshow("debugOutput", debugOutput);
    waitKey();
#endif
    }

    return accumProbability / int(ferns.size());
}

int tldFernClassifier::code(const Mat_<uchar> &image, const Ferns::value_type &fern) const
{
    int position = 0;

    const float coeffX = float(image.cols - 1) / normilizedPatchSize.width;
    const float coeffY = float(image.rows - 1) / normilizedPatchSize.height;

    for(Ferns::value_type::const_iterator measureIt = fern.begin(); measureIt != fern.end(); ++measureIt)
    {
        position <<= 1;

        const Point2f p1(measureIt->first.x * coeffX, measureIt->first.y * coeffY);
        const Point2f p2(measureIt->second.x * coeffX, measureIt->second.y * coeffY);


#ifdef FERN_DEBUG
        if(debugOutput.empty())
            cvtColor(image, debugOutput, CV_GRAY2BGR);

        static RNG rng;
        Scalar color(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
        line(debugOutput, p1, p2, color);

        vals.first = getPixelVale(image, p1);
        vals.second = getPixelVale(image, p2);
#endif

        if(getPixelVale(image, p1) < getPixelVale(image, p2))
            position++;

    }
    return position;
}

void tldFernClassifier::integrateExample(const Mat_<uchar> &image, bool isPositive)
{
#ifdef USE_BLUR
    Mat_<uchar> blurred;
    GaussianBlur(image, blurred, Size(3,3), 0.);
#endif

    for(size_t i = 0; i < ferns.size(); ++i)
    {
#ifdef USE_BLUR
        int position = code(blurred, ferns[i]);
#else
        int position = code(image, ferns[i]);
#endif

        if(isPositive)
            precedents[i][position].x++;
        else
            precedents[i][position].y++;

    }
}

uchar tldFernClassifier::getPixelVale(const Mat_<uchar> &image, const Point2f point)
{
    CV_Assert(point.x >= 0.f && point.y >= 0.f);
    CV_Assert(point.x < image.cols && point.y < image.rows);

    uchar ret;
    cv::getRectSubPix(image, cv::Size(1,1), point, cv::_OutputArray(&ret, 1));

    return ret;
}

std::vector<Mat> tldFernClassifier::outputFerns(const Size &displaySize) const
{
    RNG rng;

    float scaleW = float(displaySize.width) / normilizedPatchSize.width;
    float scaleH = float(displaySize.height) / normilizedPatchSize.height;

    const Mat black(displaySize, CV_8UC3, Scalar::all(0));

    std::vector<Mat> fernsImages;
    fernsImages.reserve(ferns.size());

    for(Ferns::const_iterator fernIt = ferns.begin(); fernIt != ferns.end(); ++fernIt)
    {
        Mat copyBlack; black.copyTo(copyBlack);
        for(Ferns::value_type::const_iterator measureIt = fernIt->begin(); measureIt != fernIt->end(); ++measureIt)
        {
            Scalar color(rng.uniform(20,255), rng.uniform(20,255), rng.uniform(20,255));

            Point p1(cvRound(measureIt->first.x * scaleW), cvRound(measureIt->first.y * scaleH));
            Point p2(cvRound(measureIt->second.x * scaleW), cvRound(measureIt->second.y * scaleH));
            line(copyBlack, p1, p2, color, 2);
            circle(copyBlack, p1, 2, color, 2);
            circle(copyBlack, p2, 2, color, 2);
        }

        fernsImages.push_back(copyBlack);

    }

    return fernsImages;
}

/*                   tldNNClassifier                   */

tldNNClassifier::tldNNClassifier(size_t actMaxNumberOfExamples, Size actNormilizedPatchSize, double actTheta) :
    theta(actTheta), maxNumberOfExamples(actMaxNumberOfExamples),
    normilizedPatchSize(actNormilizedPatchSize), normilizedPatch(normilizedPatchSize)
{}

void tldNNClassifier::isObjects(const std::vector<Hypothesis> &hypothesis, const Mat_<uchar> &images, std::vector<bool> &answers) const
{
    CV_Assert(hypothesis.size() == answers.size());

#if 0
    nearestPrecedents.clear();
    distances.clear();
#endif

    for(size_t i = 0; i < hypothesis.size(); ++i)
        if(answers[i])
            answers[i] = isObject(images(hypothesis[i].bb));
}

std::pair<Mat, Mat> tldNNClassifier::outputModel() const
{
    const int sqrtSize = cvRound(std::sqrt(std::max(positiveExamples.size(), negativeExamples.size())) + 0.5f);
    const int outputWidth = sqrtSize * normilizedPatchSize.width, outputHeight = sqrtSize * normilizedPatchSize.height;

    cv::Mat positivePrecedents(outputHeight, outputWidth, CV_8U), negativePrecedents(outputHeight, outputWidth, CV_8U);

    cv::Rect actPositionPositive(cv::Point(), normilizedPatchSize);

    for(ExampleStorage::const_iterator it = positiveExamples.begin(); it != positiveExamples.end(); ++it)
    {
        if(actPositionPositive.x + it->cols > positivePrecedents.cols)
        {
            actPositionPositive.x = 0;
            actPositionPositive.y += it->rows;
        }

        CV_Assert(actPositionPositive.y + it->rows <= positivePrecedents.rows);

        it->copyTo(positivePrecedents(actPositionPositive));

        actPositionPositive.x += it->cols;

    }

    cv::Rect actPositionNegative(cv::Point(), normilizedPatchSize);

    for(ExampleStorage::const_iterator it = negativeExamples.begin(); it != negativeExamples.end(); ++it)
    {
        if(actPositionNegative.x + it->cols > negativePrecedents.cols)
        {
            actPositionNegative.x = 0;
            actPositionNegative.y += it->rows;
        }

        CV_Assert(actPositionNegative.y + it->rows <= negativePrecedents.rows);

        it->copyTo(negativePrecedents(actPositionNegative));

        actPositionNegative.x += it->cols;

    }

    return std::make_pair(positivePrecedents, negativePrecedents);
}

bool tldNNClassifier::isObject(const Mat_<uchar> &image) const
{
    if(image.size() != normilizedPatchSize)
        resize(image, normilizedPatch, normilizedPatchSize/*, INTER_NEAREST*/);
    else
        image.copyTo(normilizedPatch);

#ifdef USE_BLUR_NN
    Mat_<uchar> blurred;
    GaussianBlur(normilizedPatch, blurred, Size(3,3), 0.);
    blurred.copyTo(normilizedPatch);
#endif

    return Sr(normilizedPatch) > theta;
}

double tldNNClassifier::Sr(const Mat_<uchar> &patch) const
{
    double splus = 0., sminus = 0.;

#ifdef NNDEBUG
    double prevSplus = splus, prevSminus = sminus;
    positive = positiveExamples.begin();
    negative = negativeExamples.begin();
#endif

    for(ExampleStorage::const_iterator it = positiveExamples.begin(); it != positiveExamples.end(); ++it)
    {
        splus = std::max(splus, 0.5 * (NCC(*it, patch) + 1.0));
        if(prevSplus != splus)
        {
            positive = it;
            prevSplus = splus;
        }
    }

    for(ExampleStorage::const_iterator it = negativeExamples.begin(); it != negativeExamples.end(); ++it)
    {
        sminus = std::max(sminus, 0.5 * (NCC(*it, patch) + 1.0));
        if(prevSminus != sminus)
        {
            negative = it;
            prevSminus = sminus;
        }
    }


    if (splus + sminus == 0.0)
        return 0.0;

    return splus / (sminus + splus);
}

double tldNNClassifier::Sc(const Mat_<uchar> &patch) const
{
    double splus = 0., sminus = 0.;

    size_t mediana = positiveExamples.size() / 2 + positiveExamples.size() % 2;

    ExampleStorage::const_iterator end = positiveExamples.begin();
    for(size_t i = 0; i < mediana; ++i) ++end;

    for(ExampleStorage::const_iterator it = positiveExamples.begin(); it != end; ++it)
        splus = std::max(splus, 0.5 * (NCC(*it, patch) + 1.0));


    for(ExampleStorage::const_iterator it = positiveExamples.begin(); it != negativeExamples.end(); ++it)
        sminus = std::max(sminus, 0.5 * (NCC(*it, patch) + 1.0));


    if (splus + sminus == 0.0)
        return 0.0;

    return splus / (sminus + splus);
}

void tldNNClassifier::addExample(const Mat_<uchar> &example, std::list<Mat_<uchar> > &storage)
{
    CV_Assert(storage.size() <= maxNumberOfExamples);

    if(example.size() != normilizedPatchSize)
        resize(example, normilizedPatch, normilizedPatchSize);
    else
        normilizedPatch = example;

#ifdef USE_BLUR_NN
    Mat_<uchar> blurred;
    GaussianBlur(normilizedPatch, blurred, Size(3,3), 0.);
    blurred.copyTo(normilizedPatch);
#endif

    if(storage.size() == maxNumberOfExamples)
    {
        int randomIndex = rng.uniform(0, int(maxNumberOfExamples));
        ExampleStorage::iterator it = storage.begin();
        for(int i = 0; i < randomIndex; ++i)
            ++it;

        storage.erase(it);
    }

    storage.push_back(normilizedPatch.clone());
}

float tldNNClassifier::NCC(const Mat_<uchar> &patch1, const Mat_<uchar> &patch2)
{
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
    CV_Assert(p1Dev > 0.);

    const float p2Dev = p2SqSum / N- p2Mean * p2Mean;
    CV_Assert(p2Dev > 0.);

    return (p1p2Sum / N - p1Mean * p2Mean) / std::sqrt(p1Dev * p2Dev);
}


}
}
