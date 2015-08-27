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

void tldVarianceClassifier::isObjects(const std::vector<Hypothesis> &hypothesis, const std::vector<Mat_<uchar> > &scaledImages, std::vector<bool> &answers) const
{
    CV_Assert(answers.empty());
    answers.reserve(hypothesis.size());

    std::vector<std::pair< Mat_<double>, Mat_<double> > > integralScaledImages(scaledImages.size());

    for(size_t i = 0; i < scaledImages.size(); ++i)
        cv::integral(scaledImages[i], integralScaledImages[i].first, integralScaledImages[i].second, CV_64F, CV_64F);

    for(std::vector<Hypothesis>::const_iterator it = hypothesis.begin(); it != hypothesis.end(); ++it)
        answers.push_back(isObject(it->bb, integralScaledImages[it->scaleId].first, integralScaledImages[it->scaleId].second));
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


tldFernClassifier::tldFernClassifier(const Size &initialSize, int actNumberOfFerns, int actNumberOfMeasurements):
    originalSize(initialSize), numberOfFerns(actNumberOfFerns), numberOfMeasurements(actNumberOfMeasurements),
    threshold(0.5), ferns(actNumberOfFerns), precedents(actNumberOfFerns)
{
    CV_Assert(originalSize.area() * (originalSize.width + originalSize.height) >= numberOfFerns * numberOfMeasurements); //is it enough measurements

    Ferns::value_type measurements;

    measurements.reserve(originalSize.area() * (originalSize.width + originalSize.height));

    for(int i = 0; i < originalSize.width; ++i) //generating all possible horizontal and vertical pixel comprations
    {
        for(int j = 0; j < originalSize.height; ++j)
        {
            Point firstPoint(i,j);

            for(int kk = 0; kk < originalSize.width; ++kk)
            {
                if(kk == i)
                    continue;

                measurements.push_back(std::make_pair(firstPoint, Point(kk, j)));
            }

            for(int kk = 0; kk < originalSize.height; ++kk)
            {
                if(kk == j)
                    continue;

                measurements.push_back(std::make_pair(firstPoint, Point(i, kk)));
            }

        }
    }

    std::random_shuffle(measurements.begin(), measurements.end());

    Precedents::value_type emptyPrecedents(1 << numberOfMeasurements, Point());

    Ferns::value_type::const_iterator originalMeasurementsIt = measurements.begin();
    for(size_t i = 0; i < size_t(numberOfFerns); ++i)
    {
        ferns[i].assign(originalMeasurementsIt, originalMeasurementsIt + numberOfMeasurements);
        originalMeasurementsIt += numberOfMeasurements;

        precedents[i] = emptyPrecedents;
    }

}

void tldFernClassifier::isObjects(const std::vector<Hypothesis> &hypothesis, const std::vector<Mat_<uchar> > &scaledImages, std::vector<bool> &answers) const
{
    CV_Assert(hypothesis.size() == answers.size());

    for(size_t i = 0; i < hypothesis.size(); ++i)
        if(answers[i])
            answers[i] = isObject(scaledImages[i](hypothesis[i].bb));
}

void tldFernClassifier::integratePositiveExample(const Mat_<uchar> &image)
{
    CV_Assert(image.size() == originalSize);
    integrateExample(image, true);
}

void tldFernClassifier::integrateNegativeExample(const Mat_<uchar> &image)
{
    CV_Assert(image.size() == originalSize);
    integrateExample(image, false);
}

bool tldFernClassifier::isObject(const Mat_<uchar> &object) const
{
    return getProbability(object) > threshold;
}

double tldFernClassifier::getProbability(const Mat_<uchar> &image) const
{
    CV_Assert(image.size() == originalSize);

    double accumProbability = 0.;
    for(size_t i = 0; i < size_t(numberOfFerns); ++i)
    {
        int position = code(image, ferns[i]);

        int posNum = precedents[i][position].x, negNum = precedents[i][position].y;

        if (posNum != 0 || negNum != 0)
            accumProbability += double(posNum) / (posNum + negNum);
    }

    return accumProbability / numberOfFerns;
}

int tldFernClassifier::code(const Mat_<uchar> &image, const Ferns::value_type &fern) const
{
    int position = 0;
    for(Ferns::value_type::const_iterator measureIt = fern.begin(); measureIt != fern.end(); ++measureIt)
    {
        position <<= 1;
        if(image.at<uchar>(measureIt->first) < image.at<uchar>(measureIt->second))
            position++;
    }
    return position;
}

void tldFernClassifier::integrateExample(const Mat_<uchar> &image, bool isPositive)
{
    for(size_t i = 0; i < ferns.size(); ++i)
    {
        int position = code(image, ferns[i]);

        if(isPositive)
            precedents[i][position].x++;
        else
            precedents[i][position].y++;
    }
}

std::vector<Mat> tldFernClassifier::outputFerns(const Size &displaySize) const
{
    RNG rng;

    double scaleW = double(displaySize.width) / originalSize.width;
    double scaleH = double(displaySize.height) / originalSize.height;

    const Mat black(displaySize, CV_8UC3, Scalar::all(0));

    std::vector<Mat> fernsImages;
    fernsImages.reserve(numberOfFerns);

    for(Ferns::const_iterator fernIt = ferns.begin(); fernIt != ferns.end(); ++fernIt)
    {
        Mat copyBlack; black.copyTo(copyBlack);
        for(Ferns::value_type::const_iterator measureIt = fernIt->begin(); measureIt != fernIt->end(); ++measureIt)
        {
            Scalar color(rng.uniform(20,255), rng.uniform(20,255), rng.uniform(20,255));

            Point p1(cvRound(measureIt->first.x * scaleW), cvRound(measureIt->first.y * scaleH));
            Point p2(cvRound(measureIt->second.x * scaleW), cvRound(measureIt->second.y * scaleH));
            line(copyBlack, p1, p2, color, 4);
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

void tldNNClassifier::isObjects(const std::vector<Hypothesis> &hypothesis, const std::vector<Mat_<uchar> > &scaledImages, std::vector<bool> &answers) const
{
    CV_Assert(hypothesis.size() == answers.size());

#ifdef DEBUG
    nearestPrecedents.clear();
    distances.clear();
#endif

    for(size_t i = 0; i < hypothesis.size(); ++i)
        if(answers[i])
            answers[i] = isObject(scaledImages[hypothesis[i].scaleId](hypothesis[i].bb));
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

std::pair<Mat, Mat> tldNNClassifier::outputNearestPrecedents(int hypothesisIndex) const
{
    std::pair<Mat, Mat> precedents = outputModel();

    ExampleStorage::const_iterator nearestPositiveExample = nearestPrecedents[hypothesisIndex].first;
    ExampleStorage::const_iterator nearestNegativeExample = nearestPrecedents[hypothesisIndex].second;

    const int sqrtSize = cvRound(std::sqrt(std::max(positiveExamples.size(), negativeExamples.size())) + 0.5f);

    Rect positiveBB;
    if(nearestPositiveExample != positiveExamples.end())
    {
        int index = std::distance(positiveExamples.begin(), nearestPositiveExample);

        const int row = index / sqrtSize;
        const int col = index % sqrtSize;

        Point tl(col * normilizedPatchSize.width /*- 3*/, row * normilizedPatchSize.height /*- 3*/);
        positiveBB = cv::Rect(tl, cv::Size(normilizedPatchSize.width /*+ 6*/, normilizedPatchSize.height /*+ 6*/));

        /*rectangle(precedents.first, positiveBB, Scalar::all(255));*/
    }

    Rect negativeBB;
    if(nearestNegativeExample != negativeExamples.end())
    {
        int index = std::distance(negativeExamples.begin(), nearestNegativeExample);

        const int row = index / sqrtSize;
        const int col = index % sqrtSize;

        Point tl(col * normilizedPatchSize.width /*- 3*/, row * normilizedPatchSize.height /*- 3*/);
        negativeBB = cv::Rect(tl, cv::Size(normilizedPatchSize.width /*+ 6*/, normilizedPatchSize.height /*+ 6*/));

        /*rectangle(precedents.second, negativeBB, Scalar::all(255));*/
    }

//    std::cout << "----------------------" << std::endl;
//    std::cout << "splus " << distances[hypothesisIndex].first << std::endl;
//    std::cout << "sminus " << distances[hypothesisIndex].second << std::endl;

    /*return precedents;*/

    return std::make_pair(precedents.first(positiveBB), precedents.second(negativeBB));
}

std::pair<float, float> tldNNClassifier::getDistancesToNearestPrecedents(int hypothesisIndex) const
{
    return std::make_pair(distances[hypothesisIndex].first, distances[hypothesisIndex].second);
}

bool tldNNClassifier::isObject(const Mat_<uchar> &object) const
{
    if(object.size() != normilizedPatchSize)
        resize(object, normilizedPatch, normilizedPatchSize, INTER_NEAREST);
    else
        object.copyTo(normilizedPatch);

    return Sr(normilizedPatch) > theta;
}

double tldNNClassifier::Sr(const Mat_<uchar> &patch) const
{
    double splus = 0., sminus = 0.;

#ifdef DEBUG
    double prevSplus = splus;
    ExampleStorage::const_iterator nearestPositivePrecedent = positiveExamples.end();
#endif

    for(ExampleStorage::const_iterator it = positiveExamples.begin(); it != positiveExamples.end(); ++it)
    {
        splus = std::max(splus, 0.5 * (NCC(*it, patch) + 1.0));
#ifdef DEBUG
        if(prevSplus != splus)
            prevSplus = splus, nearestPositivePrecedent = it;
#endif
    }

#ifdef DEBUG
    double prevSminus = sminus;
    ExampleStorage::const_iterator nearestNegativePrecedent = negativeExamples.end();
#endif

    for(ExampleStorage::const_iterator it = negativeExamples.begin(); it != negativeExamples.end(); ++it)
    {
        sminus = std::max(sminus, 0.5 * (NCC(*it, patch) + 1.0));
#ifdef DEBUG
        if(prevSminus != sminus)
            prevSminus = sminus, nearestNegativePrecedent = it;
#endif
    }

#ifdef DEBUG
    nearestPrecedents.push_back(std::make_pair(nearestPositivePrecedent, nearestNegativePrecedent));
    distances.push_back(std::make_pair(splus, sminus));
#endif

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
        resize(example, normilizedPatch, normilizedPatchSize, INTER_NEAREST);
    else
        normilizedPatch = example;


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

    for(int i = 0; i < patch1.cols; ++i)
    {
        for(int j = 0; j < patch1.rows; ++j)
        {
            const float p1 = patch1.at<uchar>(j,i);
            const float p2 = patch2.at<uchar>(j,i);

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

//    if(p1Dev <= 0.)
//        imwrite("/tmp/zerostdDev1.png", patch1);

    CV_Assert(p1Dev > 0.);

    const float p2Dev = p2SqSum / N- p2Mean * p2Mean;

//    if(p2Dev <= 0.)
//        imwrite("/tmp/zerostdDev2.png", patch2);

    CV_Assert(p2Dev > 0.);

    return (p1p2Sum / N - p1Mean * p2Mean) / std::sqrt(p1Dev * p2Dev);
}


}
}
