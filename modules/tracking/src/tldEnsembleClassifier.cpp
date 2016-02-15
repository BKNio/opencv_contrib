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

#include <sys/time.h>

#include <numeric>

#include "tldEnsembleClassifier.hpp"


namespace cv
{
namespace tld
{

/*                   tldVarianceClassifier                   */

VarianceClassifier::VarianceClassifier(double actLowCoeff, double actHighCoeff) : actVariance(-1.), lowCoeff(actLowCoeff), hightCoeff(actHighCoeff) {}

void VarianceClassifier::isObjects(const std::vector<Hypothesis> &hypothesis, const Mat_<uchar> &image, std::vector<Answers> &answers) const
{
    CV_Assert(actVariance > 0.);
    CV_Assert(answers.size() == hypothesis.size());

    cv::integral(image, integral, integralSq, CV_64F, CV_64F);

    for(size_t index = 0; index < hypothesis.size(); ++index)
        answers[index] = isObject(hypothesis[index].bb);
}

void VarianceClassifier::integratePositiveExamples(const std::vector<Mat_<uchar> > &examples)
{
    if(examples.empty())
        return;

    actVariance = 0.;

    for(std::vector< Mat_<uchar> >::const_iterator positiveExample = examples.begin(); positiveExample != examples.end(); ++positiveExample)
        actVariance += variance(*positiveExample);

    actVariance /= double(examples.size());
}

bool VarianceClassifier::isObject(const Rect &bb) const
{
    const double curVariance = variance(integral, integralSq, bb);
    return curVariance >= lowCoeff * actVariance && curVariance <= hightCoeff * actVariance;
}

double VarianceClassifier::variance(const Mat_<uchar>& img)
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

double VarianceClassifier::variance(const Mat_<double> &sum, const Mat_<double> &sumSq, const Rect &bb)
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

FernClassifier::FernClassifier(int numberOfMeasurementsPerFern, int reqNumberOfFerns, Size actNormilizedPatchSize, double actThreshold):
    patchSize(actNormilizedPatchSize), threshold(actThreshold), minSqDist(4)
{
    Ferns::value_type measurements;
    const int shift = 1;
    for(int i = shift; i < patchSize.width - shift; ++i)
    {
        for(int j = shift; j < patchSize.height - shift; ++j)
        {
            Point firstPoint(i,j);

            for(int kk = shift; kk < patchSize.width - shift; ++kk)
            {
                const Point diff = Point(kk, j) - firstPoint;
                if(diff.dot(diff) <= minSqDist)
                    continue;

                measurements.push_back(std::make_pair(firstPoint, Point(kk, j)));
            }

            for(int kk = shift; kk < patchSize.height -shift; ++kk)
            {
                const Point diff = Point(i, kk) - firstPoint;
                if(diff.dot(diff) <= minSqDist)
                    continue;

                measurements.push_back(std::make_pair(firstPoint, Point(i, kk)));
            }

        }
    }

    const int actNumberOfFerns = reqNumberOfFerns > 0 ? reqNumberOfFerns : int(measurements.size()) / numberOfMeasurementsPerFern;

    if(int(measurements.size()) < reqNumberOfFerns * numberOfMeasurementsPerFern)
        CV_Error(cv::Error::StsBadArg, "Not enough measurements");

    std::srand(0);
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

void FernClassifier::isObjects(const std::vector<Hypothesis> &hypothesis, const Mat_<uchar> &image, std::map <double, Mat_<uchar> > &scaledStorage, std::vector<Answers> &answers) const
{
    CV_Assert(hypothesis.size() == answers.size());

    /*Mat_<uchar> blurred;
    GaussianBlur(image, blurred, Size(3,3), 0.);*/

    for(size_t i = 0; i < hypothesis.size(); ++i)
    {
        if(answers[i])
        {

            const double scaleFactorX = double(patchSize.width) / hypothesis[i].bb.width;
            const double scaleFactorY = double(patchSize.height) / hypothesis[i].bb.height;

            if(scaledStorage.find(hypothesis[i].scale) == scaledStorage.end())
            {
                Mat_<uchar> resized;
                resize(image, resized, Size(), scaleFactorX, scaleFactorY);

                scaledStorage[hypothesis[i].scale] = resized;
            }

            const Point newPoint(cvRound(hypothesis[i].bb.x * scaleFactorX), cvRound(hypothesis[i].bb.y * scaleFactorY));
            const Rect newRect(newPoint, patchSize);

            answers[i]= isObject(scaledStorage[hypothesis[i].scale](newRect));
        }
    }
}

void FernClassifier::integratePositiveExamples(const std::vector<Mat_<uchar> > &examples)
{
    for(std::vector< Mat_<uchar> >::const_iterator example = examples.begin(); example != examples.end(); ++example)
        integrateExample(*example, true);

}

void FernClassifier::integrateNegativeExamples(const std::vector<Mat_<uchar> > &examples)
{
    for(std::vector< Mat_<uchar> >::const_iterator example = examples.begin(); example != examples.end(); ++example)
        integrateExample(*example, false);

}

bool FernClassifier::isObject(const Mat_<uchar> &image) const
{
    return getProbability(image) > threshold;
}

double FernClassifier::getProbability(const Mat_<uchar> &image) const
{
    float accumProbability = 0.;

    int fernsSize = int(ferns.size());

    for(int i = 0; i < fernsSize; ++i)
    {
        int position = code(image, ferns[i]);
        int posNum = precedents[i][position].x, negNum = precedents[i][position].y;

        if (posNum != 0 || negNum != 0)
            accumProbability += float(posNum) / (posNum + negNum);
        else
            fernsSize--;

        if(accumProbability / fernsSize > threshold)
            return 1.;

        if((accumProbability + (fernsSize - i)) / fernsSize <= threshold)
            return 0.;
    }

    CV_Assert(fernsSize);

    return accumProbability / fernsSize;
}

int FernClassifier::code(const Mat_<uchar> &image, const Ferns::value_type &fern) const
{
    int position = 0;

    CV_Assert(image.size() == patchSize);

    for(Ferns::value_type::const_iterator measureIt = fern.begin(); measureIt != fern.end(); ++measureIt)
    {
        position <<= 1;

//        if(image.at<uchar>(measureIt->first) < image.at<uchar>(measureIt->second))
//            position++;


        int val1 = 0, val2 = 0;
        for(int i = -1; i <= 1; ++i)
        {
            for(int j = -1; j <= 1; ++j)
            {
                if(j != 0 && j != 0)
                    continue;

                val1 += image.at<uchar>(measureIt->first.x + i, measureIt->first.y + j);
                val2 += image.at<uchar>(measureIt->second.x + i, measureIt->second.y + j);
            }
        }

        if(val1 < val2)
            position++;

    }
    return position;
}

void FernClassifier::integrateExample(const Mat_<uchar> &image, bool isPositive)
{

    //float accumProbability = 0.;
    //const int fernsSize = int(ferns.size());

    for(size_t i = 0; i < ferns.size(); ++i)
    {
        int position = code(/*blurred*/image, ferns[i]);

        if(isPositive)
            precedents[i][position].x++;
        else
            precedents[i][position].y++;

        //accumProbability += float(precedents[i][position].x) / (precedents[i][position].x + precedents[i][position].y);
    }

    /*if(accumProbability / fernsSize <= threshold && isPositive)
        std::cout << "Warning ferns are in bad state (positive)" << accumProbability / fernsSize << std::endl;

    if(accumProbability / fernsSize > threshold && !isPositive)
        std::cout << "Warning ferns are in bad state (negative)" << accumProbability / fernsSize << std::endl;*/
}

void FernClassifier::saveFern(const std::string &path) const
{
    return;
    FileStorage fernStorage(path, FileStorage::WRITE);

    for(size_t fernIndex = 0; fernIndex < precedents.size(); ++fernIndex)
    {
        std::stringstream ss; ss << "fern_" << fernIndex;
        fernStorage << ss.str() << precedents[fernIndex];
    }
}

void FernClassifier::compareFerns(const std::string &refFern, const std::string &testFern)
{
    FileStorage refFern1Storage(refFern, FileStorage::READ), testFern2Storage(testFern, FileStorage::READ);

    Precedents refPrecedents, testPrecedents;

    for(int i = 0;; ++i)
    {
        std::vector<Point> refPrecendent, testPrecendent;
        std::stringstream ss; ss << "fern_" << i;

        refFern1Storage[ss.str()] >> refPrecendent;
        testFern2Storage[ss.str()] >> testPrecendent;

        if(refPrecendent.empty() || testPrecendent.empty())
        {
            CV_Assert(refPrecendent.empty() && testPrecendent.empty());
            break;
        }

        CV_Assert(refPrecendent.size() == testPrecendent.size());

        refPrecedents.push_back(refPrecendent);
        testPrecedents.push_back(testPrecendent);

    }

    CV_Assert(refPrecedents.size() == testPrecedents.size());

    int diffPositive = 0, diffNegative = 0;
    int newDiffPositive = 0, newDiffNegative = 0;

    for(size_t outterIndex = 0; outterIndex < refPrecedents.size(); ++outterIndex)
    {
        for(size_t innerIndex = 0; innerIndex < refPrecedents[outterIndex].size(); ++innerIndex)
        {
            if(refPrecedents[outterIndex][innerIndex] != testPrecedents[outterIndex][innerIndex])
            {
                if((refPrecedents[outterIndex][innerIndex].x > refPrecedents[outterIndex][innerIndex].y && testPrecedents[outterIndex][innerIndex].x <= testPrecedents[outterIndex][innerIndex].y))
                {
                    std::cout << "negative " << outterIndex << " " << refPrecedents[outterIndex][innerIndex] << " " << testPrecedents[outterIndex][innerIndex] << std::endl;
                    diffNegative++;
                    if(refPrecedents[outterIndex][innerIndex] == Point())
                        newDiffNegative++;
                }
                if(refPrecedents[outterIndex][innerIndex].x <= refPrecedents[outterIndex][innerIndex].y && testPrecedents[outterIndex][innerIndex].x > testPrecedents[outterIndex][innerIndex].y)
                {
                    diffPositive++;
                    std::cout << "positive " << outterIndex << " " << refPrecedents[outterIndex][innerIndex] << " " << testPrecedents[outterIndex][innerIndex] << std::endl;
                    if(refPrecedents[outterIndex][innerIndex] == Point())
                        newDiffPositive++;
                }
            }
        }
    }


    std::cout << "diffPositive " << diffPositive << " " << " diffNegative " << diffNegative << std::endl;
    std::cout << "newDiffPositive " << newDiffPositive << " " << " newDiffNegative " << newDiffNegative << std::endl;
}

std::vector<Mat> FernClassifier::outputFerns(const Size &displaySize) const
{
    RNG rang;

    float scaleW = float(displaySize.width) / patchSize.width;
    float scaleH = float(displaySize.height) / patchSize.height;

    const Mat black(displaySize, CV_8UC3, Scalar::all(0));

    std::vector<Mat> fernsImages;
    fernsImages.reserve(ferns.size());

    for(Ferns::const_iterator fernIt = ferns.begin(); fernIt != ferns.end(); ++fernIt)
    {
        Mat copyBlack; black.copyTo(copyBlack);
        for(Ferns::value_type::const_iterator measureIt = fernIt->begin(); measureIt != fernIt->end(); ++measureIt)
        {
            Scalar color(rang.uniform(20,255), rang.uniform(20,255), rang.uniform(20,255));

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


double NNClassifier::dSminus, NNClassifier::dSplus;

NNClassifier::NNClassifier(size_t actMaxNumberOfExamples, Size actNormilizedPatchSize, double actTheta) :
    theta(actTheta), maxNumberOfExamples(actMaxNumberOfExamples),
    normilizedPatchSize(actNormilizedPatchSize), normilizedPatch(normilizedPatchSize)
{}

void NNClassifier::isObjects(const std::vector<Hypothesis> &hypothesis, std::map<double, Mat_<uchar> > &scaledStorage, std::vector<Answers> &answers) const
{
    static int hints;

    CV_Assert(hypothesis.size() == answers.size());

    for(size_t i = 0; i < hypothesis.size(); ++i)
    {
        if(answers[i])
        {

            const double scaleFactorX = double(normilizedPatchSize.width) / hypothesis[i].bb.width;
            const double scaleFactorY = double(normilizedPatchSize.height) / hypothesis[i].bb.height;

            CV_Assert(scaledStorage.find(hypothesis[i].scale) != scaledStorage.end());

            const Point newPoint(cvRound(hypothesis[i].bb.x * scaleFactorX), cvRound(hypothesis[i].bb.y * scaleFactorY));
            const Rect newRect(newPoint, normilizedPatchSize);

            answers[i] = isObject(scaledStorage[hypothesis[i].scale](newRect));

            if(answers[i])
                answers[i].confidence = calcConfidenceDetector(scaledStorage[hypothesis[i].scale](newRect));
        }
    }

    hints++;
}

void NNClassifier::integratePositiveExamples(const std::vector<Mat_<uchar> > &examples)
{
    for(std::vector< Mat_<uchar> >::const_iterator example = examples.begin(); example != examples.end(); ++example)
    //{
        //if(!isObject(*example))
            addExample(*example, positiveExamples);
    //}
}

void NNClassifier::integrateNegativeExamples(const std::vector<Mat_<uchar> > &examples)
{
    for(std::vector< Mat_<uchar> >::const_iterator example = examples.begin(); example != examples.end(); ++example)
        addExample(*example, negativeExamples);
}

bool NNClassifier::isObject(const Mat_<uchar> &image) const
{
    return Sr(image) > theta;
}

double NNClassifier::calcConfidenceTracker(const Mat_<uchar> &image) const
{
    if(image.size() != normilizedPatchSize)
        resize(image, normilizedPatch, normilizedPatchSize);
    else
        normilizedPatch = image;

    return Sc(normilizedPatch, true);
}

double NNClassifier::calcConfidenceDetector(const Mat_<uchar> &image) const
{
    return Sc(image);
}


double NNClassifier::Sr(const Mat_<uchar> &patch) const
{
    CV_Assert(patch.size() == normilizedPatchSize);

    double splus = 0., sminus = 0.;

    for(ExampleStorage::const_iterator it = positiveExamples.begin(); it != positiveExamples.end(); ++it)
        splus = std::max(splus, 0.5 * (NCC(*it, patch) + 1.0));

    for(ExampleStorage::const_iterator it = negativeExamples.begin(); it != negativeExamples.end(); ++it)
        sminus = std::max(sminus, 0.5 * (NCC(*it, patch) + 1.0));

    if (splus + sminus == 0.0)
        return 0.0;

    return splus / (sminus + splus);
}

double NNClassifier::Sc(const Mat_<uchar> &patch, bool isForTracker) const
{
    CV_Assert(patch.size() == normilizedPatchSize);

    double splus = 0., sminus = 0.;

    size_t mediana = positiveExamples.size() / 2 + positiveExamples.size() % 2;

    ExampleStorage::const_iterator end = positiveExamples.begin();
    for(size_t i = 0; i < mediana; ++i) ++end;

    for(ExampleStorage::const_iterator it = positiveExamples.begin(); it != end; ++it)
        splus = std::max(splus, 0.5 * (NCC(*it, patch) + 1.0));

    for(ExampleStorage::const_iterator it = negativeExamples.begin(); it != negativeExamples.end(); ++it)
        sminus = std::max(sminus, 0.5 * (NCC(*it, patch) + 1.0));

    if(!isForTracker)
        if(2 * splus - 1 < 0.75 && 2 * sminus - 1 < 0.75)
            return 0.;

    //std::cout << 2 * splus - 1 << " " <<  2 * sminus - 1  << std::endl;

    if (splus + sminus == 0.0)
        return 0.0;

    return splus / (sminus + splus);
}

void NNClassifier::addExample(const Mat_<uchar> &example, std::list<Mat_<uchar> > &storage)
{
    CV_Assert(storage.size() <= maxNumberOfExamples);
    CV_Assert(example.size() == normilizedPatchSize);

    if(storage.size() == maxNumberOfExamples)
    {
        int randomIndex = rng.uniform(0, int(maxNumberOfExamples));
        ExampleStorage::iterator it = storage.begin();
        for(int i = 0; i < randomIndex; ++i)
            ++it;

        storage.erase(it);
    }

    storage.push_back(example.clone());
}

float NNClassifier::NCC(const Mat_<uchar> &patch1, const Mat_<uchar> &patch2)
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

std::pair<Mat, Mat> NNClassifier::outputModel(int positiveMark, int negativeMark) const
{
    const int sqrtSize = cvRound(std::sqrt(std::max(positiveExamples.size(), negativeExamples.size())) + 0.5f);
    const int outputWidth = sqrtSize * normilizedPatchSize.width, outputHeight = sqrtSize * normilizedPatchSize.height;

    Mat positivePrecedents(outputHeight, outputWidth, CV_8U, Scalar::all(255)), negativePrecedents(outputHeight, outputWidth, CV_8U, Scalar::all(255));
    Mat positivePrecedent, negativePrecedent;

    cv::Rect actPositionPositive(cv::Point(), normilizedPatchSize);

    int currentPositiveExample = 0;
    for(ExampleStorage::const_iterator it = positiveExamples.begin(); it != positiveExamples.end(); ++it, ++currentPositiveExample)
    {
        if(actPositionPositive.x + it->cols > positivePrecedents.cols)
        {
            actPositionPositive.x = 0;
            actPositionPositive.y += it->rows;
        }

        CV_Assert(actPositionPositive.y + it->rows <= positivePrecedents.rows);

        it->copyTo(positivePrecedents(actPositionPositive));

        if(currentPositiveExample == positiveMark)
        {
            rectangle(positivePrecedents, actPositionPositive, Scalar::all(255));
            it->copyTo(positivePrecedent);
        }

        actPositionPositive.x += it->cols;

    }

    cv::Rect actPositionNegative(cv::Point(), normilizedPatchSize);

    int currentNegativeExample = 0;
    for(ExampleStorage::const_iterator it = negativeExamples.begin(); it != negativeExamples.end(); ++it, ++currentNegativeExample)
    {
        if(actPositionNegative.x + it->cols > negativePrecedents.cols)
        {
            actPositionNegative.x = 0;
            actPositionNegative.y += it->rows;
        }

        CV_Assert(actPositionNegative.y + it->rows <= negativePrecedents.rows);

        it->copyTo(negativePrecedents(actPositionNegative));

        if(currentNegativeExample == negativeMark)
        {
            rectangle(negativePrecedents, actPositionNegative, Scalar::all(255));
            it->copyTo(negativePrecedent);
        }

        actPositionNegative.x += it->cols;

    }

//    return std::make_pair(positivePrecedents, negativePrecedents);
    return std::make_pair(positivePrecedent, negativePrecedent);
}

std::pair<Mat, Mat> NNClassifier::getModelWDecisionMarks(const Mat_<uchar> &image, double previousConf)
{
    Mat internalNormilizedPatch;

    if(image.size() != normilizedPatchSize)
        resize(image, internalNormilizedPatch, normilizedPatchSize);
    else
        image.copyTo(internalNormilizedPatch);

    int positiveDesicionExample = -1, negativeDesicionExample = -1;

    imwrite("/tmp/req.png", internalNormilizedPatch);

    const double calcedConf = debugSr(internalNormilizedPatch, positiveDesicionExample, negativeDesicionExample);

    CV_Assert(previousConf == calcedConf);

    imwrite("/tmp/req.png", internalNormilizedPatch);

    return outputModel(positiveDesicionExample, negativeDesicionExample);

}

double NNClassifier::debugSr(const Mat_<uchar> &patch, int &positiveDecisitionExample, int &negativeDecisionExample)
{
    double splus = 0., sminus = 0.;

    double prevSplus = splus, prevSminus = sminus;


    size_t mediana = positiveExamples.size() / 2 + positiveExamples.size() % 2;

    ExampleStorage::const_iterator end = positiveExamples.begin();
    for(size_t i = 0; i < mediana; ++i) ++end;

    for(ExampleStorage::iterator it = positiveExamples.begin(); it != end; ++it)
    {
        splus = std::max(splus, 0.5 * (NCC(*it, patch) + 1.0));

        if(prevSplus != splus)
        {
            positiveDecisitionExample = std::distance(positiveExamples.begin(), it);
            prevSplus = splus;
        }
    }

    for(ExampleStorage::iterator it = negativeExamples.begin(); it != negativeExamples.end(); ++it)
    {
        sminus = std::max(sminus, 0.5 * (NCC(*it, patch) + 1.0));

        if(prevSminus != sminus)
        {
            negativeDecisionExample = std::distance(negativeExamples.begin(), it);
            prevSminus = sminus;
        }
    }

    if (splus + sminus == 0.0)
        return 0.0;

    return splus / (sminus + splus);
}

}
}
