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

#include <ctime>

#include "test_precomp.hpp"

#include "../src/tldUtils.hpp"
#include "../src/tldEnsembleClassifier.hpp"

class NNClassifierTest: public cvtest::BaseTest
{
public:
    virtual ~NNClassifierTest(){}
    virtual void run();
    NNClassifierTest() : rng(std::time(NULL)), fontScaleRange(0.7f, 2.7f), angleRange(-15.f, 15.f), scaleRange(0.9f, 1.1f),
        shiftRange(-5, 5), thicknessRange(1, 3), minimalSize(20, 20)
    {}

private:
    bool nccRandom();
    bool nccCharacters();
    bool emptyTest();
    bool simpleTest();
    bool syntheticDataTest();
    bool onlineTrainTest();

private:
    cv::RNG rng;
    const cv::Vec2f fontScaleRange, angleRange, scaleRange;
    const cv::Vec2i shiftRange, thicknessRange;
    const cv::Size minimalSize;

private:

    void EuclideanTransform(cv::Vec2i shift, cv::Vec2f scale, float angle, const cv::Mat &src, cv::Mat &dst);

    template<class _Tp> cv::Vec<_Tp, 2> getRandomValues(const cv::Vec<_Tp, 2> &range)
    {
        cv::Vec<_Tp, 2> value;

        value[0] = rng.uniform(range[0], range[1]);
        value[1] = rng.uniform(range[0], range[1]);

        return value;
    }

    void putRandomWarpedLetter(cv::Mat_<uchar> &dst, const std::string &letter)
    {
        cv::Mat notWarped, stddev;

        for(;;)
        {
            const int font = rng.uniform(cv::FONT_HERSHEY_SIMPLEX, cv::FONT_HERSHEY_SCRIPT_COMPLEX);
            const double fontScale = rng.uniform(fontScaleRange[0], fontScaleRange[1]);
            const int thickness = rng.uniform(thicknessRange[0], thicknessRange[1]);
            const float angle = rng.uniform(angleRange[0], angleRange[1]);

            const cv::Vec2i shift = getRandomValues(shiftRange);
            const cv::Vec2f scale = getRandomValues(scaleRange);

            cv::Size textSize = cv::getTextSize(letter, font, fontScale, thickness, 0);

            if(textSize.height < minimalSize.height || textSize.width < minimalSize.width)
                continue;

            if(dst.empty())
                dst = cv::Mat_<uchar>(textSize, uchar(0)), notWarped = cv::Mat_<uchar>(textSize, uchar(0));
            else
                dst.copyTo(notWarped);

            cv::putText(notWarped, letter, cv::Point(0, textSize.height), font, fontScale, cv::Scalar::all(255), thickness);

            EuclideanTransform(shift, scale, angle, notWarped, dst);

            cv::meanStdDev(dst, cv::noArray(), stddev);

            CV_Assert(stddev.depth() == CV_64F);

            if(stddev.at<double>(0) > 1e-4)
                break;
        }
    }

};


void NNClassifierTest::run()
{
    if(!nccRandom())
        FAIL() << "NCC random test failed" << std::endl;

    if(!nccCharacters())
        FAIL() << "NCC character test failed" << std::endl;

    if(!emptyTest())
        FAIL() << "Empty test failed" << std::endl;

    if(!simpleTest())
        FAIL() << "Simple test failed" << std::endl;

    if(!syntheticDataTest())
        FAIL() << "SyntheticData test failed" << std::endl;

    if(!onlineTrainTest())
        FAIL() << "OnlineTrain test failed" << std::endl;

}

bool NNClassifierTest::nccRandom()
{
    for(int n = 3; n < 31; n++)
    {
        cv::Mat img(n, n, CV_8U), templ(n, n, CV_8U);
        cv::Mat result;
        for(int i = 0; i < 300; ++i)
        {
            rng.fill(img, cv::RNG::UNIFORM, 0, 255);
            rng.fill(templ, cv::RNG::UNIFORM, 0, 255);

            float ncc =  cv::tld::tldNNClassifier::NCC(img, templ);

            cv::matchTemplate(img, templ, result, CV_TM_CCOEFF_NORMED);
            float gt = result.at<float>(0,0);

            if(std::abs(ncc - gt) > 5e-6)
                return false;

        }
    }

    return true;
}

bool NNClassifierTest::nccCharacters()
{

    const int sizeOfTest = 2500;
    const int numberOfLetters = 11;
    const std::string letters [] = {"Z", "`", "W", "R", "X", "@", "D", "*", "E", "A", "#"};


    for(int testCase = 0; testCase < sizeOfTest; ++testCase)
    {

        const std::string letter1 = letters[rng.uniform(0, numberOfLetters)];
        const std::string letter2 = letters[rng.uniform(0, numberOfLetters)];

        int rows = rng.uniform(15, 150);
        int cols = rng.uniform(15, 150);

        cv::Mat_<uchar> patch1(rows, cols, uchar(0)), patch2(rows, cols, uchar(0));

        putRandomWarpedLetter(patch1, letter1);
        putRandomWarpedLetter(patch2, letter2);

        float tldNcc = cv::tld::tldNNClassifier::NCC(patch1, patch2);
        CV_Assert(!cvIsNaN(tldNcc));

        cv::Mat result;
        cv::matchTemplate(patch1, patch2, result, CV_TM_CCOEFF_NORMED);
        float ocvNcc = result.at<float>(0,0);

        if(std::abs(ocvNcc - tldNcc) > 5e-6)
            return false;

    }

    return true;

}

bool NNClassifierTest::emptyTest()
{
    cv::Ptr<cv::tld::tldNNClassifier> nnclasifier = cv::makePtr<cv::tld::tldNNClassifier>(100);

    const cv::Mat image(480, 640, CV_8U, cv::Scalar::all(0));
    std::vector<cv::Mat_<uchar> > scaledImages(1, image);

    std::vector<cv::tld::Hypothesis> hypothesis(3);

    hypothesis[0].bb = cv::Rect(cv::Point(0,0), cv::Size(50, 80));
    hypothesis[0].scaleId = 0;

    hypothesis[1].bb = cv::Rect(cv::Point(300,100), cv::Size(250, 180));
    hypothesis[1].scaleId = 0;

    hypothesis[2].bb = cv::Rect(cv::Point(453,74), cv::Size(33, 10));
    hypothesis[2].scaleId = 0;

    std::vector<bool> answers(3);

    answers[0] = true;
    answers[1] = false;
    answers[2] = true;

    nnclasifier->isObjects(hypothesis, scaledImages, answers);

    return !answers[0] && !answers[1] && !answers[2];
}

bool NNClassifierTest::simpleTest()
{
    cv::Ptr<cv::tld::tldNNClassifier> nnclasifier = cv::makePtr<cv::tld::tldNNClassifier>(100);

    cv::Mat positiveExmpl(320, 240, CV_8U, cv::Scalar::all(0)), negativeExmpl(320, 240, CV_8U, cv::Scalar::all(0));

    cv::circle(positiveExmpl, cv::Point(positiveExmpl.cols / 2, positiveExmpl.rows / 2), std::min(positiveExmpl.cols, positiveExmpl.rows) / 2, cv::Scalar::all(255));
    cv::rectangle(negativeExmpl, cv::Rect(negativeExmpl.cols / 3, negativeExmpl.rows / 4, negativeExmpl.cols / 2, negativeExmpl.rows / 2), cv::Scalar::all(255));

    nnclasifier->addPositiveExample(positiveExmpl);
    nnclasifier->addNegativeExample(negativeExmpl);

    std::vector<cv::tld::Hypothesis> hypothesis(2);
    hypothesis[0].bb = cv::Rect(cv::Point(0,0), positiveExmpl.size());
    hypothesis[0].scaleId = 0;
    hypothesis[1].bb = cv::Rect(cv::Point(0,0), negativeExmpl.size());
    hypothesis[1].scaleId = 1;

    std::vector<cv::Mat_<uchar> > scaledImages(2);
    scaledImages[0] = positiveExmpl;
    scaledImages[1] = negativeExmpl;

    std::vector<bool> answers(2);
    answers[0] = true;
    answers[1] = true;

    nnclasifier->isObjects(hypothesis, scaledImages, answers);

    return answers[0] && !answers[1];

    return true;
}

//#define SHOW_BAD_CACES

bool NNClassifierTest::syntheticDataTest()
{
    const std::string positiveLetter = "A";
    const std::string negativeLetter = "Z";
    const int modelSize = 500;

    const cv::Size pathSize = cv::Size(20, 20);

    cv::Ptr<cv::tld::tldNNClassifier> nnclasifier = cv::makePtr<cv::tld::tldNNClassifier>(modelSize, pathSize);
    for(int i = 0; i < modelSize; ++i)
    {
        cv::Mat_<uchar> positiveExample;
        putRandomWarpedLetter(positiveExample, positiveLetter);
        nnclasifier->addPositiveExample(positiveExample);
    }

    for(int i = 0; i < modelSize; ++i)
    {
        cv::Mat_<uchar> negativeExample;
        putRandomWarpedLetter(negativeExample, negativeLetter);
        nnclasifier->addNegativeExample(negativeExample);
    }

    const int numberOfTestExamples = 2500;
    cv::Mat bigPicture = cv::Mat(900, 1800, CV_8U);
    cv::Point cuurentPutPoint(0, 0);
    int nextLineY = 0;

    std::vector<cv::Mat_<uchar> > scaledImages;
    std::vector<cv::tld::Hypothesis> hypothesis(numberOfTestExamples);
    std::vector<bool> gt(numberOfTestExamples);
    std::vector<bool> test(numberOfTestExamples, true);

    for(int i = 0; i < numberOfTestExamples; ++i)
    {
        bool isPositiveExmpl = true;

        std::string actLetter;
        if(rng.uniform(0,2))
            actLetter = positiveLetter;
        else
            actLetter = negativeLetter, isPositiveExmpl = false;

        cv::Mat_<uchar> testExample;
        putRandomWarpedLetter(testExample, actLetter);

        if(cuurentPutPoint.x + testExample.cols > bigPicture.cols)
        {
            cuurentPutPoint = cv::Point(0, cuurentPutPoint.y + nextLineY + 10);
            nextLineY = 0;
        }

        if(cuurentPutPoint.y + testExample.rows > bigPicture.rows)
        {
            scaledImages.push_back(bigPicture.clone());
            bigPicture = cv::Scalar::all(0);

            cuurentPutPoint = cv::Point();
            nextLineY = 0;
        }

        const cv::Rect bb = cv::Rect(cuurentPutPoint, testExample.size());
        testExample.copyTo(bigPicture(bb));

        hypothesis[i].bb = bb;
        hypothesis[i].scaleId = scaledImages.size();
        gt[i] = isPositiveExmpl;

        cuurentPutPoint += cv::Point(testExample.cols + 10, 0);
        nextLineY = std::max(nextLineY, testExample.rows);

    }

    scaledImages.push_back(bigPicture.clone());

    nnclasifier->isObjects(hypothesis, scaledImages, test);

    float tP = 0.f, tN = 0.f, fP = 0.f, fN = 0.f;

#ifdef SHOW_BAD_CACES
    std::vector<cv::Mat> errorImages(scaledImages.size());
#endif

    for(size_t i = 0; i < hypothesis.size(); ++i)
    {
        if(gt[i] != test[i])
        {

#ifdef SHOW_BAD_CACES

            {
                std::pair<cv::Mat, cv::Mat> model = nnclasifier->outputNearestPrecedents(i);
                cv::Mat copy = scaledImages[hypothesis[i].scaleId](hypothesis[i].bb);//(400, 400, CV_8U, cv::Scalar::all(0));
                //scaledImages[hypothesis[i].scaleId](hypothesis[i].bb).copyTo(copy(cv::Rect(cv::Point(), hypothesis[i].bb.size()
                cv::Mat resizedCopy; cv::resize(copy, resizedCopy, pathSize, cv::INTER_NEAREST);
                cv::Mat resultPositive;
                cv::matchTemplate(resizedCopy, model.first, resultPositive, CV_TM_CCOEFF_NORMED);

                cv::Mat resultNegative;
                cv::matchTemplate(resizedCopy, model.second, resultNegative, CV_TM_CCOEFF_NORMED);

                CV_Assert(resultNegative.size().area() == 1);
                CV_Assert(resultPositive.size().area() == 1);
                CV_Assert(resultNegative.depth() == CV_32F && resultPositive.depth() == CV_32F);

                const float actSplus = 0.5f * (resultPositive.at<float>(0) + 1.f);
                const float actSminus = 0.5f * (resultNegative.at<float>(0) + 1.f);
                const std::pair<float, float> distancesToNearestPrecedents = nnclasifier->getDistancesToNearestPrecedents(i);

                /*std::cout << 0.5f * (resultPositive.at<float>(0) + 1.f) << " " << 0.5f * (resultNegative.at<float>(0) + 1.f) << std::endl;

                cv::imshow("resizedCopy", resizedCopy);
                cv::imshow("Nearest positive example", model.first);
                cv::imshow("Nearest negative example", model.second);
                std::cout <<"gt "<< gt[i] <<" test "<< test[i] << std::endl;

                cv::Mat absDiffPositive, absDiffNegative;
                cv::absdiff(resizedCopy, model.first, absDiffPositive);
                cv::absdiff(resizedCopy, model.second, absDiffNegative);

                std::cout << cv::sum(absDiffPositive)[0] << " " << cv::sum(absDiffNegative)[0] << std::endl;

                cv::imshow("absDiffPositive", absDiffPositive);
                cv::imshow("absDiffNegative", absDiffNegative);

                cv::waitKey();*/

                CV_Assert(std::abs(actSplus - distancesToNearestPrecedents.first) < 5e-5 || std::abs(actSminus - distancesToNearestPrecedents.second) < 5e-5);
            }

            /*const cv::tld::Hypothesis &errorHypothesis = hypothesis[i];

            if(errorImages[errorHypothesis.scaleId].empty())
                cv::cvtColor(scaledImages[errorHypothesis.scaleId], errorImages[errorHypothesis.scaleId], cv::COLOR_GRAY2BGR);

            if(gt[i])
            {
                cv::putText(errorImages[errorHypothesis.scaleId], "FN", errorHypothesis.bb.br(), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(128,0,255), 2);
                cv::rectangle(errorImages[errorHypothesis.scaleId], errorHypothesis.bb, cv::Scalar(128,0,255), 1);
            }
            else
            {
                cv::putText(errorImages[errorHypothesis.scaleId], "FP", errorHypothesis.bb.br(), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0,128,255), 2);
                cv::rectangle(errorImages[errorHypothesis.scaleId], errorHypothesis.bb, cv::Scalar(0,128,255), 1);
            }*/
#endif

            if(!gt[i] && test[i])
                fP += 1.f;
            else
                fN += 1.f;

        }
        else
            if(gt[i])
                tP += 1.f;
            else
                tN += 1.f;
    }

    CV_Assert(tP + tN + fN + fP == hypothesis.size());

    int numberOfPositives = std::count(gt.begin(), gt.end(), true);

    std::cout << "recall " << tP / numberOfPositives << " precision " << tP / (tP + fP) << std::endl;
    std::cout << "total misclassified " << fP + fN << " misclassified percent " << (fP + fN) / numberOfTestExamples << std::endl;

#ifdef SHOW_BAD_CACES
    for(std::vector<cv::Mat>::const_iterator it = errorImages.begin(); it != errorImages.end(); ++it)
    {
        if(!it->empty())
        {
            cv::imshow("error img", *it);
            cv::waitKey();
        }
    }
#endif

    return true;
}

bool NNClassifierTest::onlineTrainTest()
{
    const std::string positiveLetter = "A";
    const std::string negativeLetter = "Z";
    const int modelSize = 500;
    const int iterations = 10000;

    const cv::Size pathSize = cv::Size(20, 20);

    cv::Ptr<cv::tld::tldNNClassifier> nnclasifier = cv::makePtr<cv::tld::tldNNClassifier>(modelSize, pathSize);

    float correctClassified = 0.f, misclassified = 0.f;

    for(int iteration = 0; iteration < iterations; ++iteration)
    {
        std::string letter;
        bool isObject = true;
        if(rng.uniform(0,2))
            letter = positiveLetter;
        else
            letter = negativeLetter, isObject = false;

        cv::Mat_<uchar> example;
        putRandomWarpedLetter(example, letter);

        std::vector<cv::tld::Hypothesis> hypothesis(1);
        hypothesis[0].scaleId = 0;
        hypothesis[0].bb = cv::Rect(cv::Point(), example.size());

        std::vector<cv::Mat_<uchar> > scaledImgs(1);
        scaledImgs[0] =  example;

        std::vector<bool> answers(1);
        answers[0] = true;

        nnclasifier->isObjects(hypothesis, scaledImgs, answers);

        if(answers[0] != isObject)
        {
            if(isObject)
                nnclasifier->addPositiveExample(example);
            else
                nnclasifier->addNegativeExample(example);

            misclassified += 1.f;

            answers[0] = true;

            nnclasifier->isObjects(hypothesis, scaledImgs, answers);

            CV_Assert(answers[0] == isObject);
        }
        else
            correctClassified += 1.f;


        if(iteration % 100 == 0)
            std::cout << "correctClassified percent " << correctClassified / (iteration + 1) << " misclassified percent " << misclassified / (iteration + 1) << std::endl;

    }

    return true;

}

void NNClassifierTest::EuclideanTransform(cv::Vec2i shift, cv::Vec2f scale, float angle, const cv::Mat &src, cv::Mat &dst)
{
    cv::Mat shiftTransform = cv::Mat::eye(3, 3, CV_32F);
    shiftTransform.at<float>(0,2) = shift[0];
    shiftTransform.at<float>(1,2) = shift[1];

    cv::Mat scaleTransform = cv::Mat::eye(3, 3, CV_32F);
    scaleTransform.at<float>(0,0) = scale[0];
    scaleTransform.at<float>(1,1) = scale[1];

    cv::Mat rotationShiftTransform = cv::Mat::eye(3, 3, CV_32F);
    rotationShiftTransform.at<float>(0,2) = -src.cols / 2;
    rotationShiftTransform.at<float>(1,2) = -src.rows / 2;

    cv::Mat rotationTransform = cv::Mat::eye(3, 3, CV_32F);
    rotationTransform.at<float>(0,0) = rotationTransform.at<float>(1,1) = std::cos(angle * CV_PI / 180);
    rotationTransform.at<float>(0,1) = std::sin(angle * CV_PI / 180);
    rotationTransform.at<float>(1,0) = - rotationTransform.at<float>(0,1);

    const cv::Mat resultTransform = rotationShiftTransform.inv() * rotationTransform * rotationShiftTransform * scaleTransform * shiftTransform;

    cv::warpAffine(src, dst, resultTransform(cv::Rect(0,0,3,2)), dst.size());

}

TEST(TLD, NNClassifier) { NNClassifierTest test; test.run(); }

