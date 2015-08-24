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

#include "test_precomp.hpp"

#include "../src/tldUtils.hpp"
#include "../src/tldEnsembleClassifier.hpp"

class NNClassifierTest: public cvtest::BaseTest
{
public:
    virtual ~NNClassifierTest(){}
    virtual void run();

private:
    bool nccTest();
    bool emptyTest();
    bool simpleTest();
    bool syntheticDataTest();
    bool realDataTest() { return true; }
};


void NNClassifierTest::run()
{
    if(!nccTest())
        FAIL() << "NCC correctness test failed" << std::endl;

    if(!emptyTest())
        FAIL() << "Empty test failed" << std::endl;

    if(!simpleTest())
        FAIL() << "Simple test failed" << std::endl;

    if(!syntheticDataTest())
        FAIL() << "syntheticDataTest test failed" << std::endl;
}

bool NNClassifierTest::nccTest()
{
    for(int n = 3; n < 31; n++)
    {
        cv::Mat img(n, n, CV_8U), templ(n, n, CV_8U);
        cv::RNG rng;
        cv::Mat result;
        for(int i = 0; i < 300; ++i)
        {
            rng.fill(img, cv::RNG::UNIFORM, 0, 255);
            rng.fill(templ, cv::RNG::UNIFORM, 0, 255);

            float ncc =  cv::tld::NCC(img, templ);

            cv::matchTemplate(img, templ, result, CV_TM_CCOEFF_NORMED);
            float gt = result.at<float>(0,0);

            if(std::abs(ncc - gt) > 5e-6)
                return false;

        }
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

//    cv::imshow("circle", positiveExmpl);
//    cv::imshow("rectan", negativeExmpl);

//    cv::waitKey();

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

bool NNClassifierTest::syntheticDataTest()
{

    const int modelSize = 100;
    cv::Ptr<cv::tld::tldNNClassifier> nnclasifier = cv::makePtr<cv::tld::tldNNClassifier>(modelSize);

    cv::RNG rng;

    for(int i = 0; i < modelSize; ++i)
    {
        int rows = rng.uniform(100, 400);
        int cols = rng.uniform(100, 400);
        cv::Mat positiveExmpl(rows, cols, CV_8U, cv::Scalar::all(0));

        cv::Size textSize = getTextSize("A", cv::FONT_HERSHEY_COMPLEX, 1., 1, 0);

        if(textSize.area() > positiveExmpl.size().area())
        {
            i--;
            continue;
        }

        cv::putText(positiveExmpl, "A", cv::Point(20,20), cv::FONT_HERSHEY_COMPLEX, 1., cv::Scalar::all(255), 1);

        cv::imshow("pos", positiveExmpl/*(cv::Rect(cv::Point(0,0), textSize))*/);
        cv::waitKey();

    }







//    cv::Mat negativeExmpl(320, 240, CV_8U, cv::Scalar::all(0));

//    cv::circle(positiveExmpl, cv::Point(positiveExmpl.cols / 2, positiveExmpl.rows / 2), std::min(positiveExmpl.cols, positiveExmpl.rows) / 2, cv::Scalar::all(255));
//    cv::rectangle(negativeExmpl, cv::Rect(negativeExmpl.cols / 3, negativeExmpl.rows / 4, negativeExmpl.cols / 2, negativeExmpl.rows / 2), cv::Scalar::all(255));

////    cv::imshow("circle", positiveExmpl);
////    cv::imshow("rectan", negativeExmpl);

////    cv::waitKey();

//    nnclasifier->addPositiveExample(positiveExmpl);
//    nnclasifier->addNegativeExample(negativeExmpl);

//    std::vector<cv::tld::Hypothesis> hypothesis(2);
//    hypothesis[0].bb = cv::Rect(cv::Point(0,0), positiveExmpl.size());
//    hypothesis[0].scaleId = 0;
//    hypothesis[1].bb = cv::Rect(cv::Point(0,0), negativeExmpl.size());
//    hypothesis[1].scaleId = 1;

//    std::vector<cv::Mat_<uchar> > scaledImages(2);
//    scaledImages[0] = positiveExmpl;
//    scaledImages[1] = negativeExmpl;

//    std::vector<bool> answers(2);
//    answers[0] = true;
//    answers[1] = true;

//    nnclasifier->isObjects(hypothesis, scaledImages, answers);

//    return answers[0] && !answers[1];

    return true;
}

TEST(TLD, NNClassifier) { NNClassifierTest test; test.run(); }

