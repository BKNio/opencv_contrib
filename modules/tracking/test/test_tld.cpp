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
#include <fstream>
#include <iterator>

#include "test_precomp.hpp"

#include "../src/tldUtils.hpp"
#include "../src/tldDetector.hpp"
#include "../src/tldEnsembleClassifier.hpp"

namespace cv
{
    std::istream& operator >> (std::istream& in, cv::Rect& rect)
    {
        std::string str;
        std::getline(in, str);

        if(str.empty())
            return in;

        std::stringstream sstr(str);

        float dataToRead[4];

        for(size_t position = 0; position < sizeof dataToRead / sizeof dataToRead[0]; ++position)
        {
            std::string item;
            std::getline(sstr, item, ',');

            std::stringstream ss; ss << item;

            try
            {
                ss >> dataToRead[position];
            }
            catch(const std::logic_error &e)
            {
                std::cerr << "Error: " << e.what() << std::endl;
                std::exit(EXIT_FAILURE);
            }

        }


        rect = cv::Rect(cv::Point(dataToRead[0], dataToRead[1]), cv::Point(dataToRead[2], dataToRead[3]));

        return in;
    }
}

class ClassifiersTest: public cvtest::BaseTest
{
public:
    virtual ~ClassifiersTest(){}
    virtual void run();
    ClassifiersTest() : rng(/*std::time(NULL)*/), fontScaleRange(0.7f, 2.7f), angleRange(-15.f, 15.f), scaleRange(0.9f, 1.1f),
        shiftRange(-5, 5), thicknessRange(1, 3), minimalSize(20, 20), pathToTLDDataSet("/home/dinar/proj_src/test_data/TLD")
    {
//        testCases.push_back("01_david");
//        testCases.push_back("02_jumping");
//        testCases.push_back("03_pedestrian1");
//        testCases.push_back("04_pedestrian2");
//        testCases.push_back("05_pedestrian3");
//        testCases.push_back("06_car");
//        testCases.push_back("07_motocross");
//        testCases.push_back("08_volkswagen");
        testCases.push_back("09_carchase");
//        testCases.push_back("10_panda");
    }

private:
    bool nccRandomFill();
    bool nccRandomCharacters();
    bool emptyTest();
    bool simpleTest();
    bool syntheticDataTest();
    bool onlineTrainTest();
    bool realDataTest();


    bool pixelComprassionTest();
    bool scaleTest();

private:
    cv::RNG rng;
    const cv::Vec2f fontScaleRange, angleRange, scaleRange;
    const cv::Vec2i shiftRange, thicknessRange;
    const cv::Size minimalSize;
    const std::string pathToTLDDataSet;
    std::vector<std::string> testCases;

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


void ClassifiersTest::run()
{
    /*if(!nccRandomFill())
        FAIL() << "nccRandom test failed" << std::endl;

    if(!nccRandomCharacters())
        FAIL() << "nccCharacter test failed" << std::endl;

    if(!emptyTest())
        FAIL() << "empty test failed" << std::endl;

    if(!simpleTest())
        FAIL() << "simple test failed" << std::endl;

    if(!syntheticDataTest())
        FAIL() << "syntheticData test failed" << std::endl;

    if(!onlineTrainTest())
        FAIL() << "onlineTrain test failed" << std::endl;*/

    if(!realDataTest())
        FAIL() << "realData test failed" << std::endl;

//    if(!scaleTest())
//        FAIL() << "getPixelValue test failed" << std::endl;

//    if(!pixelComprassionTest())
//        FAIL() << "pixelComprassion test failed" << std::endl;

}

bool ClassifiersTest::nccRandomFill()
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

bool ClassifiersTest::nccRandomCharacters()
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

bool ClassifiersTest::emptyTest()
{
    bool result = true;
    for(int classifierId = 0; classifierId < 2; ++classifierId)
    {
        cv::Ptr<cv::tld::tldIClassifier> clasifier;
        if(classifierId == 0)
            clasifier = cv::makePtr<cv::tld::tldNNClassifier>();
        else
            clasifier = cv::makePtr<cv::tld::tldFernClassifier>(13, 50);

        const cv::Mat image(480, 640, CV_8U, cv::Scalar::all(0));

        std::vector<cv::tld::Hypothesis> hypothesis(3);
        hypothesis[0].bb = cv::Rect(cv::Point(0,0), cv::Size(50, 80));
        hypothesis[1].bb = cv::Rect(cv::Point(300,100), cv::Size(250, 180));
        hypothesis[2].bb = cv::Rect(cv::Point(453,74), cv::Size(33, 10));

        std::vector<bool> answers(3);
        answers[0] = true;
        answers[1] = false;
        answers[2] = true;

        clasifier->isObjects(hypothesis, image, answers);

        result &= !answers[0] && !answers[1] && !answers[2];
    }

    return result;
}

bool ClassifiersTest::simpleTest()
{
    const cv::Size exampleSize(240, 320);
    const cv::Size testImageSize(exampleSize.width, exampleSize.height * 2);

    cv::Mat image(testImageSize, CV_8U, cv::Scalar::all(0));

    cv::Mat positiveExample = image(cv::Rect(cv::Point(), exampleSize));
    cv::Mat negativeExample = image(cv::Rect(cv::Point(0, exampleSize.height), exampleSize));

    cv::circle(positiveExample, cv::Point(exampleSize.width / 2, exampleSize.height / 2), std::min(image.cols, image.rows) / 2, cv::Scalar::all(255));

    cv::rectangle(negativeExample, cv::Rect(exampleSize.width / 3, exampleSize.height / 4, exampleSize.width / 2, exampleSize.height / 2),
                  cv::Scalar::all(255));

    /*cv::imshow("simpletest test image", image);
    cv::waitKey(0);*/

    bool ret = true;
    for(int classifierIdi = 0; classifierIdi < 2; ++classifierIdi)
    {
        cv::Ptr<cv::tld::tldIClassifier> clasifier;

        if(classifierIdi == 0)
            clasifier = cv::makePtr<cv::tld::tldNNClassifier>();
        else
            clasifier = cv::makePtr<cv::tld::tldFernClassifier>(13, 50);

        clasifier->integratePositiveExample(positiveExample);
        clasifier->integrateNegativeExample(negativeExample);

        std::vector<cv::tld::Hypothesis> hypothesis(2);
        hypothesis[0].bb = cv::Rect(cv::Point(), exampleSize);
        hypothesis[1].bb = cv::Rect(cv::Point(0,exampleSize.height), exampleSize);

        std::vector<bool> answers(2);
        answers[0] = true;
        answers[1] = true;

        clasifier->isObjects(hypothesis, image, answers);

        ret &= answers[0] && !answers[1];
    }

    return ret;
}

bool ClassifiersTest::syntheticDataTest()
{
    const std::string positiveLetter = "A";
    const std::string negativeLetter = "Z";
    const int trainDataSize = 3000;

    std::vector<cv::Mat_<uchar> > trainDataPositive; trainDataPositive.reserve(trainDataSize);
    std::vector<cv::Mat_<uchar> > trainDataNegative; trainDataNegative.reserve(trainDataSize);

    for(int i = 0; i < trainDataSize; ++i)
    {
        cv::Mat_<uchar> trainExamplePositive;
        putRandomWarpedLetter(trainExamplePositive, positiveLetter);
        trainDataPositive.push_back(trainExamplePositive);
    }

    for(int i = 0; i < trainDataSize; ++i)
    {
        cv::Mat_<uchar> trainExampleNegative;
        putRandomWarpedLetter(trainExampleNegative, negativeLetter);
        trainDataNegative.push_back(trainExampleNegative);
    }

    const int numberOfTestExamples = 1500;
    cv::Mat_<uchar> tempPicture = cv::Mat(900, 1800, CV_8U);
    cv::Point currentPutPoint;
    int nextLineY = 0;

    std::vector<cv::Mat_<uchar> > scaledImages;
    std::vector<cv::tld::Hypothesis> hypothesis(numberOfTestExamples);
    std::vector<bool> gt(numberOfTestExamples);

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

        if(currentPutPoint.x + testExample.cols > tempPicture.cols)
        {
            currentPutPoint = cv::Point(0, currentPutPoint.y + nextLineY + 10);
            nextLineY = 0;
        }

        if(currentPutPoint.y + testExample.rows > tempPicture.rows)
        {
            scaledImages.push_back(tempPicture.clone());
            tempPicture = 0u;

            currentPutPoint = cv::Point();
            nextLineY = 0;
        }

        testExample.copyTo(tempPicture(cv::Rect(currentPutPoint, testExample.size())));

        hypothesis[i].bb = cv::Rect(currentPutPoint + cv::Point(0, tempPicture.rows * scaledImages.size()), testExample.size());
        gt[i] = isPositiveExmpl;

        currentPutPoint += cv::Point(testExample.cols + 10, 0);
        nextLineY = std::max(nextLineY, testExample.rows);

    }

    scaledImages.push_back(tempPicture.clone());

    const cv::Size hugePictureSize(tempPicture.cols, tempPicture.rows * scaledImages.size());
    const cv::Mat_<uchar> hugePicture(hugePictureSize, 0);

    cv::Rect currentPos(cv::Point(), tempPicture.size());
    for(std::vector< cv::Mat_<uchar> >::const_iterator scaledImage = scaledImages.begin(); scaledImage != scaledImages.end(); ++scaledImage)
    {
        scaledImage->copyTo(hugePicture(currentPos));
        currentPos.y += tempPicture.rows;
    }

    /*for(std::vector<cv::tld::Hypothesis>::const_iterator it = hypothesis.begin(); it != hypothesis.end(); ++it)
        cv::rectangle(hugePicture, it->bb, cv::Scalar::all(255));
    cv::imwrite("/tmp/zalupka.png", hugePicture);*/

    bool ret = true;
    for(int classifierId = 0; classifierId < 2; ++classifierId)
    {
        cv::Ptr<cv::tld::tldIClassifier> clasifier;
        std::string title;

        if(classifierId == 0)
        {
            clasifier = cv::makePtr<cv::tld::tldNNClassifier>();
            title = "NNClasifier";
        }
        else
        {
            clasifier = cv::makePtr<cv::tld::tldFernClassifier>(10, 100);
            title = "FernClassifier";
        }

        for(std::vector<cv::Mat_<uchar> >::const_iterator positiveExample = trainDataPositive.begin(); positiveExample != trainDataPositive.end(); ++positiveExample)
            clasifier->integratePositiveExample(*positiveExample);

        for(std::vector<cv::Mat_<uchar> >::const_iterator negativeExample = trainDataNegative.begin(); negativeExample != trainDataNegative.end(); ++negativeExample)
            clasifier->integrateNegativeExample(*negativeExample);


        std::vector<bool> test(numberOfTestExamples, true);
        clasifier->isObjects(hypothesis, hugePicture, test);

        float tP = 0.f, tN = 0.f, fP = 0.f, fN = 0.f;

        for(size_t i = 0; i < hypothesis.size(); ++i)
        {
            if(gt[i] != test[i])
            {

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
        const int numberOfPositives = std::count(gt.begin(), gt.end(), true);

        const float recall = tP / numberOfPositives;
        const float precission = tP / (tP + fP);

        std::cout << title + " recall " << recall << " precission " << precission << std::endl;

        ret &= recall > 0.95f && precission > 0.95f;

    }

    return ret;
}

bool ClassifiersTest::onlineTrainTest()
{
    const std::string positiveLetter = "Z";
    const std::string negativeLetter = "A";
    const int modelSize = 500;
    const int iterationsNumber = 10000;

    cv::Ptr<cv::tld::tldNNClassifier> nnclasifier = cv::makePtr<cv::tld::tldNNClassifier>(modelSize);
    cv::Ptr<cv::tld::tldFernClassifier> fernclassifier = cv::makePtr<cv::tld::tldFernClassifier>(10, 100);

    float correctClassifiedNN = 0.f, misclassifiedNN = 0.f;
    float correctClassifiedFern = 0.f, misclassifiedFern = 0.f;

    for(int iteration = 0; iteration < iterationsNumber; ++iteration)
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
        hypothesis[0].bb = cv::Rect(cv::Point(), example.size());

        std::vector<bool> answers(1);
        answers[0] = true;

        nnclasifier->isObjects(hypothesis, example, answers);

        if(answers[0] != isObject)
        {
            if(isObject)
                nnclasifier->integratePositiveExample(example);
            else
                nnclasifier->integrateNegativeExample(example);

            misclassifiedNN += 1.f;

            answers[0] = true;

            nnclasifier->isObjects(hypothesis, example, answers);

            if(answers[0] != isObject)
                return false;
        }
        else
            correctClassifiedNN += 1.f;

        answers[0] = true;
        fernclassifier->isObjects(hypothesis, example, answers);

        if(answers[0] != isObject)
        {
            if(isObject)
                fernclassifier->integratePositiveExample(example);
            else
                fernclassifier->integrateNegativeExample(example);

            misclassifiedFern += 1.f;

        }
        else
            correctClassifiedFern += 1.f;

    }

    /*std::cout <<"FernClassifier: correct classified "<< correctClassifiedFern / iterationsNumber << " misclassified " << misclassifiedFern / iterationsNumber << std::endl;
    std::cout <<"NNClasifier   : correct classified "<< correctClassifiedNN / iterationsNumber << " misclassified " << misclassifiedNN / iterationsNumber << std::endl;*/

    return correctClassifiedFern / iterationsNumber > 0.95 && correctClassifiedFern / iterationsNumber > 0.95;
}

//#define SHOW_BAD_ROI
//#define SHOW_MISCLASSIFIED
//#define SHOW_TRAIN_DATA
//#define SHOW_ADDITIONAL_EXAMPLES
#define SIMPLE_TESTS
bool ClassifiersTest::realDataTest()
{

    const int numberOfAdditionalExamples = 10;
    for(std::vector<std::string>::const_iterator testCase = testCases.begin(); testCase != testCases.end(); ++testCase)
    {
        const std::string path = pathToTLDDataSet + "/" + *testCase + "/";
        const std::string suffix = *testCase == "07_motocross" ? "%05d.png" : "%05d.jpg";

        cv::VideoCapture capture(path + suffix);

        if(!capture.isOpened())
            return std::cerr << "unable to open " + path + suffix, false;

        std::fstream gtData((path + "/gt.txt").c_str());
        if(!gtData.is_open())
            return std::cerr << "unable to open " + path + "/gt.txt", false;

        std::vector<cv::Rect> gtBB;
        std::copy(std::istream_iterator<cv::Rect>(gtData), std::istream_iterator<cv::Rect>(), std::back_inserter(gtBB));

        CV_Assert(!gtBB.empty());

        std::vector<cv::Mat> frames; frames.reserve(gtBB.size());
        cv::Mat frame;
        while(capture >> frame, !frame.empty())
        {
            cv::Mat grayFrame;
            cv::cvtColor(frame, grayFrame, CV_BGR2GRAY);
            frames.push_back(grayFrame);
        }

        CV_Assert(frames.size() == gtBB.size());

        const size_t trainSize = gtBB.size() * 0.75;
        const size_t testSize = gtBB.size() - trainSize;

        cv::Ptr<cv::tld::tldCascadeClassifier> cascadeClasifier = cv::makePtr<cv::tld::tldCascadeClassifier>(frames.front(), gtBB.front(), 300, 200, 7);
        const std::vector<cv::tld::Hypothesis> hypothesis = cascadeClasifier->generateHypothesis();

        for(size_t index = 0; index < frames.size() / 2; ++index)
        {
            const size_t randomIndex = rng.uniform(int(frames.size() / 2), frames.size());

            std::swap(frames[randomIndex], frames[index]);
            std::swap(gtBB[randomIndex], gtBB[index]);
        }

       float numberOfBadTrainExamples = 0.f, numberOfBadTestExamples = 0.f;
       float miscalssifiedNN = 0.f, miscalssifiedFern = 0.f;

       const cv::Rect roi(cv::Point(), frames.front().size());

       std::cerr << "training..." << std::endl;

       size_t dataSetIndex = 0;
       for(; dataSetIndex < trainSize; ++dataSetIndex)
       {
           const cv::Rect &bb = gtBB[dataSetIndex];

           if( bb.area() == 0 || (bb & roi).area() < bb.area())
           {
#ifdef SHOW_BAD_ROI
               cv::Mat badROI;
               cv::cvtColor(frames[dataSetIndex], badROI, CV_GRAY2BGR);
               cv::rectangle(badROI, bb, cv::Scalar(255, 0, 139));
               cv::imshow("badROI train", badROI);
               cv::waitKey();
#endif
               ++numberOfBadTrainExamples;
               continue;
           }

           cascadeClasifier->nnClassifier->integratePositiveExample(frames[dataSetIndex](bb));
           //cascadeClasifier->fernClassifier->integratePositiveExample(frames[dataSetIndex](bb));

           std::vector<cv::tld::Hypothesis> closestNPositive = getClosestN(hypothesis, bb, numberOfAdditionalExamples);

           for(std::vector<cv::tld::Hypothesis>::const_iterator closestPositive = closestNPositive.begin(); closestPositive != closestNPositive.end(); ++closestPositive)
           {
#ifdef SHOW_ADDITIONAL_EXAMPLES
                cv::Mat addPositiveExample; frames[dataSetIndex].copyTo(addPositiveExample);
                cv::rectangle(addPositiveExample, closestPositive->bb, cv::Scalar::all(255));
                cv::imshow("closest positive", addPositiveExample);
                cv::waitKey();
#endif
                cascadeClasifier->nnClassifier->integratePositiveExample(frames[dataSetIndex](closestPositive->bb));
                //cascadeClasifier->fernClassifier->integratePositiveExample(frames[dataSetIndex](closestPositive->bb));
           }

           std::vector<cv::tld::Hypothesis> closestNNegative = getClosestN(hypothesis, bb, numberOfAdditionalExamples, 0.01);

           for(std::vector<cv::tld::Hypothesis>::const_iterator closestNegative = closestNNegative.begin(); closestNegative != closestNNegative.end(); ++closestNegative)
           {
#ifdef SHOW_ADDITIONAL_EXAMPLES
                cv::Mat addNegativeExample; frames[dataSetIndex].copyTo(addNegativeExample);
                cv::rectangle(addNegativeExample, closestNegative->bb, cv::Scalar::all(255));
                cv::imshow("closes negative", addNegativeExample);
                cv::waitKey();
#endif
               cascadeClasifier->nnClassifier->integrateNegativeExample(frames[dataSetIndex](closestNegative->bb));
               //cascadeClasifier->fernClassifier->integrateNegativeExample(frames[dataSetIndex](closestNegative->bb));
           }

#ifdef SHOW_TRAIN_DATA
           cv::Mat trainExample;
           cv::cvtColor(frames[dataSetIndex], trainExample, CV_GRAY2BGR);
           cv::rectangle(trainExample, bb, cv::Scalar(255, 0, 139));
           cv::imshow("trainExample", trainExample);
           cv::waitKey();
#endif

       }

       std::cerr << "testing..." << std::endl;

       for(; dataSetIndex < frames.size(); ++dataSetIndex)
       {
           const cv::Rect &bb = gtBB[dataSetIndex];

#ifdef SIMPLE_TESTS
           if( bb.area() == 0 || (bb & roi).area() < bb.area())
           {
#ifdef SHOW_BAD_ROI
               cv::Mat badROI;
               cv::cvtColor(frames[dataSetIndex], badROI, CV_GRAY2BGR);
               cv::rectangle(badROI, bb, cv::Scalar(255, 0, 139));
               cv::imshow("badROI test", badROI);
               cv::waitKey();
#endif
               ++numberOfBadTestExamples;
               continue;
           }

           bool nnAnswer = cascadeClasifier->nnClassifier->isObject(frames[dataSetIndex](gtBB[dataSetIndex]));
           if(!nnAnswer)
           {
#ifdef SHOW_MISCLASSIFIED
               cv::Mat miscalssified; frames[dataSetIndex].copyTo(miscalssified);
               cv::rectangle(miscalssified, gtBB[dataSetIndex], cv::Scalar::all(255));
               cv::imshow("error nnclassifier", miscalssified);
               cv::waitKey();
#endif

               cv::Mat_<uchar> precedents =  cascadeClasifier->nnClassifier->outputPrecedents();
               cv::imshow("last precedents", precedents);
               cv::waitKey();

               ++miscalssifiedNN;
           }

           //bool fernAnswer = cascadeClasifier->fernClassifier->isObject(frames[dataSetIndex](gtBB[dataSetIndex]));
           //if(!fernAnswer)
           {
#ifdef SHOW_MISCLASSIFIED
               //std::cout << gtBB[dataSetIndex] << std::endl;
               cv::Mat miscalssified; frames[dataSetIndex].copyTo(miscalssified);
               cv::rectangle(miscalssified, gtBB[dataSetIndex], cv::Scalar::all(255));
               cv::imshow("error fernclassifier", miscalssified);
               cv::waitKey();
#endif
               ++miscalssifiedFern;
           }
       }
#else
           if(bb.area() < 225)
               continue;

           std::vector<bool> fernAnswers(hypothesis.size(), true);
           cascadeClasifier->fernClassifier->isObjects(hypothesis, frames[dataSetIndex], fernAnswers);

           std::vector<bool> nnAnswers(hypothesis.size(), true);
           cascadeClasifier->nnClassifier->isObjects(hypothesis, frames[dataSetIndex], nnAnswers);

           CV_Assert(fernAnswers.size() == nnAnswers.size());
           CV_Assert(hypothesis.size() == nnAnswers.size());

           double tPFern = 0., tNFern = 0., fPFern = 0., fNFern = 0.;
           double tPNN = 0., tNNN = 0., fPNN = 0., fNNN = 0.;
           for(size_t answerIndex = 0; answerIndex < fernAnswers.size(); answerIndex++)
           {
                bool gt = overlap(hypothesis[answerIndex].bb, bb) > 0.5;

                if(gt != nnAnswers[i])
                    if(gt)
                        fNNN++;
                    else
                        fPNN++;
                else
                    if(gt)
                        tPNN++;
                    else
                        tNNN++;


                if(gt != fernAnswers[i])
                    if(gt)
                        fNFern++;
                    else
                        fPFern++;
                else
                    if(gt)
                        tPFern++;
                    else
                        tNFern++;

           }

#endif
       std::cout << "-----" + *testCase + "-----" << std::endl;
       /*std::cout << "Bad train examples: " << std::setprecision(3) << numberOfBadTrainExamples / trainSize << std::endl;
       std::cout << "Bad test examples: " << std::setprecision(3) << numberOfBadTestExamples / testSize << std::endl;*/

       if(miscalssifiedNN)
           std::cout << "NN error " << std::setprecision(3) << miscalssifiedNN / (testSize - numberOfBadTestExamples) <<" abs value " << miscalssifiedNN << std::endl;

       if(miscalssifiedFern)
           std::cout << "Fern error " << std::setprecision(3) << miscalssifiedFern / (testSize - numberOfBadTestExamples) << " abs value " << miscalssifiedFern << std::endl;


    }
    return true;
}

bool ClassifiersTest::pixelComprassionTest()
{
//    const std::string letter = "$";
//    const cv::Size textSize = cv::getTextSize(letter, cv::FONT_HERSHEY_COMPLEX, 3.5, 3, NULL);
//    cv::Mat_<uchar> letterImage(textSize, 0);

//    cv::putText(letterImage, letter, cv::Point(0, textSize.height), cv::FONT_HERSHEY_COMPLEX, 3.5, cv::Scalar::all(255), 3);

//    std::vector<cv::Mat_<uchar> > scaledImages;

//#ifndef USE_BLUR
//    scaledImages.push_back(letterImage);
//#else
//    cv::Mat blurred;
//    cv::GaussianBlur(letterImage, blurred, cv::Size(3,3), 0);
//    scaledImages.push_back(blurred.clone());
//#endif

////    cv::imshow("letterImage", scaledImages.back());
////    cv::waitKey(0);

//    const float scales[] = {3.f, 2.f, 1.5f, 0.66f, 0.5f, 0.33f};
//    cv::Mat_<uchar> resized;
//    for(size_t scaleIndex = 0; scaleIndex < sizeof scales / sizeof (float); ++scaleIndex)
//    {
//        cv::Size resizedSize(letterImage.cols * scales[scaleIndex], letterImage.rows * scales[scaleIndex]);
//        cv::resize(letterImage, resized, resizedSize);

//#ifndef USE_BLUR
//        scaledImages.push_back(resized.clone());
//#else
//        cv::GaussianBlur(resized, blurred, cv::Size(3,3), 0);
//        scaledImages.push_back(blurred.clone());
//#endif

////        cv::imshow("resized", scaledImages.back());
////        cv::waitKey();
//    }


//    float numberErrors = 0.f;
//    float total = 0.f;

//    cv::Ptr<cv::tld::tldFernClassifier> fernClassifier = cv::makePtr<cv::tld::tldFernClassifier>(1);
//    for(cv::tld::tldFernClassifier::Ferns::const_iterator fern = fernClassifier->ferns.begin(); fern != fernClassifier->ferns.end(); ++fern)
//    {
//        fernClassifier->debugOutput = cv::Mat();
//        const int initialCode = fernClassifier->code(scaledImages.front(), *fern);
//        const std::pair<uchar, uchar> initialVals = fernClassifier->vals;
//        cv::Mat initialFern;
//        fernClassifier->debugOutput.copyTo(initialFern);

//        for(std::vector<cv::Mat_<uchar> >::iterator scaledImage = scaledImages.begin(); scaledImage != scaledImages.end(); ++scaledImage)
//        {
//            total++;
//            int currentCode = fernClassifier->code(*scaledImage, *fern);

//            if(currentCode != initialCode)
//            {
//                numberErrors++;
//                fernClassifier->debugOutput = cv::Mat();
//                fernClassifier->code(*scaledImage, *fern);

//                //std::stringstream ss; ss <<" "<< scales[std::distance(scaledImages.begin(), scaledImage)];
//                cv::imshow("not equal case"/* + ss.str()*/, fernClassifier->debugOutput);
//                cv::imshow("initialFern", initialFern);

//                std::cout << "initial vals " << unsigned(initialVals.first) << " " << unsigned(initialVals.second);
//                std::cout << " current vals " << unsigned(fernClassifier->vals.first) << " " << unsigned(fernClassifier->vals.second) << std::endl;

//                cv::waitKey();
//            }
//        }
//    }

//    std::cout <<"Number errors " << numberErrors / total  << " total " << total << " numberErrors " << numberErrors << std::endl;

    return true;
}


bool ClassifiersTest::scaleTest() //fern and nnc must be scale invariant
{
//    std::vector<std::string> positiveLetters;
//    positiveLetters.push_back("Z");
//    positiveLetters.push_back("`");
//    positiveLetters.push_back("W");
//    positiveLetters.push_back("R");
//    positiveLetters.push_back("X");
//    positiveLetters.push_back("@");
//    positiveLetters.push_back("D");
//    positiveLetters.push_back("*");
//    positiveLetters.push_back("O");
//    positiveLetters.push_back("A");

//    std::vector<std::string> negativeLetters;
//    negativeLetters.push_back("F");
//    negativeLetters.push_back("!");
//    negativeLetters.push_back("Q");
//    negativeLetters.push_back("C");
//    negativeLetters.push_back("E");
//    negativeLetters.push_back("2");
//    negativeLetters.push_back("B");
//    negativeLetters.push_back("&");
//    negativeLetters.push_back("V");
//    negativeLetters.push_back(";");

//    const float scales[] = {3.f, 2.f, 1.5f, 1., 0.66f, 0.5f, 0.33f};

//    cv::Ptr<cv::tld::tldFernClassifier> fernClassifier = cv::makePtr<cv::tld::tldFernClassifier>(13);
//    cv::Ptr<cv::tld::tldNNClassifier> nnclassifier = cv::makePtr<cv::tld::tldNNClassifier>(10);

//    std::vector<cv::Mat_<uchar> > positiveExamples;
//    std::vector<cv::Mat_<uchar> > negativeExamples;

//    for(int lettersIndex = 0; lettersIndex < 2; ++lettersIndex)
//    {
//        const std::vector<std::string> &currentLettersSet = lettersIndex ? negativeLetters : positiveLetters;

//        for(std::vector<std::string>::const_iterator it = currentLettersSet.begin(); it != currentLettersSet.end(); ++it)
//        {
//            const cv::Size textSize = cv::getTextSize(*it, cv::FONT_HERSHEY_COMPLEX, 3.5, 3, NULL);

//            cv::Mat_<uchar> image(int(textSize.height * 1.06), int(textSize.width * 1.06), uchar(0));
//            cv::putText(image, *it, cv::Point(0, textSize.height), cv::FONT_HERSHEY_COMPLEX, 3.5, cv::Scalar::all(255), 3);

//            //cv::imshow("image", image);
//            //cv::waitKey();

//            if(lettersIndex)
//            {
//                fernClassifier->integrateNegativeExample(image);
//                nnclassifier->integrateNegativeExample(image);
//            }
//            else
//            {
//                fernClassifier->integratePositiveExample(image);
//                nnclassifier->integratePositiveExample(image);
//            }

//            cv::Mat_<uchar> resized;
//            for(size_t scaleIndex = 0; scaleIndex < sizeof scales / sizeof (float); ++scaleIndex)
//            {
//                cv::Size resizedSize(image.cols * scales[scaleIndex], image.rows * scales[scaleIndex]);

//                cv::resize(image, resized, resizedSize);

//                if(lettersIndex)
//                    negativeExamples.push_back(resized);
//                else
//                    positiveExamples.push_back(resized);

//                /*cv::imshow("resized", resized);
//                cv::waitKey();*/
//            }

//        }

//    }


//    cv::Mat bigPicture = cv::Mat(900, 1800, CV_8U);

//    std::vector<cv::Mat_<uchar> > scaledImages;
//    std::vector<cv::tld::Hypothesis> hypothesis;
//    std::vector<bool> gt;

//    cv::Point currentPutPoint(0, 0);
//    int nextLineY = 0;

//    for(int setIndex = 0; setIndex < 2; ++setIndex)
//    {
//        const std::vector<cv::Mat_<uchar> > &currentExamples = setIndex ? negativeExamples : positiveExamples;

//        bool isPositive = setIndex == 0;

//        for(std::vector<cv::Mat_<uchar> >::const_iterator it = currentExamples.begin(); it != currentExamples.end(); ++it)
//        {

//            if(currentPutPoint.x + it->cols > bigPicture.cols)
//            {
//                currentPutPoint = cv::Point(0, currentPutPoint.y + nextLineY + 10);
//                nextLineY = 0;
//            }

//            if(currentPutPoint.y + it->rows > bigPicture.rows)
//            {
//                scaledImages.push_back(bigPicture.clone());
//                bigPicture = cv::Scalar::all(0);

//                currentPutPoint = cv::Point();
//                nextLineY = 0;
//            }

//            const cv::Rect bb = cv::Rect(currentPutPoint, it->size());
//            it->copyTo(bigPicture(bb));
//            /*cv::rectangle(bigPicture, bb, cv::Scalar::all(255));*/

//            hypothesis.push_back(cv::tld::Hypothesis());

//            hypothesis.back().bb = bb;
//            hypothesis.back().scaleId = scaledImages.size();

//            gt.push_back(isPositive);

//            currentPutPoint += cv::Point(it->cols + 10, 0);
//            nextLineY = std::max(nextLineY, it->rows);

//        }
//    }

//    scaledImages.push_back(bigPicture.clone());

//    std::vector<bool> fernAnswers(hypothesis.size(), true);
//    std::vector<bool> nncAnswers(hypothesis.size(), true);



//    /*for(std::vector<cv::Mat_<uchar> >::const_iterator it = scaledImages.begin(); it != scaledImages.end(); ++it)
//    {
//        cv::imshow("scaledImages", *it);
//        cv::waitKey();
//    }*/

//    fernClassifier->isObjects(hypothesis, scaledImages, fernAnswers);
//    nnclassifier->isObjects(hypothesis, scaledImages, nncAnswers);

//    CV_Assert(gt.size() == fernAnswers.size());
//    CV_Assert(fernAnswers.size() == nncAnswers.size());

//    const float numberOfPositives = std::count(gt.begin(), gt.end(), true);

//    for(int classifierId = 0; classifierId < 1; ++classifierId)
//    {
//        const std::vector<bool> &currentAnswers = classifierId ? nncAnswers : fernAnswers;
//        const std::string title = classifierId ? "nnc_error" : "fern_error";

//        float tP = 0.f, tN = 0.f, fP = 0.f, fN = 0.f;

//        for(size_t i = 0; i < fernAnswers.size(); ++i)
//        {
//            if(gt[i] != currentAnswers[i])
//            {
////                const cv::tld::Hypothesis &errorHypothesis = hypothesis[i];
////                const cv::Mat sourceImg = scaledImages[errorHypothesis.scaleId];
////                cv::imshow(title, sourceImg(errorHypothesis.bb));
////                cv::waitKey();

//                if(gt[i])
//                    fN += 1.f;
//                else
//                    fP += 1.f;

//            }
//            else
//                if(gt[i])
//                    tP += 1.f;
//                else
//                    tN += 1.f;

//        }

//        CV_Assert(tP + tN + fN + fP == hypothesis.size());
//        const float recall = tP / numberOfPositives;
//        const float precission = tP / (tP + fP);
//        std::cout << title + " recall " <<  recall << " precision " << precission << std::endl;
//        std::cout << "error present " << (fN + fP) / hypothesis.size() << std::endl;
//    }

    return true;

}

void ClassifiersTest::EuclideanTransform(cv::Vec2i shift, cv::Vec2f scale, float angle, const cv::Mat &src, cv::Mat &dst)
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

TEST(TLD, NNClassifier) { ClassifiersTest test; test.run(); }
