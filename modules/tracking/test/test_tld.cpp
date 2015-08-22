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

struct TLDNCC: public cvtest::BaseTest
{
  virtual ~TLDNCC(){}
  virtual void run();
};


void TLDNCC::run()
{

    cv::Ptr<cv::Tracker> tracker = cv::Tracker::create("TLD");
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

            if(std::abs(ncc - gt) < 5e-6)
            {
                FAIL() << "NCC correctness  test failed" << std::endl;
            }

        }
    }
}

TEST(TLD, NCC_test) { TLDNCC test; test.run(); }
