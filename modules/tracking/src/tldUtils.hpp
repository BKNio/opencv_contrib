#ifndef OPENCV_TLD_UTILS
#define OPENCV_TLD_UTILS

#include "precomp.hpp"
#include "opencv2/highgui.hpp"
#include "tldEnsembleClassifier.hpp"

namespace cv
{
namespace tld
{
extern Rect2d etalon;

//const int STANDARD_PATCH_SIZE = 15;
//const int NEG_EXAMPLES_IN_INIT_MODEL = 300;
//const int MAX_EXAMPLES_IN_MODEL = 500;
//const int MEASURES_PER_CLASSIFIER = 13;
//const int GRIDSIZE = STANDARD_PATCH_SIZE;
//const int DOWNSCALE_MODE = cv::INTER_LINEAR;
//const double THETA_NN = 0.6;
//const double CORE_THRESHOLD = 0.5;
//const double SCALE_STEP = 1.2;
//const double ENSEMBLE_THRESHOLD = 0.5;
//const double VARIANCE_THRESHOLD = 0.5;
//const double NEXPERT_THRESHOLD = 0.2;
//const cv::Size GaussBlurKernelSize(3, 3);
//const Size standardPath = Size(15, 15);
//const Size minimalBBSize = Size(20,20);

void myassert(const Mat& img);
void printPatch(const Mat_<uchar>& standardPatch);
std::string type2str(const Mat& mat);
void drawWithRects(const Mat& img, std::vector<Rect2d>& blackOnes, Rect2d whiteOne = Rect2d(-1.0, -1.0, -1.0, -1.0));
void drawWithRects(const Mat& img, std::vector<Rect2d>& blackOnes, std::vector<Rect2d>& whiteOnes, String fileName = "");

//aux functions and variables
template<typename T> inline T CLIP(T x, T a, T b){ return std::min(std::max(x, a), b); }

/** Computes overlap between the two given rectangles. Overlap is computed as ratio of rectangles' intersection to that
        * of their union.*/
double CV_EXPORTS_W overlap(const Rect &r1, const Rect &r2);

/** Resamples the area surrounded by r2 in img so it matches the size of samples, where it is written.*/
void resample(const Mat& img, const RotatedRect& r2, Mat_<uchar>& samples);

/** Specialization of resample() for rectangles without retation for better performance and simplicity.*/
void resample(const Mat& img, const Rect2d& r2, Mat_<uchar>& samples);

/** Computes the variance of single given image.*/
double variance(const Mat& img);

/** Computes patch variance using integral images */
double variance(const Mat_<double>& intImgP, const Mat_<double>& intImgP2, Point pt, Size size);

typedef std::vector<std::pair<size_t, double> > Overlaps;
bool comparartor(Overlaps::value_type a, Overlaps::value_type b);

//std::vector<Hypothesis> CV_EXPORTS_W getClosestN(const std::vector<Hypothesis> &hypothesis, const Rect &bBox, size_t n, double maxOverlap = 1.);
//std::pair<std::vector<Rect>, std::vector<Rect> > CV_EXPORTS_W generateClosestN(const Rect &bBox, size_t N);

double scaleAndBlur(const Mat& originalImg, int scale, Mat& scaledImg, Mat& blurredImg, Size GaussBlurKernelSize, double scaleStep);

int getMedian(const std::vector<int>& values, int size = -1);

//void generateScanGridInternal(const Size &imageSize, const Size2d &bbSize, std::vector<Rect>& res);

Rect myGroupRectangles(std::vector<Rect> &rectList, double eps = 1);


///////////////////////////////////////////////////////////
std::pair<double, Rect2d> augmentedOverlap(const Rect2d rect, const Rect2d bb);
///////////////////////////////////////////////////////////

}
}

#endif
