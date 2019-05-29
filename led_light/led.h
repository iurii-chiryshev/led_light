#pragma once
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <ccl.h>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace std;
using namespace cv;



/**
 * @brief The LedDetector class
 */
class Led
{
public:
    Led(int lpSize = 120, // размер log polar картинки
                int clusters=6, // количество кластеров сегментации
                int ledClusters = 1, // количество кластеров, примерно относящихся к led
                int ledCount = 4 // число led на картинке, которое хотим обнаружить
            );

    /**
     * @brief operator ()
     * @param src - входное изображение
     * @param vec - выходной вектор признаков
     */
    void operator() (const Mat& src, Mat &vec);

    static double compare(const Mat& model, Mat& target);

    static void estimate(const Mat& src,Mat &dst, int size, double &mean, double &std);

    int cartRadius() const;

    Mat lp() const;

    int lpRadius() const;

    Mat lpKmean() const;

    std::vector<KeyPoint> lpKeypoints() const;

    Mat lpShiftAligned() const;

    Mat lpBin() const;

protected:
    int _estimateLpRadius(const Mat& src);

    void _clear();

    void _detectLogPolarBlobs(const Mat& src, float thresh, std::vector<KeyPoint> &keypoints);

    Point2d _logPolarToCart(double ro, double phi, cv::Size cartSize, cv::Size polarSize);

    void _shiftAlign(const Mat& src, Mat& dst, int shiftX, int shiftY = 0);

protected:
    //log polar img
    Mat m_lp;
    // log polar с выравниванием к одному виду по сдвигу
    Mat m_lpShiftAligned;
    //log polar kmean img
    Mat m_lpKmean;
    //Бинарная картинка с прожекторами
    Mat m_lpBin;
    //
    Mat m_lpLabels;
    // предсказаный радиус на кототором лежат led в log polar координатах
    int m_lpRadius;
    // предсказаный радиус на кототором лежат led на исходной картинке
    int m_cartRadius;
    // ключевые точки найденные на log polar картинке
    std::vector<KeyPoint> m_lpKeypoints;


    int m_ledCount;
    // размер log polar картинки
    int m_lpSize;
    // количество класеров для скгменации
    int m_numClusters;
    // какое количество "ярких" кластеров относится к led
    int m_ledClusters;
};


/**
 * @brief correctGamma - гамма коррекция
 * @param src
 * @param dst
 * @param gamma
 */
void correctGamma(const Mat& src,Mat& dst, double gamma = 2.0);


/**
 * @brief kMeans
 * @param src
 * @param dst
 * @param clusters
 * @param k
 */
void kMeans(const Mat& src,Mat &dst, Mat &clusters,int k=3);

/**
 * @brief drawHist
 * @param src
 * @param dst
 * @param thickness
 * @param histPart
 */
void drawHist(const Mat & src, Mat &dst, int thickness = 1, double histPart = 0.2);

/**
 * @brief fitFeaturesProb
 * @param src
 * @param results
 * @param num_strips
 * @return
 */
void calculateFeatures(const Mat &src,
                       vector<double> &results,
                       const vector<int> &strip_counts,
                       const vector<int> &match_methods,
                       const vector<int> &hist_methods);


bool findLeds(const Mat &src,vector<Point> &leds);

