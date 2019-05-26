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
class LedDetector
{
public:
    LedDetector(int lpSize = 120, // размер log polar картинки
                int clusters=6, // количество кластеров сегментации
                int ledClusters = 2, // количество кластеров, примерно относящихся к led
                int ledCount = 4 // число led на картинке, которое хотим обнаружить
            );
    /**
     * @brief predict - предсказать наличие на картинке прожекторов
     * @param src - ихожражение
     * @param centers - найденые центры прожекторов
     * @return 1 - да, 0 - нет
     */
    int predict(const Mat& src, vector<Point> centers);

    int cartRadius() const;

    Mat lp() const;

    int lpRadius() const;

    Mat lpKmean() const;

    std::vector<KeyPoint> lpKeypoints() const;

protected:
    int _fitLpRadius(const Mat& src);

    void _clear();

    void _detectLogPolarBlobs(const Mat& src, float thresh, std::vector<KeyPoint> &keypoints);

    Point2d _logPolarToCart(double ro, double phi, cv::Size cartSize, cv::Size polarSize);

protected:
    //log polar img
    Mat m_lp;
    //log polar kmean img
    Mat m_lpKmean;
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
double fitFeaturesProb(const Mat & src, vector<double> &results,int num_strips = 4);


