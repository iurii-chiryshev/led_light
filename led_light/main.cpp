#include <iostream>
#include <windows.h>
#include <conio.h>
#include <time.h>
#define _USE_MATH_DEFINES // for C++
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ccl.h>

using namespace std;
using namespace cv;
using namespace ccl;

/**
 * @brief k_means - кластеризация  К средних
 * @param src - входное изображение
 * @param dst - выходное изображение
 * @param k - количество кластеров
 * @param centers - метки кластеров
 */
void k_means(const Mat& src,Mat &dst, Mat &labels,int k=3){
    CV_Assert(src.type() == CV_8UC1 && "src.type() == CV_8UC1"); // работаем только с серым
    Mat data = Mat::zeros(src.cols*src.rows, 1, CV_32F);
    //
    for(int i=0; i<src.cols*src.rows; i++) {
        data.at<float>(i,0) = (float)src.data[i];
    }
    Mat bestLabels,centers,clustered = Mat(src.rows, src.cols, CV_32F);
    //
    cv::kmeans(data, k, bestLabels,
            TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1),
            3, KMEANS_PP_CENTERS,centers);
    for(int i=0; i<src.cols*src.rows; i++) {
        clustered.at<float>(i/src.cols, i%src.cols) = centers.at<float>(bestLabels.at<int>(i,0),0);
    }
    clustered.convertTo(dst, CV_8U);
    centers.convertTo(labels, CV_8U);
}

/**
 * @brief connected_components
 * @param src
 * @param blobs
 */
void connected_components(const Mat & src,vector<Blob> &blobs){
    blobs.clear();
//    Mat mask;
//    cv::dilate(src, mask, cv::Mat());
//    cv::erode(mask, mask, cv::Mat());
    ccl::Blobs blobs_ptr;
    Mat labels;
    ccl::connectedComponents(src,labels,blobs_ptr,8,CCL_SAUF,50);
    for(int i = 0; i < blobs_ptr.size(); i++){
        if (blobs_ptr[i] == NULL) continue; // todo check NULL
        const ccl::Blob &b = *blobs_ptr[i];
        blobs.push_back(b);
    }
    ccl::freeBlobs(blobs_ptr);
}


int main(int argc, char *argv[])
{
    const string path = "D://images//good_2.png";
    // картинку
    Mat img = cv::imread(path,IMREAD_GRAYSCALE);
    cv::imshow("original",img);
    Mat k_mean, labels;
    const int num_cluster = 6; // количество кластеров (цветов) на которое квантуем картику
    // кластеризация картинки
    k_means(img,k_mean,labels,num_cluster);
    cv::imshow("k_mean",k_mean);
    // сортируем метки кластеров
    // предполагаем, что:
    // 1. самая темная метка - это фон,
    // 2. самая светлая метка - это светодиоды
    // 3. м/у ними - это ладонь
    cv::sort(labels,labels,CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);
    // смотрим на консоли
    for(int i = 0; i <labels.rows; i++ ){
        std::cout << (int)labels.at<uchar>(i,0) << endl;
    }
    int led_labels = 1; // сколько ярких кластеров считать диодами
    int led_thresh = (int)labels.at<uchar>(labels.rows - led_labels,0);
    Mat led;
    // получаем бинарную картинку светодиодов
    cv::threshold(k_mean, led, led_thresh - 1, 255, CV_THRESH_BINARY);
    cv::imshow("leds",led);

    // маркировка связных компонент
    vector<ccl::Blob> blobs;
    connected_components(led,blobs);
    Mat bbox;
    led.convertTo(bbox,CV_8UC3);
    cv::cvtColor(img, bbox, CV_GRAY2BGR);
    cv::Point centre(img.cols / 2, img.rows / 2);
    for(int i = 0; i < blobs.size();i++){
        cv::Scalar color(0,255,0);
        // bbox
        cv::rectangle(bbox,blobs[i].rect,color,2);
        // centroid
        cv::circle(bbox,blobs[i].centroid,2,color,2);
        // line to centre
        cv::line(bbox,blobs[i].centroid,centre,color,2);
    }
    cv::imshow("bbox",bbox);




    while (!_kbhit())
    {
        cv::waitKey(10);
    }
    cv::destroyAllWindows();
    return 0;
}
