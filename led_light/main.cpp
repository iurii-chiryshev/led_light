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

using namespace std;
using namespace cv;

/**
 * @brief k_means - кластеризация  К средних
 * @param src - входное изображение
 * @param dst - выходное изображение
 * @param k - количество кластеров
 * @param centers - центры кластеров
 */
void k_means(const Mat& src,Mat &dst, Mat &centers,int k=3){
    CV_Assert(src.type() == CV_8UC1 && "src.type() == CV_8UC1"); // работаем только с серым
    Mat data = Mat::zeros(src.cols*src.rows, 1, CV_32F);
    //
    for(int i=0; i<src.cols*src.rows; i++) {
        data.at<float>(i,0) = (float)src.data[i];
    }
    Mat labels;
    //
    cv::kmeans(data, k, labels,
            TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1),
            3, KMEANS_PP_CENTERS,centers);
    //
    dst.create(src.rows,src.cols,src.type());
    for(int i=0; i<src.cols*src.rows; i++) {
        dst.at<uchar>(i/src.cols, i%src.cols) = (uchar)centers.at<float>(labels.at<int>(i,0),0);
    }
}


int main(int argc, char *argv[])
{
    const string path = "D://images//good_4.png";
    Mat src = cv::imread(path,IMREAD_GRAYSCALE);
    cv::imshow("original",src);
    Mat dst, centers;
    k_means(src,dst,centers,5);
    cv::imshow("k_mean",dst);

    while (!_kbhit())
    {
        cv::waitKey(10);
    }
    cv::destroyAllWindows();
    return 0;
}
