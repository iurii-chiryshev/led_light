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
#include <opencv2/contrib/contrib.hpp>
#include <ccl.h>
#include <led.h>

using namespace std;
using namespace cv;
using namespace ccl;

int main(int argc, char *argv[])
{
    const string path = "D://images//";
    vector<pair<string,bool>> imgs = {
        {"good_0.png",true},
        {"good_1.png",true},
        {"good_2.png",true},
        {"good_3.png",true},
        {"good_4.png",true},
        {"good_5.png",true},
        {"bad.png",false},
    };

    for (int i = 0; i < imgs.size(); i++){
        LedDetector detector(240);
        const string &name = imgs[i].first;
        string fullName = path + name;
        Mat gray = cv::imread(fullName,IMREAD_GRAYSCALE);
        if(gray.empty()) continue;
        vector<Point> centers;
        float value = detector.predict(gray,centers);
        cout << name << " predicted: " << value << ", expected: " << imgs[i].second << endl;

        // рисуем картинки

        // original
        Mat rgb;
        drawHist(gray,rgb);
        cv::imshow(name,rgb);
        // log polar
        Mat lp;
        drawHist(detector.lp(),lp);
        int lpRad = detector.lpRadius(); // log polar
        cv::line(lp,Point(lpRad,0),Point(lpRad,lp.rows),cv::Scalar(255,0,0));
        drawKeypoints( lp, detector.lpKeypoints(), lp,
                       Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
        cv::imshow(name + " lp",lp);
        // kmean log polar
        Mat lpKmean;
        drawHist(detector.lpKmean(),lpKmean);
        cv::imshow(name + " lp kmean",lpKmean);

        cv::waitKey(1);
    }



//    // картинку
//    Mat img = cv::imread(path+ "good_5.png",IMREAD_GRAYSCALE);
//    if(img.empty()) return -1;
//    Mat img_hist;
//    drawHist(img,img_hist);
//    cv::imshow("original",img_hist);
//    Mat k_mean, labels;
//    const int num_cluster = 6; // количество кластеров (цветов) на которое квантуем картику
//    // кластеризация картинки
//    kMeans(img,k_mean,labels,num_cluster);
//    cv::imshow("k_mean",k_mean);
//    // сортируем метки кластеров
//    // предполагаем, что:
//    // 1. самая темная метка - это фон,
//    // 2. самая светлая метка - это светодиоды
//    // 3. м/у ними - это ладонь
//    cv::sort(labels,labels,CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);
//    // смотрим на консоли
//    for(int i = 0; i <labels.rows; i++ ){
//        std::cout << (int)labels.at<uchar>(i,0) << endl;
//    }
//    int led_labels = 1; // сколько ярких кластеров считать диодами
//    int led_thresh = (int)labels.at<uchar>(labels.rows - led_labels,0);
//    Mat led;
//    // получаем бинарную картинку светодиодов
//    cv::threshold(k_mean, led, led_thresh - 1, 255, CV_THRESH_BINARY);
//    cv::imshow("leds",led);

//    // маркировка связных компонент
//    vector<ccl::Blob> blobs;
//    connected_components(led,blobs);
//    Mat bbox;
//    led.convertTo(bbox,CV_8UC3);
//    cv::cvtColor(img, bbox, CV_GRAY2BGR);
//    cv::Point centre(img.cols / 2, img.rows / 2);
//    for(int i = 0; i < blobs.size();i++){
//        cv::Scalar color(0,255,0);
//        // bbox
//        cv::rectangle(bbox,blobs[i].rect,color,2);
//        // centroid
//        cv::circle(bbox,blobs[i].centroid,2,color,2);
//        // line to centre
//        cv::line(bbox,blobs[i].centroid,centre,color,2);
//    }
//    cv::imshow("bbox",bbox);

////    Mat logPolar;
////    cvLogPolar( img, logPolar, cvPoint2D32f(img.cols/2,img.rows/2), 40,
////            CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS );

//    LogPolar_Interp logPolar(img.rows, img.cols, cv::Point(img.cols/2,img.rows/2),120);
//    Mat lp = logPolar.to_cortical(img), lp_hist;
//    drawHist(lp,lp_hist);
//    cv::imshow("log-polar",lp_hist);
//    Mat lp_kmeans, lp_labels;
//    kMeans(lp,lp_kmeans,lp_labels,num_cluster);
//    cv::imshow("k-means log-polar",lp_kmeans);


//    Mat histImg;
//    drawHist(img,histImg);
//    cv::imshow("origial + hist",histImg);

//    cv::waitKey(10);
//    vector<double> vec;
//    fitFeaturesProb(lp,vec);




    while (!_kbhit())
    {
        cv::waitKey(10);
    }
    cv::destroyAllWindows();
    return 0;
}
