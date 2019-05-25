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
#include <opencv2/contrib/contrib.hpp>

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
            TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 5, 1),
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


double corr(const Mat & src, vector<double> &results,int num_strips = 4){
     CV_Assert(src.type() == CV_8UC1 && "src.type() == CV_8UC1"); // работаем только с серым
     vector<Mat> strips;
     const int strip_rows = src.rows / num_strips, strip_cols = src.cols;
     for(int i = 0; i < num_strips; i++){
         Rect roi(0,i*strip_rows,strip_cols,strip_rows);
         Mat strip;
         src(roi).convertTo(strip,src.type());
         strips.push_back(strip);
         imshow(std::to_string(i),strip);
     }
     vector<Mat> hists;
     int histSize = 256;
     float range[] = { 0, 256 } ;
     const float* histRange = { range };
     bool uniform = true; bool accumulate = false;
     for(int i = 0; i < strips.size();i++){
         Mat hist;
         calcHist( &strips[i], 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
         hists.push_back(hist);
     }

     results.clear();
     vector<int> mt_methods = {
         CV_TM_CCORR_NORMED,
         CV_TM_CCOEFF_NORMED
     };
     vector<int> hist_methods = {
         CV_COMP_CORREL,
         //CV_COMP_BHATTACHARYYA,
     };
     for(int i =0; i < strips.size(); i++){
         for(int j=i+1; j < strips.size();j++){
             for (int k = 0; k < mt_methods.size();k++){
                 Mat res(1,1,CV_32FC1);
                 cv::matchTemplate(strips[i],strips[j],res,mt_methods[k]);
                 results.push_back(res.at<float>(0,0));
             }
             for (int k = 0; k < hist_methods.size();k++){
                 double diff = cv::compareHist(hists[i],hists[j],hist_methods[k]);
                 results.push_back(diff);
             }
         }
     }
     return 0;
}

/**
 * @brief draw_hist
 * @param src
 * @param dst
 * @param thickness
 * @param hist_part
 */
void draw_hist(const Mat & src,Mat &dst,int thickness = 1, double hist_part = 0.2){
    CV_Assert(src.type() == CV_8UC1 && "src.type() == CV_8UC1"); // работаем только с серым
    Mat rowSum, colSum;
    cv::reduce(src,rowSum,1,CV_REDUCE_SUM,CV_32SC1);
    cv::reduce(src,colSum,0,CV_REDUCE_SUM,CV_32SC1);
    int rowPart = src.cols * hist_part, colPart = src.rows * hist_part;
    normalize(rowSum, rowSum, 0, rowPart, NORM_MINMAX, -1, Mat() );
    normalize(colSum, colSum, 0, colPart, NORM_MINMAX, -1, Mat() );

    cv::cvtColor(src, dst, CV_GRAY2BGR);

    for( int i = 1; i < colSum.cols; i++ ){
        line( dst, Point( i-1, src.rows - colSum.at<int>(0,i-1) ) ,
                     Point( i, src.rows - colSum.at<int>(0,i) ),
                     Scalar( 0, 255, 0), thickness);
    }

    for( int i = 1; i < rowSum.rows; i++ ){
        line( dst, Point(src.cols - rowSum.at<int>(i-1,0), i-1 ) ,
                     Point(src.cols - rowSum.at<int>(i,0), i ),
                     Scalar( 0, 0, 255), thickness);
    }





//    Mat hist;
//    int histSize = 256;
//    float range[] = { 0, 256 } ;
//    const float* histRange = { range };
//    cv::calcHist(&src,1,0,Mat(),hist,1,&histSize, &histRange,true,false);
}

int main(int argc, char *argv[])
{
    const string path = "D://images//good_0.png";
    // картинку
    Mat img = cv::imread(path,IMREAD_GRAYSCALE);
    Mat img_hist;
    draw_hist(img,img_hist);
    cv::imshow("original",img_hist);
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

//    Mat logPolar;
//    cvLogPolar( img, logPolar, cvPoint2D32f(img.cols/2,img.rows/2), 40,
//            CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS );

    LogPolar_Interp logPolar(img.rows, img.cols, cv::Point(img.cols/2,img.rows/2),120);
    Mat lp = logPolar.to_cortical(img), lp_hist;
    draw_hist(lp,lp_hist);
    cv::imshow("log-polar",lp_hist);
    Mat lp_kmeans, lp_labels;
    k_means(lp,lp_kmeans,lp_labels,num_cluster);
    cv::imshow("k-means log-polar",lp_kmeans);


    Mat histImg;
    draw_hist(img,histImg);
    cv::imshow("origial + hist",histImg);

    cv::waitKey(10);
    vector<double> vec;
    corr(lp,vec);




    while (!_kbhit())
    {
        cv::waitKey(10);
    }
    cv::destroyAllWindows();
    return 0;
}
