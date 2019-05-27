#include "led.h"
#define _USE_MATH_DEFINES // for C++
#include <math.h>

using namespace std;
using namespace cv;

LedDetector::LedDetector(int lpSize, int clusters, int ledClusters,int ledCount):
    m_lpSize(lpSize), m_numClusters(clusters), m_ledClusters(ledClusters), m_ledCount(ledCount)
{

}

int LedDetector::predict(const Mat &src, vector<Point> centers)
{
    _clear();
    // в log-polar координаты
    LogPolar_Interp logPolar(src.rows, src.cols,
                              cv::Point(src.cols/2,src.rows/2),
                              m_lpSize,3.0,INTER_LINEAR,1, m_lpSize);
    m_lp = logPolar.to_cortical(src);
    // кластеризуем lp
    kMeans(m_lp,m_lpKmean,m_lpLabels,m_numClusters);
    // находимм максимум нижней проекции (линейный сдвиг по х в log polar соответствует масштабированию в cart координатах)
    m_lpRadius = _fitLpRadius(m_lp);
    // сдвинем по х так, чтоб максимум оказался по середине картинки
    Mat lpShiftAlign;
    _shiftAlign(m_lp,lpShiftAlign,m_lp.cols / 2 - m_lpRadius);
    m_lpShiftAligned = lpShiftAlign;
    // сортируем по возрастанию кластеров
    cv::sort(m_lpLabels,m_lpLabels,CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);
    // выбираем порог для поиска блобов для SimpleBlobDetector
    int blob_thresh = (int)m_lpLabels.at<uchar>(m_lpLabels.rows - (m_ledClusters + 1),0);
    _detectLogPolarBlobs(m_lp,blob_thresh,m_lpKeypoints);

    // пересчет радиуса(масштаба) из log polar в cart
    Point2d p = _logPolarToCart(m_lpRadius,0,
                                cv::Size(src.cols,src.rows),
                                cv::Size(m_lp.cols,m_lp.rows));
    m_cartRadius = sqrt(p.x*p.x + p.y*p.y);
    cout << "log polar radius = " << m_lpRadius << ", cart radius = " << m_cartRadius << endl;

    //бинаризация картинки
    // предполагаем, что:
    // 1. самая темная метка - это фон,
    // 2. самая светлая метка - это светодиоды
    // 3. м/у ними - это ладонь
    int led_thresh = (int)m_lpLabels.at<uchar>(m_lpLabels.rows - m_ledClusters,0);
    // получаем бинарную картинку светодиодов
    cv::threshold(m_lpKmean, m_lpBin, led_thresh - 1, 255, CV_THRESH_BINARY);

    // true, если мы обнаружили как минимум 3 "хороших" пятна
    return m_lpKeypoints.size() >= m_ledCount - 1;
}

int LedDetector::_fitLpRadius(const Mat &src)
{
    Mat colSum;
    cv::reduce(src,colSum,0,CV_REDUCE_SUM,CV_32SC1);
    double minVal,maxVal;
    cv::Point minLoc, maxLoc;
    minMaxLoc(src,&minVal,&maxVal,&minLoc,&maxLoc);
    return maxLoc.x;
}

void LedDetector::_clear()
{
    // todo
}

void LedDetector::_detectLogPolarBlobs(const Mat &src, float thresh, std::vector<KeyPoint> &keypoints)
{
    SimpleBlobDetector::Params params;
    params.minThreshold = thresh;
    params.thresholdStep = 1;
    params.maxThreshold = 255;

    params.filterByColor = true;
    params.blobColor = 255;

    params.filterByArea = true;
    // можно и поумнее как считать
    params.minArea = (m_lpSize / m_ledCount) * (m_lpSize / m_ledCount) / 8;
    params.maxArea = (m_lpSize / m_ledCount) * (m_lpSize / m_ledCount) * 2;

    params.filterByCircularity = true;
    params.minCircularity = 0.7;

    params.filterByConvexity = true;
    params.minConvexity = 0.8;

    params.filterByInertia = true;
    params.minInertiaRatio = 0.01;

    SimpleBlobDetector detector(params);
    keypoints.clear();
    keypoints.reserve(128);
    // где-то внутри detect есть ошибка
    detector.detect( src, keypoints);
}


Point2d LedDetector::_logPolarToCart(double ro, double phi, Size cartSize, Size polarSize)
{
    //ro = polarSize.width;
    //double rectRad = min(cartSize.height / 2,cartSize.width / 2);
    //double maxRadius = sqrt(2 * rectRad * rectRad);
    double maxRadius = sqrt((cartSize.height / 2) * (cartSize.height / 2) + (cartSize.width / 2) * (cartSize.width / 2));
    double M = polarSize.width / log(maxRadius);
    double Ky = polarSize.height / 360.;
    double r = exp(ro / M);
    //cout << r << endl;
    double angle = phi / Ky;
    double Krad = M_PI / 180;
    return Point2d(r * cos(angle * Krad), r * sin(angle * Krad));

}

void LedDetector::_shiftAlign(const Mat &src, Mat &dst, int shiftX, int shiftY)
{
    Mat m = Mat::zeros(2, 3, CV_32FC1);
    m.at<float>(0,0) = 1;
    m.at<float>(1,1) = 1;
    m.at<float>(0,2) = shiftX;
    m.at<float>(1,2) = shiftY;
    cv::warpAffine(src,dst,m,Size(src.cols,src.rows),INTER_LINEAR,BORDER_REPLICATE);

}

Mat LedDetector::lpBin() const
{
    return m_lpBin;
}

Mat LedDetector::lpShiftAligned() const
{
    return m_lpShiftAligned;
}

std::vector<KeyPoint> LedDetector::lpKeypoints() const
{
    return m_lpKeypoints;
}

Mat LedDetector::lpKmean() const
{
    return m_lpKmean;
}

int LedDetector::lpRadius() const
{
    return m_lpRadius;
}

Mat LedDetector::lp() const
{
    return m_lp;
}

int LedDetector::cartRadius() const
{
    return m_cartRadius;
}

void kMeans(const Mat &src, Mat &dst, Mat &clusters, int k){
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
    centers.convertTo(clusters, CV_8U);
}

void drawHist(const Mat &src, Mat &dst, int thickness, double histPart){
    CV_Assert(src.type() == CV_8UC1 && "src.type() == CV_8UC1"); // работаем только с серым
    Mat rowSum, colSum;
    cv::reduce(src,rowSum,1,CV_REDUCE_SUM,CV_32SC1);
    cv::reduce(src,colSum,0,CV_REDUCE_SUM,CV_32SC1);
    int rowPart = src.cols * histPart, colPart = src.rows * histPart;
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
}

double fitFeaturesProb(const Mat &src, vector<double> &results, int num_strips){
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
