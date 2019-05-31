#include "led.h"
#define _USE_MATH_DEFINES // for C++
#include <math.h>
#include <ccl.h>

using namespace std;
using namespace cv;
using namespace ccl;

/**
 * @brief The LogPolar_Interp_Ext class
 * Расширение log-polar преобразования. Уж очень хитро он считает.
 * И обратное преобразование по классическим формулам не работает.
 */
class LogPolar_Interp_Ext: public LogPolar_Interp{

public:
    LogPolar_Interp_Ext(): LogPolar_Interp() {}
    LogPolar_Interp_Ext(int w,
                        int h,
                        Point2i center,
                        int R=70,
                        double ro0=3.0,
                        int interp=INTER_LINEAR,
                        int full=1,
                        int S=117,
                        int sp=1): LogPolar_Interp(w,h,center,R,ro0,interp,full,S,sp) {}
    template <typename T>
    Point_<T> toCart(Point point)
    {
        const int u = point.x, v = point.y;
        return Point_<T>(cv::saturate_cast<T>(Csri.at<float>(v,u) - left),
                    cv::saturate_cast<T>(Rsri.at<float>(v,u) - top));
    }
};

template <typename T>
static Point_<T> _lpToCart(T ro, T phi, Size cartSize, Size polarSize)
{
    //ro = polarSize.width;
    //double rectRad = min(cartSize.height / 2,cartSize.width / 2);
    //double maxRadius = sqrt(2 * rectRad * rectRad);
    double maxRadius = sqrt((cartSize.height / 2) * (cartSize.height / 2) + (cartSize.width / 2) * (cartSize.width / 2));
    double M = polarSize.width / log(maxRadius);
    double Ky = polarSize.height / 360.;
    double r = exp( ro / M );
    //cout << r << endl;
    double angle = phi / Ky;
    double Krad = M_PI / 180;
    return Point_<T>(cv::saturate_cast<T>(r * cos(angle * Krad) + cartSize.width / 2),
                     cv::saturate_cast<T>(r * sin(angle * Krad) + cartSize.height / 2));

}
template <typename T>
static Point_<T> _lpToCart(Point_<T> point, Size cartSize, Size polarSize)
{
    return _lpToCart<T>(point.x,point.y,cartSize,polarSize);
}


Led::Led(int lpSize, int clusters, int ledClusters,int ledCount):
    m_lpSize(lpSize), m_numClusters(clusters), m_ledClusters(ledClusters), m_ledCount(ledCount)
{

}

void Led::operator() (const Mat &src, Mat &vec)
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
    m_lpRadius = _estimateLpRadius(m_lp);
    // сдвинем по х так, чтоб максимум оказался по середине картинки
    Mat lpShiftAlign;
    _shiftAlign(m_lp,lpShiftAlign,m_lp.cols / 2 - m_lpRadius);
    m_lpShiftAligned = lpShiftAlign;
    vec = lpShiftAlign;
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
    //cout << "log polar radius = " << m_lpRadius << ", cart radius = " << m_cartRadius << endl;

    //бинаризация картинки
    // предполагаем, что:
    // 1. самая темная метка - это фон,
    // 2. самая светлая метка - это светодиоды
    // 3. м/у ними - это ладонь
    int led_thresh = (int)m_lpLabels.at<uchar>(m_lpLabels.rows - m_ledClusters,0);
    // получаем бинарную картинку светодиодов
    cv::threshold(m_lpKmean, m_lpBin, led_thresh - 1, 255, CV_THRESH_BINARY);
}

static void _norm(const Mat &src, Mat &dst) {
    // out = (in - mean) / std
    Scalar mean,std;
    cv::meanStdDev(src,mean,std);
    src.convertTo(dst,CV_32FC1,1,-mean[0]);
    dst.convertTo(dst,CV_32FC1,1 / std[0],0);
}

double Led::compare(const Mat &model, Mat &target)
{
    Mat norm_model, norm_target;
    // устраняем влияние освещенности
    _norm(model,norm_model);
    _norm(target,norm_target);
    Mat res(1,1,CV_32FC1);
    const vector<int> methods = {
        TM_SQDIFF_NORMED,
        TM_CCORR_NORMED,
        TM_CCOEFF_NORMED
    };
    int method = methods[0] ;
    cv::matchTemplate(norm_model,norm_target,res,method);
    if (method == methods[0])
        return 1 - res.at<float>(0,0);
    return res.at<float>(0,0);
}

void Led::estimate(const Mat &src,Mat &dst,int size, double &mean, double &std)
{
    const vector<double> scales = {
        0.5,
        0.75,
        1.0,
        1.25,
        1.5
    };
    const vector<double> gammas = {
        0.6,
        0.9,
        1.0,
        1.3,
        1.9
    };
    const vector<int> flips = {-1,0,1};

    vector<Mat> leds;
    for(int s = 0; s < scales.size(); s++)
        for(int g = 0; g < gammas.size(); g++)
            for(int f = 0; f < flips.size(); f++){
                string name = std::to_string(s) + std::to_string(g) + std::to_string(f);
                // гамма коррекция
                Mat gamma;
                correctGamma(src,gamma,gammas[g]);
                // flip
                Mat flip;
                cv::flip(gamma,flip,flips[f]);
                // масштаб
                Mat scale,m = getRotationMatrix2D(Point2f(src.cols / 2,src.rows / 2), 0, scales[s]);
                cv::warpAffine(flip,scale,m,Size(flip.cols,flip.rows),INTER_LINEAR,BORDER_REPLICATE);

                Mat vec;
                Led led(size);
                led(scale,vec);
                leds.push_back(vec);

                // рисуем
//                Mat rgb;
//                drawHist(led.lpShiftAligned(),rgb);
//                cv::imshow(name,rgb);
//                cv::waitKey(1);
            }
    Mat mean_led = Mat::zeros(size,size,CV_32FC1);
    for(int i = 0; i < leds.size(); i++){
        cv::add(leds[i],mean_led,mean_led,cv::noArray(),mean_led.type());
    }
    mean_led.convertTo(dst,CV_8UC1,1./leds.size());
    Mat rgb;
    drawHist(dst,rgb);
    cv::imshow("mean vector",rgb);
    vector<float> dist;
    // попарное сраввнение векторов
    for(int i = 0; i < leds.size(); i++){
        dist.push_back(1 - Led::compare(leds[i],mean_led));
//        for(int j = 0; j < leds.size(); j++){
//            dist.push_back(1 - Led::compare(leds[i],leds[j]));
//        }
    }
    // считаем mean и std
    Mat mat_dist(1,dist.size(),CV_32FC1,(void*)dist.data());
    Scalar s_mean, s_std;
    cv::meanStdDev(mat_dist,s_mean,s_std);
    mean = 1 - s_mean[0];
    cv::multiply(mat_dist,mat_dist,mat_dist);
    cv::meanStdDev(mat_dist,s_mean,s_std);
    std = sqrt(s_mean[0]);


}

int Led::_estimateLpRadius(const Mat &src)
{
    Mat colSum;
    cv::reduce(src,colSum,0,CV_REDUCE_SUM,CV_32SC1);
    double minVal,maxVal;
    cv::Point minLoc, maxLoc;
    minMaxLoc(src,&minVal,&maxVal,&minLoc,&maxLoc);
    return maxLoc.x;
}

void Led::_clear()
{
    // todo
}

void Led::_detectLogPolarBlobs(const Mat &src, float thresh, std::vector<KeyPoint> &keypoints)
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


Point2d Led::_logPolarToCart(double ro, double phi, Size cartSize, Size polarSize)
{
    return _lpToCart<double>(ro,phi,cartSize,polarSize);
}

void Led::_shiftAlign(const Mat &src, Mat &dst, int shiftX, int shiftY)
{
    Mat m = Mat::zeros(2, 3, CV_32FC1);
    m.at<float>(0,0) = 1;
    m.at<float>(1,1) = 1;
    m.at<float>(0,2) = shiftX;
    m.at<float>(1,2) = shiftY;
    cv::warpAffine(src,dst,m,Size(src.cols,src.rows),INTER_LINEAR,BORDER_REPLICATE);

}

Mat Led::lpBin() const
{
    return m_lpBin;
}

Mat Led::lpShiftAligned() const
{
    return m_lpShiftAligned;
}

std::vector<KeyPoint> Led::lpKeypoints() const
{
    return m_lpKeypoints;
}

Mat Led::lpKmean() const
{
    return m_lpKmean;
}

int Led::lpRadius() const
{
    return m_lpRadius;
}

Mat Led::lp() const
{
    return m_lp;
}

int Led::cartRadius() const
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

void calculateFeatures(const Mat &src,
                       vector<double> &results,
                       const vector<int> &strip_counts,
                       const vector<int> &match_methods,
                       const vector<int> &hist_methods
                       ){
    CV_Assert(src.type() == CV_8UC1 && "src.type() == CV_8UC1"); // работаем только с серым
//    vector<int> match_methods = {
//        CV_TM_CCORR_NORMED,
//        CV_TM_CCOEFF_NORMED
//    };
//    vector<int> hist_methods = {
//        CV_COMP_CORREL,
//        //CV_COMP_BHATTACHARYYA,
//    };
    results.clear();
    for(int s = 0; s < strip_counts.size(); s++){
        const int num_strip = strip_counts[s];
        const int strip_rows = src.rows / num_strip, strip_cols = src.cols;
        vector<Mat> strips;
        for(int i = 0; i < num_strip; i++){
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
        for(int i =0; i < strips.size(); i++){
            for(int j=i+1; j < strips.size();j++){
                for (int k = 0; k < match_methods.size();k++){
                    Mat res(1,1,CV_32FC1);
                    cv::matchTemplate(strips[i],strips[j],res,match_methods[k]);
                    results.push_back(res.at<float>(0,0));
                }
                for (int k = 0; k < hist_methods.size();k++){
                    double diff = cv::compareHist(hists[i],hists[j],hist_methods[k]);
                    results.push_back(diff);
                }
            }
        }
    }
}

void correctGamma(const Mat &src, Mat &dst, double gamma)
{
    CV_Assert(src.depth() == CV_8U);
    double power = 1.0 / gamma;

    Mat lut(1, 256, CV_8UC1 );
    uchar * ptr = lut.ptr();
    for( int i = 0; i < 256; i++ )
    {
        ptr[i] = saturate_cast<uchar>( pow( i / 255.0, power ) * 255.0 );
    }
    cv::LUT( src, lut, dst);
}


static int _getGaussKsize(double sigma){
    int ksize = ((sigma - 0.8) / 0.3 + 1)*2;
    if (ksize % 2 == 0) ksize++;
    return ksize;
}

static cv::Mat _getGaussianKernel(double sigmaX, double sigmaY,int ktype = CV_64F)
{
    int ksizeX = _getGaussKsize(sigmaX);
    int ksizeY = _getGaussKsize(sigmaY);
    cv::Mat gaussX = cv::getGaussianKernel(ksizeX, sigmaX, ktype);
    cv::Mat gaussY = cv::getGaussianKernel(ksizeY, sigmaY, ktype);
    cv::Mat k = gaussY * gaussX.t() ;
    cv::Scalar sum = cv::sum(k);
    return k;

}

static void _DoG(const Mat& src, Mat& dst,double sigmaX,double sigmaY){
    //CV_Assert(src.type() == CV_32FC1 && "src.type() == CV_32FC1");
    double delta = 5e-6;
    Mat k1 = _getGaussianKernel(sigmaX+delta,sigmaY+delta);
    Mat k2 = _getGaussianKernel(sigmaX,sigmaY);
    Mat dog = k1 - k2;
    double minVal,maxVal;
    dog.convertTo(dog,dog.type(),-sigmaX*sigmaY / delta);
    cv::minMaxLoc(dog,&minVal,&maxVal);
    Mat dog_gray;
    dog.convertTo(dog_gray,dog.type(),1./(maxVal-minVal),-minVal/(maxVal-minVal));
    cv::minMaxLoc(dog_gray,&minVal,&maxVal);
    dog_gray.convertTo(dog_gray,CV_8UC1,255);
    imshow("dog kernel", dog_gray);
    //cout << "log kernel, maxVal = " << maxVal << " minVal = " << minVal << endl;
    cv::filter2D(src,dst,-1,dog);
    cv::minMaxLoc(dst,&minVal,&maxVal);
    //cout << "maxVal = " << maxVal << endl;
    dst.convertTo(dst,dst.type(),1./(maxVal-minVal),-minVal/(maxVal-minVal));
    dst.convertTo(dst,dst.type(),255);
    cv::minMaxLoc(dst,&minVal,&maxVal);
    return;
}

static void _nonMaximaSuppression(const cv::Mat& src, cv::Mat& dst, bool removePlateaus = false) {
    cv::dilate(src, dst, cv::Mat());
    //cv::imshow("mask1",dst);
    cv::compare(src,dst,dst, cv::CMP_GE);

    // убираем плато -> морф. оконтуривание
    if (removePlateaus) {
        cv::Mat non_plateau_mask;
        cv::erode(src, non_plateau_mask, cv::Mat());
        cv::compare(src, non_plateau_mask, non_plateau_mask, cv::CMP_GT);
        cv::bitwise_and(dst, non_plateau_mask, dst);
    }
}

static void _findBestMaximums(const Mat &src,vector<Point> &points,int maxPoints = 4, int radius = 1){
    CV_Assert(radius > 0 && "radius > 0");
    points.clear();
    Mat mat;
    src.convertTo(mat,src.type());
    for(int i = 0; i < maxPoints;i++){
        double minVal, maxVal;
        Point minId, maxId;
        cv::minMaxLoc(mat,&minVal,&maxVal,&minId,&maxId);
        // ракрашиваем в 0 участок вокруг области с радусом
        Point tl(MAX(0,maxId.x - radius),
                 MAX(0,maxId.y - radius));
        int size = radius * 2 + 1;
        Rect rect(tl.x,tl.y,
                  MIN(mat.cols - tl.x,radius * 2 + 1),
                  MIN(mat.rows - tl.y,radius * 2 + 1));
        mat(rect).setTo(Scalar::all(0));
        points.push_back(maxId);
        //imshow(std::to_string(i),mat);
        //waitKey(1);
    }
}

static bool _isLedPoint(const vector<Point> &points,Size area,int numPoint){
    if (points.size() != numPoint) return false;
    // среднее по X
    int meanX = 0; for (auto p : points)  meanX += p.x;
    meanX /= numPoint;
    // нормируем
    vector<Point> norms;
    const int period = area.height / numPoint;
    bool result = true;
    int expected_sum = 0, received_sum = 0;
    for(int i = 0 ; i < points.size(); i++){
        Point n(points[i].x - meanX, points[i].y % period - period / 2);
        norms.push_back(n);
        result = result && sqrt(n.ddot(n)) < period / 2;
        expected_sum += i;
        received_sum += points[i].y / period;
    }
    return result && expected_sum == received_sum;
}

bool findLeds(const Mat &src, vector<Point> &leds)
{
    CV_Assert(src.type() == CV_8UC1 && "src.type() == CV_8UC1");
    const int lpSize = 256, ledCount = 4;
    // преобразуем в log polar координаты
    LogPolar_Interp_Ext logPolar(src.rows, src.cols,
                              cv::Point(src.cols/2,src.rows/2),
                              lpSize,3.0,INTER_LINEAR,1, lpSize);
    Mat lp = logPolar.to_cortical(src), lp_rgb;

    // применяем оператор лапласа
    // сначала сгладим
    GaussianBlur( lp, lp, Size(3,3), 0, 0, BORDER_DEFAULT );
    drawHist(lp,lp_rgb);
    cv::imshow("log polar",lp_rgb);
    int r = lpSize / 16;
    double sigmaY = r / sqrt(2.);
    double sigmaX = sigmaY; // * sqrt(2.);
    lp.convertTo(lp,CV_32FC1);
    Mat dog,dog_rgb;
    _DoG(lp,dog,sigmaX,sigmaY);
    dog.convertTo(dog,CV_8UC1);
    drawHist(dog,dog_rgb);
    cv::imshow("dog",dog_rgb);
    // бинаризация
    Mat otsu,otsu_rgb;
    cv::threshold(dog, otsu, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    drawHist(otsu,otsu_rgb);
    cv::imshow("otsu",otsu_rgb);
    // kmean на 3 класса
    Mat kmean,kmena_rgb, clasters;
    kMeans(dog,kmean,clasters,2);
    drawHist(kmean,kmena_rgb);
    cv::imshow("kmean",kmena_rgb);

    //
    vector<Point> points;
    _findBestMaximums(dog,points,ledCount,lpSize / 8);
    Mat points_rgb;
    drawHist(dog,points_rgb);
    for (int i = 0; i < points.size(); i++){
        cv::circle(points_rgb,points[i],1,Scalar(0,255,0),2);
    }
    cv::imshow("dog max",points_rgb);
    bool res = _isLedPoint(points,Size(dog.cols,dog.rows),ledCount);
    leds.clear();
    for(auto p : points) leds.push_back(logPolar.toCart<int>(p));
    return res;
}
