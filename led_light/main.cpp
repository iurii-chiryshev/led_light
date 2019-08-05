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


int main(int argc, char *argv[]){
    const string path = "D://images//";
    vector<pair<string,bool>> imgs = {
        {"good_0.png",true},
        {"good_1.png",true},
        {"good_2.png",true},
        {"good_3.png",true},
        {"good_4.png",true},
        {"good_5.png",true},
        {"good_6.png",true},
        {"good_7.png",true},
        {"good_8.png",true},
        {"bad_0.png",false},
        {"bad_1.png",false},
        {"bad_2.png",false},
        {"bad_3.png",false},
//        {"10_50_30_1030.bmp",false},
//        {"12_36_57_1057.bmp",false},
//        {"17_28_22_1022.bmp",false},
//        {"Pl_18_16_52_718.bmp",false},
//        {"Pl_18_16_52_562.bmp",false},
//        {"Pl_18_16_52_718.bmp",false},
//        {"Pl_18_16_52_640.bmp",false},
    };

    for (int i = 0; i < imgs.size(); i++){
        const string &name = imgs[i].first;
        string fullName = path + name;
        Mat gray = cv::imread(fullName,IMREAD_GRAYSCALE);
        vector<Point> leds;
        bool find = findLeds(gray,leds);
        cout << name << ", detected status = " << find << ", expected = " << imgs[i].second << endl;
        Mat rgb;
        drawHist(gray,rgb);
        for(auto p : leds) cv::circle(rgb,p,2,Scalar(0,255,0),2);
        cv::imshow(name,rgb);
        waitKey(2000);
    }

    cout << "Press enter to exit..";
    while (!_kbhit())
    {
        cv::waitKey(10);
    }
    cv::destroyAllWindows();
    return 0;
}

//int main(int argc, char *argv[])
//{
//    const string path = "D://images//";
//    vector<pair<string,bool>> imgs = {
//        {"good_0.png",true},
//        {"good_1.png",true},
//        {"good_2.png",true},
//        {"good_3.png",true},
//        {"good_4.png",true},
//        {"good_5.png",true},
//        {"bad.png",false},
//    };

//    int size = 128;
//    Led target_led(size);
//    // любой из "good" берем за опорный, с которым будем сравнивать
//    int target_id = 2;
//    string target_name = path + imgs[target_id].first;
//    Mat target_gray = cv::imread(target_name,IMREAD_GRAYSCALE);
//    Mat target;
//    // считаем целевой вектор
//    target_led(target_gray,target);
//    double mean, std;
//    Led::estimate(target_gray,target,size,mean,std);
//    double threshold = mean - 2*std;
//    cout << "estimate mean = " << mean << ", std = " << std << ", threshold(mean - 2*std) = " << threshold << endl;

//    for (int i = 0; i < imgs.size(); i++){
//        Led led(size);
//        const string &name = imgs[i].first;
//        string fullName = path + name;
//        Mat gray = cv::imread(fullName,IMREAD_GRAYSCALE);
//        //correctGamma(gray,gray,2);
//        if(gray.empty()) continue;
//        Mat model;
//        led(gray,model);
//        double received = Led::compare(model,target);
//        cout << name << " compare result = " << received << endl;
//        cout << name << " expected: " <<  imgs[i].second << ", receive: " << (received >= threshold) <<endl;



//        // рисуем картинки

//        // original
//        Mat rgb;
//        drawHist(gray,rgb);
//        cv::imshow(name,rgb);
//        // log polar
//        Mat lp;
//        drawHist(led.lp(),lp);
//        int lpRad = led.lpRadius(); // log polar
//        cv::line(lp,Point(lpRad,0),Point(lpRad,lp.rows),cv::Scalar(255,0,0));
//        drawKeypoints( lp, led.lpKeypoints(), lp,
//                       Scalar(255,0,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
//        cv::imshow(name + " lp",lp);
//        // binary lp
//        cv::imshow(name + " lp bin",led.lpBin());
//        //выравненный log polar
//        Mat lpShift;
//        drawHist(led.lpShiftAligned(),lpShift);
//        cv::imshow(name + " lp shift",lpShift);
//        // kmean log polar
//        Mat lpKmean;
//        drawHist(led.lpKmean(),lpKmean);
//        cv::imshow(name + " lp kmean",lpKmean);

//        cv::waitKey(1);
//    }

//    while (!_kbhit())
//    {
//        cv::waitKey(10);
//    }
//    cv::destroyAllWindows();
//    return 0;
//}
