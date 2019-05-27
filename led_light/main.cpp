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

    int size = 128;
    Led target_led(size);
    // любой из "good" берем за опорный, с которым будем сравнивать
    int target_id = 4;
    string target_name = path + imgs[target_id].first;
    Mat target_gray = cv::imread(target_name,IMREAD_GRAYSCALE);
    Mat target;
    // считаем целевой вектор
    target_led(target_gray,target);

    for (int i = 0; i < imgs.size(); i++){
        Led led(size);
        const string &name = imgs[i].first;
        string fullName = path + name;
        Mat gray = cv::imread(fullName,IMREAD_GRAYSCALE);
        if(gray.empty()) continue;
        Mat model;
        led(gray,model);
        double result = Led::compare(model,target);
        cout << name << " compare result = " << result << endl;



        // рисуем картинки

        // original
        Mat rgb;
        drawHist(gray,rgb);
        cv::imshow(name,rgb);
        // log polar
        Mat lp;
        drawHist(led.lp(),lp);
        int lpRad = led.lpRadius(); // log polar
        cv::line(lp,Point(lpRad,0),Point(lpRad,lp.rows),cv::Scalar(255,0,0));
        drawKeypoints( lp, led.lpKeypoints(), lp,
                       Scalar(255,0,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
        cv::imshow(name + " lp",lp);
        // binary lp
        cv::imshow(name + " lp bin",led.lpBin());
        // выравненный log polar
        Mat lpShift;
        drawHist(led.lpShiftAligned(),lpShift);
        cv::imshow(name + " lp shift",lpShift);
        // kmean log polar
        Mat lpKmean;
        drawHist(led.lpKmean(),lpKmean);
        cv::imshow(name + " lp kmean",lpKmean);

        cv::waitKey(1);
    }

    while (!_kbhit())
    {
        cv::waitKey(10);
    }
    cv::destroyAllWindows();
    return 0;
}
