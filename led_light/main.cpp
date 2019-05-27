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
        LedDetector detector(256);
        const string &name = imgs[i].first;
        string fullName = path + name;
        Mat gray = cv::imread(fullName,IMREAD_GRAYSCALE);
        if(gray.empty()) continue;
        vector<Point> centers;
        float value = detector.predict(gray,centers);
        cout << name << " predicted: " << value << ", expected: " << imgs[i].second << endl;

        // рисуем картинки

        // original
//        Mat rgb;
//        drawHist(gray,rgb);
//        cv::imshow(name,rgb);
        // log polar
        Mat lp;
        drawHist(detector.lp(),lp);
        int lpRad = detector.lpRadius(); // log polar
        cv::line(lp,Point(lpRad,0),Point(lpRad,lp.rows),cv::Scalar(255,0,0));
        drawKeypoints( lp, detector.lpKeypoints(), lp,
                       Scalar(255,0,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
        cv::imshow(name + " lp",lp);
        // binary lp
        cv::imshow(name + " lp bin",detector.lpBin());
        // выравненный log polar
        Mat lpShift;
        drawHist(detector.lpShiftAligned(),lpShift);
        cv::imshow(name + " lp shift",lpShift);
        // kmean log polar
        Mat lpKmean;
        drawHist(detector.lpKmean(),lpKmean);
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
