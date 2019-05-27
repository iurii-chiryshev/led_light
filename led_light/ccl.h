#pragma once

#include <opencv/cv.h>

// std
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>



namespace ccl{

	using namespace std;
	using namespace cv;
/************************************************************************/
/*                       Blob                                           */
/************************************************************************/
struct Blob
{
public:
	Blob():area(0),M10(0),M01(0),M11(0),M20(0),M02(0){}
//    Blob(const Blob& other){
//        area = other.area;
//        M10 = other.M10;
//        M01 = other.M01;
//        M11 = other.M11;
//        M20 = other.M20;
//        M02 = other.M02;
//        Mu11 = other.Mu11;
//        Mu20 = other.Mu20;
//        Mu02 = other.Mu02;
//        centroid = other.centroid;
//        label = other.label;
//        rect = other.rect;
//        pix = other.pix;
//    }

	// площадь
	int area; // area = M00
	// моменты
	__int64 M10; 
	__int64 M01;
	__int64 M11; 
	__int64 M20; 
	__int64 M02; 
	// центральные моменты
	double Mu11; 
	double Mu20; 
	double Mu02; 
	// центройд (центр масс блоба)
	cv::Point centroid;
	// метка
	int label;
	// ограничивающий прямоугольник
	cv::Rect rect;
	// каким цветом был объект 
	// на исходном изображении
	// добавил, чтоб отличать объект и фон
	uchar pix;
};
typedef Blob* BlobPtr;
typedef vector<BlobPtr> Blobs;
/************************************************************************/
/*               каким алгоритмом желаем получить связные компоненты    */
/************************************************************************/
enum 
{
	CCL_SAUF, // Two Strategies to Speed up Connected Component Labeling Algorithms
	CCL_FLOODFILL, // cv::floodFill
};

/************************************************************************/
/*            освободить память под блобы                               */
/************************************************************************/
void freeBlobs(Blobs& blobs);
/************************************************************************/
/*                       маркировка связных компонент                   */
/************************************************************************/


/**
 * @brief connectedComponents
 * @param src_img
 * @param dst_labels - выходной массив меток [int]
 * @param blobs - блобы
 * @param connectivity - связность 4 или 8
 * @param algorithm - какой алгоритм
 * @param minarea - минимально допустимая площадь региона
 * @param maxarea - максимально допустимая площадь региона
 */
void connectedComponents(InputOutputArray src_img,
                         OutputArray dst_labels,
                         Blobs& blobs,
                         int connectivity = 4,
                         int algorithm = CCL_SAUF,
                         int minarea = 0,
                         int maxarea = INT_MAX);

void connectedComponents(const Mat & src,
                         vector<Blob> &blobs,
                         int connectivity = 4,
                         int algorithm = CCL_SAUF,
                         int minarea = 0,
                         int maxarea = INT_MAX);

}



