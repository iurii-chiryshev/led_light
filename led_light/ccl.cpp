#include "ccl.h"
#include "ccl_sauf.h"

namespace ccl{
	using namespace cv;
	using namespace std;
	/************************************************************************/
    /*               закрасить мелкие области                              */
	/************************************************************************/
	static inline void clean_labels(Mat& img, Mat& labels,const int val, const cv::Rect& rect)
	{
		const int ncols = img.cols;
		uchar* idata0 = img.ptr(rect.y);
		int* ldata0 = (int*)labels.ptr(rect.y);
		for(int y = 0; y<rect.height;++y)
		{
			uchar* idata = idata0 + ncols*y + rect.x;
			int* ldata = ldata0 + ncols*y + rect.x;
			for(int x = 0; x< rect.width; ++x)
			{
				if(ldata[x] == val)
				{
					ldata[x] = 0;
					idata[x] = 0;
				}
			}
		}
	}

	/************************************************************************/
	/*            маркировка с использованием FloodFill                   */
	/************************************************************************/
	static inline void ccl_ff(InputOutputArray src,OutputArray dst, Blobs& blobs,int connectivity, int minarea, int maxarea)
	{
		/* входной массив */
		cv::Mat image = src.getMat();
		const int nrows = image.rows;
		const int ncols = image.cols;

		/* выходной массив */
		// заполним картинку меток отрицательными значениями
		image.convertTo(dst,CV_32SC1,-1);
		Mat label_image = dst.getMat();

		CvMat cvlabel_image = label_image;
		CvScalar zero_scalar = cvScalarAll(0);
		int label_count = 0; 

		int* data0 = (int*)label_image.ptr();
		uchar* img0 = image.ptr();
		const size_t blob_size = sizeof(Blob);
		// пямять под максимум меток
		const size_t Plength = size_t(ncols*nrows);
		BlobPtr* P  = (BlobPtr*)fastMalloc(Plength);
		P[0] = NULL;
		int sz = 0;
		for(int y = 0; y < nrows; ++y) 
		{
			int* data = data0 + ncols*y;
			uchar* img = img0 + ncols*y;
			for(int x = 0; x < ncols; ++x) 
			{
				if(data[x] >= 0) 
				{
					//если уже помечен или 0 на следующий пиксель
					continue;
				}

				// начинаем заливку с текущего пикселя
				CvConnectedComp ccomp;
				/*
				  для ускорения можно переписать cvFloodFill
				  чтоб статистику считала внутри, а не доп. проходом
				*/
				cvFloodFill(&cvlabel_image,cvPoint(x,y), cvScalarAll(++label_count), zero_scalar, zero_scalar, &ccomp, connectivity, 0);
				int area = (int)ccomp.area;
				cv::Rect rect = ccomp.rect;
				if(area < minarea || area > maxarea)
				{
					// закрасить в 0
					clean_labels(image,label_image,label_count,rect);
					P[label_count] = NULL;
				}
				else
				{
					// считаем статистику + заполняем нашу структуру
					Blob tmp_blob;
					tmp_blob.area = area;
					tmp_blob.rect = rect;
					tmp_blob.label = label_count;
					tmp_blob.pix = img[x];
					const int rh = rect.height , rw = rect.width;
					/*расчет статистики*/
					int* roi0 = data0 + rect.y*ncols + rect.x;
					for(int i = 0; i < rh; ++i)
					{
						int* roi = roi0 + i*ncols;
						for(int j = 0; j < rw; ++j)
						{
							if(roi[j] == label_count)
							{
								__int64 xx = j + x;
								__int64 yy = i + y;
								tmp_blob.M10 += xx;
								tmp_blob.M01 += yy;
								tmp_blob.M11 += xx*yy;
								tmp_blob.M20 += xx*xx;
								tmp_blob.M02 += yy*yy;


							}
						}
					}
					// расчитать остальное
                    tmp_blob.centroid = cv::Point(tmp_blob.M10 / tmp_blob.area, tmp_blob.M01 / tmp_blob.area);
					// добавить блоб в список
					P[label_count] = (BlobPtr)malloc(blob_size);
					memcpy((void*)P[label_count],(void*)(&tmp_blob),blob_size);
					++sz;
				}
			}
		}
		/*выделяем память под */
		blobs.resize(label_count);
		for(int i = 0; i < label_count;++i)
		{
			if(P[i] != NULL)
				blobs[i] = P[i];
		}
		fastFree((void*)P);
	}

	/************************************************************************/
	/*						    SAUF						                */
	/************************************************************************/

	static inline void ccl_sauf(InputOutputArray src,OutputArray dst, Blobs& blobs,int connectivity, int minarea, int maxarea)
	{
		const size_t blob_size = sizeof(Blob);
		Mat stats_32,stats_64;
		/*выполняем маркировку*/
		const int nlabels = connectedComponentsWithStats_sauf(src,dst,stats_32,stats_64,connectivity,CV_32S);
		/*записываем данные в нашу структуру*/
		blobs.resize(nlabels-1,NULL); 
		int sz = 0;
		int* row0 = (int*)stats_32.ptr();
		for(int i = 1; i< nlabels; ++i)
		{
			
			int* row = (int*)stats_32.ptr(i);
			cv::Rect rect = cv::Rect(row[CC_32_STAT_LEFT],row[CC_32_STAT_TOP],row[CC_32_STAT_WIDTH],row[CC_32_STAT_HEIGHT]);
			int area = row[CC_32_STAT_AREA];
			uchar pix = (uchar)row[CC_32_STAT_PIX];
			int label = i;

			if(area < minarea || area > maxarea)
			{
				//не проходим по размерам
				//закрасить labels и src 0
				clean_labels(src.getMat(),dst.getMat(),label,rect);
				blobs.pop_back();
			}
			else
			{
				BlobPtr blobptr = (BlobPtr)malloc(blob_size);
				blobptr->rect = rect;
				blobptr->area = area;
				blobptr->pix = pix;
				blobptr->label = label;
				//////////////////////////////////////////////////////////////////////////
				__int64* row64 = (__int64*)(stats_64.ptr(i));
				blobptr->M10 = row64[CC_64_STAT_M10];
				blobptr->M01 = row64[CC_64_STAT_M01];
				blobptr->M20 = row64[CC_64_STAT_M20];
				blobptr->M02 = row64[CC_64_STAT_M02];
				blobptr->M11 = row64[CC_64_STAT_M11];
                blobptr->centroid = cv::Point(blobptr->M10 / blobptr->area, blobptr->M01 / blobptr->area);
				blobs[sz++] = blobptr;
			}

		}
		/*здеся можно пофильровать по размеру + досчитать статистику*/
		//....
	}
	
	/************************************************************************/
	/*                       маркировка связных компонент                   */
	/************************************************************************/

	void connectedComponents(InputOutputArray src_img,OutputArray dst_labels, Blobs& blobs,int connectivity /*= 4*/,int algorithm /*= CC_FLOODFILL*/, int minarea /*= 0*/, int maxarea /*= INT_MAX*/)
	{
		/*входной массив толко CV_8UC1*/
		CV_Assert( !src_img.empty() && src_img.type() == CV_8UC1);
		/*связность может быть либо 4 либо 8*/
		CV_Assert(connectivity == 4 || connectivity == 8);
		if(algorithm == CCL_FLOODFILL)
		{
			ccl_ff(src_img,dst_labels,blobs,connectivity,minarea,maxarea);
		}
		else //CC_SAUF
		{
			ccl_sauf(src_img,dst_labels,blobs,connectivity,minarea,maxarea);
		}
	}



}
