#include "ccl_sauf.h"

namespace ccl{
    namespace connectedcomponents{
	using namespace cv;
    struct NoOp{
        NoOp(){
        }
        void init(int /*labels*/){
        }
        inline
        void operator()(int r, int c, int l,int pix){
            (void) r;
            (void) c;
            (void) l;
			(void) pix;
        }
        void finish(){}
    };
    struct Point2ui64{
        uint64 x, y;
        Point2ui64(uint64 _x, uint64 _y):x(_x), y(_y){}
    };

    struct CCStatsOp{
        const _OutputArray* _mstats_32s;
        cv::Mat stats_32s;
        const _OutputArray* _mstats_64s;
        cv::Mat stats_64s;
        /*std::vector<Point2ui64> integrals;*/

        CCStatsOp(OutputArray _stats_32s, OutputArray _stats_64s): _mstats_32s(&_stats_32s), _mstats_64s(&_stats_64s)
		{
        }
        inline
        void init(int nlabels)
		{
            _mstats_32s->create(cv::Size(CC_32_STAT_MAX, nlabels), cv::DataType<int>::type);
            stats_32s = _mstats_32s->getMat();
			const int char_ncols = CC_64_STAT_MAX*sizeof(__int64);
			_mstats_64s->create(cv::Size(char_ncols, nlabels), cv::DataType<char>::type);
			stats_64s = _mstats_64s->getMat();
			
            for(int l = 0; l < nlabels; ++l)
			{
                memset((void*)stats_64s.ptr(l),0,char_ncols);
				int *row = (int*)stats_32s.ptr(l);
				row[CC_32_STAT_LEFT] = INT_MAX;
                row[CC_32_STAT_TOP] = INT_MAX;
                row[CC_32_STAT_WIDTH] = INT_MIN;
                row[CC_32_STAT_HEIGHT] = INT_MIN;
                row[CC_32_STAT_AREA] = 0;
            }
        }
        void operator()(int r, int c, int l,int pix)
		{
            int *row = (int*)stats_32s.ptr(l);
            row[CC_32_STAT_LEFT] = MIN(row[CC_32_STAT_LEFT], c);
			row[CC_32_STAT_TOP] = MIN(row[CC_32_STAT_TOP], r);
            row[CC_32_STAT_WIDTH] = MAX(row[CC_32_STAT_WIDTH], c);
            row[CC_32_STAT_HEIGHT] = MAX(row[CC_32_STAT_HEIGHT], r);
            row[CC_32_STAT_AREA]++;
			row[CC_32_STAT_PIX] = pix;
			//////////////////////////////////////////////////////////////////////////
			__int64* row64 = (__int64*)stats_64s.ptr(l);
			row64[CC_64_STAT_M10] += c;
			row64[CC_64_STAT_M01] += r;
			row64[CC_64_STAT_M20] += c*c;
			row64[CC_64_STAT_M02] += r*r;
			row64[CC_64_STAT_M11] += c*r;


        }
        void finish()
		{
            for(int l = 0; l < stats_32s.rows; ++l)
			{
                int *row = (int*)stats_32s.ptr(l);
                row[CC_32_STAT_WIDTH] = row[CC_32_STAT_WIDTH] - row[CC_32_STAT_LEFT] + 1;
                row[CC_32_STAT_HEIGHT] = row[CC_32_STAT_HEIGHT] - row[CC_32_STAT_TOP] + 1;
            }
        }
    };

    //Find the root of the tree of node i
    template<typename LabelT>
    inline static
    LabelT findRoot(const LabelT *P, LabelT i){
        LabelT root = i;
        while(P[root] < root)
		{
            root = P[root];
        }
        return root;
    }

    //Make all nodes in the path of node i point to root
    template<typename LabelT>
    inline static
    void setRoot(LabelT *P, LabelT i, LabelT root){
        while(P[i] < i)
		{
            LabelT j = P[i];
            P[i] = root;
            i = j;
        }
        P[i] = root;
    }

    //Find the root of the tree of the node i and compress the path in the process
    template<typename LabelT>
    inline static
    LabelT find(LabelT *P, LabelT i){
        LabelT root = findRoot(P, i);
        setRoot(P, i, root);
        return root;
    }

    //unite the two trees containing nodes i and j and return the new root
    template<typename LabelT>
    inline static
    LabelT set_union(LabelT *P, LabelT i, LabelT j){
        LabelT root = findRoot(P, i);
        if(i != j)
		{
            LabelT rootj = findRoot(P, j);
            if(root > rootj)
			{
                root = rootj;
            }
            setRoot(P, j, root);
        }
        setRoot(P, i, root);
        return root;
    }

    //Flatten the Union Find tree and relabel the components
    template<typename LabelT>
    inline static
    LabelT flattenL(LabelT *P, LabelT length){
        LabelT k = 1;
        for(LabelT i = 1; i < length; ++i){
            if(P[i] < i){
                P[i] = P[P[i]];
            }else{
                P[i] = k; k = k + 1;
            }
        }
        return k;
    }

    //Based on "Two Strategies to Speed up Connected Components Algorithms", the SAUF (Scan array union find) variant
    //using decision trees
    //Kesheng Wu, et al
    //Note: rows are encoded as position in the "rows" array to save lookup times
    //reference for 4-way: {{-1, 0}, {0, -1}};//b, d neighborhoods
    const int G4[2][2] = {{1, 0}, {0, -1}};//b, d neighborhoods
    //reference for 8-way: {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}};//a, b, c, d neighborhoods
    const int G8[4][2] = {{1, -1}, {1, 0}, {1, 1}, {0, -1}};//a, b, c, d neighborhoods

	template<typename LabelT, typename PixelT, typename StatsOp = NoOp >
	struct Labeling8Impl
	{
		LabelT operator()(const cv::Mat &I, cv::Mat &L, StatsOp &sop)
		{
			CV_Assert(L.rows == I.rows);
			CV_Assert(L.cols == I.cols);
			const int rows = L.rows;
			const int cols = L.cols;
			const int a = 0;
			const int b = 1;
			const int c = 2;
			const int d = 3;
			//A quick and dirty upper bound for the maximimum number of labels.  The 4 comes from
			//the fact that a 3x3 block can never have more than 4 unique labels for both 4 & 8-way
			/*const size_t Plength = 4 * (size_t(rows + 3 - 1)/3) * (size_t(cols + 3 - 1)/3);*/
			const size_t Plength = rows*cols;
			LabelT *P = (LabelT *) fastMalloc(sizeof(LabelT) * Plength);
			P[0] = 0;
			LabelT lunique = 1;
			//scanning phase
			for(int r_i = 0; r_i < rows; ++r_i)
			{
				LabelT *Lrow = (LabelT *)(L.data + L.step.p[0] * r_i);
				LabelT *Lrow_prev = (LabelT *)(((char *)Lrow) - L.step.p[0]);
				const PixelT *Irow = (PixelT *)(I.data + I.step.p[0] * r_i);
				const PixelT *Irow_prev = (const PixelT *)(((char *)Irow) - I.step.p[0]);
				LabelT *Lrows[2] = { Lrow, Lrow_prev };
				const PixelT *Irows[2] = { Irow, Irow_prev };
				const bool T_a_r = (r_i - G8[a][0]) >= 0;
				const bool T_b_r = (r_i - G8[b][0]) >= 0;
				const bool T_c_r = (r_i - G8[c][0]) >= 0;
				for(int c_i = 0; Irows[0] != Irow + cols; ++Irows[0], c_i++)
				{
					if(!*Irows[0])
					{
						Lrow[c_i] = 0;
						continue;
					}
					Irows[1] = Irow_prev + c_i;
					Lrows[0] = Lrow + c_i;
					Lrows[1] = Lrow_prev + c_i;
					const bool T_a = T_a_r && (c_i + G8[a][1]) >= 0   && *(Irows[G8[a][0]] + G8[a][1]) == *Irows[0];
					const bool T_b = T_b_r                            && *(Irows[G8[b][0]] + G8[b][1]) == *Irows[0];
					const bool T_c = T_c_r && (c_i + G8[c][1]) < cols && *(Irows[G8[c][0]] + G8[c][1]) == *Irows[0];
					const bool T_d =          (c_i + G8[d][1]) >= 0   && *(Irows[G8[d][0]] + G8[d][1]) == *Irows[0];
					//decision tree
					if(T_b)
					{
						//copy(b)
						*Lrows[0] = *(Lrows[G8[b][0]] + G8[b][1]);
					}
					else
					{//not b
						if(T_c)
						{
							if(T_a)
							{
								//copy(c, a)
								*Lrows[0] = set_union(P, *(Lrows[G8[c][0]] + G8[c][1]), *(Lrows[G8[a][0]] + G8[a][1]));
							}
							else
							{
								if(T_d)
								{
									//copy(c, d)
									*Lrows[0] = set_union(P, *(Lrows[G8[c][0]] + G8[c][1]), *(Lrows[G8[d][0]] + G8[d][1]));
								}
								else
								{
									//copy(c)
									*Lrows[0] = *(Lrows[G8[c][0]] + G8[c][1]);
								}
							}
						}
						else
						{//not c
							if(T_a)
							{
								//copy(a)
								*Lrows[0] = *(Lrows[G8[a][0]] + G8[a][1]);
							}
							else
							{
								if(T_d)
								{
									//copy(d)
									*Lrows[0] = *(Lrows[G8[d][0]] + G8[d][1]);
								}
								else
								{
									//new label
									*Lrows[0] = lunique;
									P[lunique] = lunique;
									++lunique;
								}
							}
						}
					}
				}
			}

			//analysis
			LabelT nLabels = flattenL(P, lunique);
			sop.init(nLabels);

			for(int r_i = 0; r_i < rows; ++r_i)
			{
				LabelT *Lrow = (LabelT *)(L.ptr(r_i));
				PixelT* Irow = (PixelT *)(I.ptr(r_i));
				for(int c_i = 0; c_i < cols; ++c_i)
				{
					const LabelT l = P[Lrow[c_i]];
					Lrow[c_i] = l;
					if(l)
					{
						sop(r_i, c_i, l,Irow[c_i]);
					}

				}
			}

			sop.finish();
			fastFree(P);

			return nLabels;
		}//End function Labeling8Impl operator()

    };//End struct Labeling8Impl

	template<typename LabelT, typename PixelT, typename StatsOp = NoOp >
	struct Labeling4Impl
	{
		LabelT operator()(const cv::Mat &I, cv::Mat &L, StatsOp &sop)
		{
			CV_Assert(L.rows == I.rows);
			CV_Assert(L.cols == I.cols);
			const int rows = L.rows;
			const int cols = L.cols;
			const int b = 0;
			const int d = 1;
			//A quick and dirty upper bound for the maximimum number of labels.  The 4 comes from
			//the fact that a 3x3 block can never have more than 4 unique labels for both 4 & 8-way
			/*const size_t Plength = 4 * (size_t(rows + 3 - 1)/3) * (size_t(cols + 3 - 1)/3);*/
			const size_t Plength = rows*cols;
			LabelT *P = (LabelT *) fastMalloc(sizeof(LabelT) * Plength);
			P[0] = 0;
			LabelT lunique = 1;
			//scanning phase
			for(int r_i = 0; r_i < rows; ++r_i)
			{
				LabelT *Lrow = (LabelT *)(L.data + L.step.p[0] * r_i);
				LabelT *Lrow_prev = (LabelT *)(((char *)Lrow) - L.step.p[0]);
				const PixelT *Irow = (PixelT *)(I.data + I.step.p[0] * r_i);
				const PixelT *Irow_prev = (const PixelT *)(((char *)Irow) - I.step.p[0]);
				LabelT *Lrows[2] = { Lrow, Lrow_prev };
				const PixelT *Irows[2] = { Irow, Irow_prev };
				const bool T_b_r = (r_i - G4[b][0]) >= 0;
				for(int c_i = 0; Irows[0] != Irow + cols; ++Irows[0], c_i++)
				{
					if(!*Irows[0])
					{
						Lrow[c_i] = 0;
						continue;
					}
					Irows[1] = Irow_prev + c_i;
					Lrows[0] = Lrow + c_i;
					Lrows[1] = Lrow_prev + c_i;
					const bool T_b = T_b_r                            && *(Irows[G4[b][0]] + G4[b][1]) == *Irows[0];
					const bool T_d =          (c_i + G4[d][1]) >= 0   && *(Irows[G4[d][0]] + G4[d][1]) == *Irows[0];
					if(T_b)
					{
						if(T_d)
						{
							//copy(d, b)
							*Lrows[0] = set_union(P, *(Lrows[G4[d][0]] + G4[d][1]), *(Lrows[G4[b][0]] + G4[b][1]));
						}
						else
						{
							//copy(b)
							*Lrows[0] = *(Lrows[G4[b][0]] + G4[b][1]);
						}
					}
					else
					{
						if(T_d)
						{
							//copy(d)
							*Lrows[0] = *(Lrows[G4[d][0]] + G4[d][1]);
						}
						else
						{
							//new label
							*Lrows[0] = lunique;
							P[lunique] = lunique;
							++lunique;
						}
					}
				}
			}

			//analysis
			LabelT nLabels = flattenL(P, lunique);
			sop.init(nLabels);

			for(int r_i = 0; r_i < rows; ++r_i)
			{
				LabelT *Lrow = (LabelT *)(L.ptr(r_i));
				PixelT* Irow = (PixelT *)(I.ptr(r_i));
				for(int c_i = 0; c_i < cols; ++c_i)
				{
					const LabelT l = P[Lrow[c_i]];
					Lrow[c_i] = l;
					if(l)
					{
						sop(r_i, c_i, l,Irow[c_i]);
					}
					
				}
			}

			sop.finish();
			fastFree(P);

			return nLabels;
		}//End function LabelingImpl operator()
	};

}//end namespace connectedcomponents

//L's type must have an appropriate depth for the number of pixels in I
template<typename StatsOp>
static
int connectedComponents_sub1(const cv::Mat &I, cv::Mat &L, int connectivity, StatsOp &sop){
    CV_Assert(L.channels() == 1 && I.channels() == 1);
    CV_Assert(connectivity == 8 || connectivity == 4);
    int lDepth = L.depth();
    int iDepth = I.depth();
    /*using connectedcomponents::LabelingImpl;*/
	using connectedcomponents::Labeling4Impl;
	using connectedcomponents::Labeling8Impl;
    //warn if L's depth is not sufficient?

    CV_Assert(iDepth == CV_8U || iDepth == CV_8S);

// 	if(lDepth == CV_8U)
// 	{
// 		return (int)(connectivity == 4 ? Labeling4Impl<uchar, uchar, StatsOp>()(I, L, sop) : Labeling8Impl<uchar, uchar, StatsOp>()(I, L, sop));
// 	}
// 	else if(lDepth == CV_16U)
// 	{
// 		return (int)(connectivity == 4 ? Labeling4Impl<ushort, uchar, StatsOp>()(I, L, sop) : Labeling8Impl<ushort, uchar, StatsOp>()(I, L, sop));
// 	}
	/*else*/ if(lDepth == CV_32S)
	{
		//note that signed types don't really make sense here and not being able to use unsigned matters for scientific projects
		//OpenCV: how should we proceed?  .at<T> typechecks in debug mode
		return (int)(connectivity == 4 ? Labeling4Impl<int, uchar, StatsOp>()(I, L, sop) : Labeling8Impl<int, uchar, StatsOp>()(I, L, sop));
	}

    CV_Error(CV_StsUnsupportedFormat, "unsupported label/image type");
    return -1;
}


int connectedComponents_sauf(InputArray _img, OutputArray _labels, int connectivity, int ltype){
	const cv::Mat img = _img.getMat();
	_labels.create(img.size(), CV_MAT_DEPTH(ltype));
	cv::Mat labels = _labels.getMat();
	connectedcomponents::NoOp sop;
// 	if(ltype == CV_16U)
// 	{
// 		return connectedComponents_sub1(img, labels, connectivity, sop);
// 	}
	/*else */if(ltype == CV_32S)
	{
		return connectedComponents_sub1(img, labels, connectivity, sop);
	}
	else
	{
		CV_Error(CV_StsUnsupportedFormat, "the type of labels must be 16u or 32s");
		return 0;
	}
}

// InputArray image, OutputArray labels,
// 	OutputArray stats_32s, OutputArray stat_64s,
// 	int connectivity = 8, int ltype = CV_3
int connectedComponentsWithStats_sauf(InputArray _img, OutputArray _labels, OutputArray stats_32s,
										   OutputArray stats_64s, int connectivity, int ltype)
{
	const cv::Mat img = _img.getMat();
	_labels.create(img.size(), CV_MAT_DEPTH(ltype));
	cv::Mat labels = _labels.getMat();
	connectedcomponents::CCStatsOp sop(stats_32s, stats_64s);
// 	if(ltype == CV_16U)
// 	{
// 		return connectedComponents_sub1(img, labels, connectivity, sop);
// 	}
	/*else */if(ltype == CV_32S)
	{
		return connectedComponents_sub1(img, labels, connectivity, sop);
	}
	else
	{
		CV_Error(CV_StsUnsupportedFormat, "the type of labels must be 16u or 32s");
		return 0;
	}
}

}


