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
	using namespace cv;
	using namespace std;

//! connected components algorithm output formats
enum { CC_32_STAT_LEFT   = 0,
	CC_32_STAT_TOP    = 1,
	CC_32_STAT_WIDTH  = 2,
	CC_32_STAT_HEIGHT = 3,
	CC_32_STAT_AREA   = 4,
	CC_32_STAT_PIX = 5,
	CC_32_STAT_MAX    = 6
};

enum { CC_64_STAT_M10   = 0,
	CC_64_STAT_M01    = 1,
	CC_64_STAT_M20  = 2,
	CC_64_STAT_M02 = 3,
	CC_64_STAT_M11   = 4,
	CC_64_STAT_MAX    = 5
};

int connectedComponents_sauf(InputArray image, OutputArray labels,
									 int connectivity = 8, int ltype = CV_32S);

int connectedComponentsWithStats_sauf(InputArray image, OutputArray labels,
											  OutputArray stats_32s, OutputArray stat_64s,
											  int connectivity = 8, int ltype = CV_32S);

}
