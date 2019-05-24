#include "ccl.h"

namespace ccl{
	using namespace cv;
	using namespace std;

	void freeBlobs(Blobs& blobs)
	{
		auto iter = blobs.begin();
		while(iter != blobs.end())
		{
			free(*(iter++));
		}
		blobs.clear();
	}

}
