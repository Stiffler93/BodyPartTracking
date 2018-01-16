#include "Ergonomics.h"

using namespace ergonomics;
using namespace cv;

ImageOperation::ImageOperation(imgOperator imgOp)
	:imgOp(imgOp)
{
}

ImageOperation::~ImageOperation()
{
}

void ImageOperation::apply(Mat& img)
{
	imgOp(img);
}

ImageListener::ImageListener(imgListener listener)
	:listener(listener)
{
}

ImageListener::~ImageListener()
{
}

void ImageListener::trigger(Mat& image)
{
	listener(image);
}
