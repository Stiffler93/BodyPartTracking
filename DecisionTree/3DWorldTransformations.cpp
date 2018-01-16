#include "3DWorldTransformations.h"

using namespace cv;

world::CoordinateTransformator* world::CoordinateTransformator::instance = NULL;

world::CoordinateTransformator::CoordinateTransformator()
{
}

world::CoordinateTransformator::~CoordinateTransformator()
{
}

void world::CoordinateTransformator::init(openni::VideoStream* videoStream)
{
	stream = videoStream;
}

cv::Point3d world::CoordinateTransformator::transformToWorldSpace(int row, int col, ushort depth)
{
	if (stream == NULL)
		throw std::exception("You have to call the init() method in advance!");

	Point3f p;
	openni::CoordinateConverter::convertDepthToWorld(*stream, col, row, depth, &(p.x), &(p.y), &(p.z));

	return p;
}
