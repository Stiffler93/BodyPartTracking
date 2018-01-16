#include "FrameReader.h"
#include "OniSampleUtilities.h"
#include "opencv2\opencv.hpp"

using namespace cv;
using namespace openni;

static FrameReader* self = NULL;

FrameReader::FrameReader(Device& device, VideoStream& stream) :
	stream(stream), device(device)
{
	self = this;
	recordFrame = false;
	printf("FrameReader is instantiated\n");
}

FrameReader::~FrameReader()
{
	delete[] texMap;

	self = NULL;

	if (streams != NULL)
	{
		delete[] streams;
	}
}

void FrameReader::displayCV() {
	if (!stream.isValid()) {
		printf("Stream isn't valid -> return\n");
		return;
	}

	if (!recordFrame)
		return;

	stream.readFrame(&frame);

	if (!frame.isValid()) {
		printf("Frame isn't valid!\n");
		return;
	}

	cv::Mat cv_depth_frame(cv::Size(640, 480), CV_16UC1, NULL);
	cv::namedWindow("Depth", CV_WINDOW_AUTOSIZE);

	cv_depth_frame.data = (uchar*)frame.getData();
	cv::imshow("Depth", cv_depth_frame);
}

void FrameReader::printFrame()
{
	printf("Call printFrame..\n");

	if (&frame != NULL) {
		printf("Print:\n");

		VideoMode mode = frame.getVideoMode();
		printf("Resolution: %d:%d\n", mode.getResolutionX(), mode.getResolutionY());
		printf("FPS: %d\n", mode.getFps());
		printf("Format: %d\n\n", mode.getPixelFormat());

		printf("Image Pixels: \n");
		
		const openni::DepthPixel* pDepthRow = (const openni::DepthPixel*)frame.getData();
		int rowSize = frame.getStrideInBytes() / sizeof(openni::DepthPixel);
		int min = INT_MAX;
		int max = INT_MIN;

		for (int y = 0; y < frame.getHeight(); ++y)
		{
			const openni::DepthPixel* pDepth = pDepthRow;

			for (int x = 0; x < frame.getWidth(); ++x, ++pDepth)
			{
				if (*pDepth != 0)
				{
					if (*pDepth < min)
						min = *pDepth;

					if (*pDepth > max)
						max = *pDepth;
				}
			}

			pDepthRow += rowSize;
		}

		printf("Min: >%d<, Max: >%d<.\n", min, max);
	}
}

void FrameReader::glutKey(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key) {
	case 27:
		cv::destroyWindow("Depth");
		stream.stop();
		stream.destroy();
		device.close();
		openni::OpenNI::shutdown();

		exit(1);
	case 'r':
		recordFrame = true;
		break;
	case 'p':
		printFrame();
		break;
	}
}
