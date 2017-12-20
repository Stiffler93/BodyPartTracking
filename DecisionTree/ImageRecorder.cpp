#include "ImageRecorder.h"
#include "TreeConstants.h"

using namespace util;

//typedef cv::Point_<uint8_t> Pixel;

ImageRecorder::ImageRecorder(Device & device, VideoStream ** streams, tree::Node* decisionTree, function<void(Mat& img, Mat& classifiedImg, tree::Dataset**& featureMatrix, tree::Node* decTree)> classification)
	:device(device), streams(streams), decTree(decisionTree), classification(classification) {
	featureMatrix = new tree::Dataset*[MAX_ROW];
	for (int i = 0; i < MAX_ROW; i++)
		featureMatrix[i] = new tree::Dataset[MAX_COL];
}

ImageRecorder::~ImageRecorder() {}

void ImageRecorder::createWindows() { namedWindow(WINDOW_DEPTH, CV_WINDOW_AUTOSIZE); }

void ImageRecorder::initCV() { depthMat.create(MAX_ROW, MAX_COL, CV_16U); classifiedMat.create(MAX_ROW, MAX_COL, CV_8UC3); }

void ImageRecorder::readStreams()
{
	int stream;
	int key = -1;
	while (device.isValid() && key != 27) {
		OpenNI::waitForAnyStream(streams, 1, &stream);

		switch (stream) {
		case DEPTH_STREAM:
			readFrame(*streams[DEPTH_STREAM], depthFrame);
			depthMat.data = (uchar*)depthFrame.getData();
			processImage(depthMat, DEPTH_STREAM, WINDOW_DEPTH);
			break;
		}

		key = waitKey(10);

		if (key == 27)
			break;
	}
}

void ImageRecorder::readFrame(VideoStream& stream, VideoFrameRef& frame)
{
	stream.readFrame(&frame);

	if (!frame.isValid()) {
		puts("Frame wasn't valid -> return");
		return;
	}
}

void ImageRecorder::processImage(Mat& img, Stream stream, const char * window)
{
	classification(img, classifiedMat, featureMatrix, decTree);
	imshow(window, img);
	imshow("Classified Image", classifiedMat);
}

void ImageRecorder::run()
{
	if (!device.isValid()) {
		puts("Device isn't valid -> return");
		return;
	}

	createWindows();
	initCV();
	readStreams();

	destroyAllWindows();
}


