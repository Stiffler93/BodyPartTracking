#include "ImageRecorder.h"

using namespace util;

//typedef cv::Point_<uint8_t> Pixel;

ImageRecorder::ImageRecorder(Device & device, VideoStream ** streams/*, tree::Node* decisionTree*/)
	:device(device), streams(streams)
{
}

ImageRecorder::~ImageRecorder()
{
}

void ImageRecorder::createWindows() {
	namedWindow(WINDOW_DEPTH, CV_WINDOW_AUTOSIZE);
}

void ImageRecorder::initCV()
{
	depthMat.create(MAT_ROWS, MAT_COLS, CV_16UC1);
	/*colorMat.create(MAT_ROWS, MAT_COLS, CV_8UC3);
	depthMatRecorded.create(MAT_ROWS, MAT_COLS, CV_16UC1);
	colorMatRecorded.create(MAT_ROWS, MAT_COLS, CV_8UC3);*/
}

void ImageRecorder::readStreams()
{
	int stream;
	int key = -1;
	while (device.isValid() && key != 27) {
		OpenNI::waitForAnyStream(streams, 2, &stream);

		switch (stream) {
		case DEPTH_STREAM:
			readFrame(*streams[DEPTH_STREAM], depthFrame);
			depthMat.data = (uchar*)depthFrame.getData();
			processImage(depthMat, DEPTH_STREAM, WINDOW_DEPTH);
			break;
		}

		key = waitKey(10);
		handleKey(key);
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

	imshow(window, img);
}

void ImageRecorder::handleKey(int key)
{
	if (key != RECORD && key != FREE && key != SAVE) return;

	/*switch (key) {
	case RECORD:
		puts("record Image");
		depthMat.copyTo(depthMatRecorded);
		colorMat.copyTo(colorMatRecorded);
		break;
	case FREE:
		puts("free Image");
		depthMatRecorded = Scalar(0);
		colorMatRecorded = Scalar(0, 0, 0);
		break;
	case SAVE:
		puts("save Image");
		save();
		break;
	}*/
}

//void ImageRecorder::save()
//{
//	int length = 1;
//	for (int n = img/10; n >= 1; n /= 10)
//		length++;
//
//	stringstream s;
//
//	for (int i = 0; i < 3 - length; i++) {
//		s << '0';
//	}
//	s << img;
//
//	printf("String: >%s<\n", s.str().c_str());
//
//	String depthImg = IMG_FOLDER + s.str() + "_depth.png";
//	String colorImg = IMG_FOLDER + s.str() + "_color.png";
//
//	printf("Save Depth Img to %s\n", depthImg.c_str());
//	printf("Save Color Img to %s\n", colorImg.c_str());
//
//	bool b1 = imwrite(depthImg, depthMatRecorded);
//	bool b2 = imwrite(colorImg, colorMatRecorded);
//
//	if (b1) puts("Saved Depth Image");
//	else puts("Couldn't save Depth Image");
//
//	if (b2) puts("Saved Color Image");
//	else puts("Couldn't save Color Image");
//
//	if (b1 || b2)
//		img++;
//}

void ImageRecorder::run()
{
	if (!device.isValid()) {
		puts("Device isn't valid -> return");
		return;
	}

	createWindows();
	initCV();
	readStreams();

	destroyWindow(WINDOW_DEPTH);
}


