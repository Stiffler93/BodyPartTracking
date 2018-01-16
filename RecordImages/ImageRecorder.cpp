#include "ImageRecorder.h"

using namespace util;
using namespace openni;
using namespace cv;
using namespace ergonomics;

typedef Point_<uint8_t> Pixel;

ImageRecorder::ImageRecorder(Device & device, VideoStream ** streams)
	:device(device), streams(streams)
{
}

ImageRecorder::~ImageRecorder()
{
}

void ImageRecorder::addDepthOperation(ImageOperation imgOp)
{
	depthOperations.push_back(imgOp);
}

void ImageRecorder::addColorOperation(ImageOperation imgOp)
{
	colorOperations.push_back(imgOp);
}

void ImageRecorder::addDepthListener(ImageListener listener)
{
	depthListeners.push_back(listener);
}

void ImageRecorder::addColorListener(ImageListener listener)
{
	colorListeners.push_back(listener);
}

void ImageRecorder::createWindows() {
	namedWindow(WINDOW_DEPTH, CV_WINDOW_AUTOSIZE);
	namedWindow(WINDOW_COLOR, CV_WINDOW_AUTOSIZE);
}

void ImageRecorder::initCV()
{
	depthMat.create(MAT_ROWS, MAT_COLS, CV_16UC1);
	colorMat.create(MAT_ROWS, MAT_COLS, CV_8UC3);
	depthMatRecorded.create(MAT_ROWS, MAT_COLS, CV_16UC1);
	colorMatRecorded.create(MAT_ROWS, MAT_COLS, CV_8UC3);
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
			triggerListener(depthMat, DEPTH_STREAM);
			processImage(depthMat, DEPTH_STREAM, WINDOW_DEPTH);
			break;
		case COLOR_STREAM:
			readFrame(*streams[COLOR_STREAM], colorFrame);
			colorMat.data = (uchar*)colorFrame.getData();
			triggerListener(colorMat, COLOR_STREAM);
			processImage(colorMat, COLOR_STREAM, WINDOW_COLOR);
			break;
		}

		key = waitKey(10);
		handleKey(key);

		if (counter++ % 30 == 0) {
			ImageSaver::save(depthMat, "DepthImage", numImage);
			ImageSaver::save(colorMat, "ColorImage", numImage);
			counter = 1;
			numImage++;
		}
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

void ImageRecorder::triggerListener(Mat& img, Stream stream)
{
	switch (stream) {
	case DEPTH_STREAM:
		for (unsigned i = 0; i < depthListeners.size(); i++) {
			depthListeners[i].trigger(img);
		}
		break;
	case COLOR_STREAM:
		for (unsigned i = 0; i < colorListeners.size(); i++) {
			colorListeners[i].trigger(img);
		}
		break;
	}
}

void ImageRecorder::processImage(Mat& img, Stream stream, const char * window)
{
	switch (stream) {
	case DEPTH_STREAM:
		for (unsigned i = 0; i < depthOperations.size(); i++) {
			depthOperations[i].apply(img);
		}
		break;
	case COLOR_STREAM:
		for (unsigned i = 0; i < colorOperations.size(); i++) {
			colorOperations[i].apply(img);
		}
		break;
	}

	imshow(window, img);
}

void ImageRecorder::handleKey(int key)
{
	if (key != RECORD && key != FREE && key != SAVE) return;

	switch (key) {
	case RECORD:
		puts("record Image");
		depthMat.copyTo(depthMatRecorded);
		colorMat.copyTo(colorMatRecorded);
		imshow(WINDOW_DEPTH_RECORD, depthMatRecorded);
		imshow(WINDOW_COLOR_RECORD, colorMatRecorded);
		break;
	case FREE:
		puts("free Image");
		depthMatRecorded = Scalar(0);
		colorMatRecorded = Scalar(0, 0, 0);
		imshow(WINDOW_DEPTH_RECORD, depthMatRecorded);
		imshow(WINDOW_COLOR_RECORD, colorMatRecorded);
		break;
	case SAVE:
		puts("save Image");
		save();
		break;
	}
}

void ImageRecorder::save()
{
	int length = 1;
	for (int n = img/10; n >= 1; n /= 10)
		length++;

	std::stringstream s;

	for (int i = 0; i < 3 - length; i++) {
		s << '0';
	}
	s << img;

	printf("String: >%s<\n", s.str().c_str());

	String depthImg = IMG_FOLDER + s.str() + "_depth.png";
	String colorImg = IMG_FOLDER + s.str() + "_color.png";

	printf("Save Depth Img to %s\n", depthImg.c_str());
	printf("Save Color Img to %s\n", colorImg.c_str());

	bool b1 = imwrite(depthImg, depthMatRecorded);
	bool b2 = imwrite(colorImg, colorMatRecorded);

	if (b1) puts("Saved Depth Image");
	else puts("Couldn't save Depth Image");

	if (b2) puts("Saved Color Image");
	else puts("Couldn't save Color Image");

	if (b1 || b2)
		img++;
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

	destroyWindow(WINDOW_DEPTH);
	destroyWindow(WINDOW_COLOR);
	destroyWindow(WINDOW_DEPTH_RECORD);
	destroyWindow(WINDOW_COLOR_RECORD);
}

int ImageSaver::numImage = 0;

void util::ImageSaver::save(Mat image, std::string name, int num)
{
	if (name.empty()) {
		int length = 1;
		for (int n = numImage / 10; n >= 1; n /= 10)
			length++;

		std::stringstream s;

		s << IMG_FOLDER;
		s << "img";

		for (int i = 0; i < 3 - length; i++) {
			s << '0';
		}
		s << numImage;
		s << ".png";

		name = s.str();
		numImage++;
	}
	else if (num != 0) {
		int length = 1;
		for (int n = num / 10; n >= 1; n /= 10)
			length++;

		std::stringstream s;

		s << IMG_FOLDER;
		s << name;

		for (int i = 0; i < 3 - length; i++) {
			s << '0';
		}
		s << num;
		s << ".png";

		name = s.str();
	}
	
	bool b = imwrite(name, image);

	if (b) puts("Saved Image");
	else puts("Couldn't save Image");
}

ImageSaver::~ImageSaver()
{
}
