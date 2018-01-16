
#include <OpenNI.h>
#include "FrameReader.h"
#include "ImageRecorder.h"
#include "Ergonomics.h"

#include "opencv2\opencv.hpp"

#include <string>
#include <iostream>

typedef cv::Point_<uint8_t> Pixel;

using namespace std;
using namespace cv;
using namespace ergonomics;
using namespace openni;


int main(int argc, char** argv) {
	OpenNI::initialize();
	puts("Asus Xtion Pro initialization...");
	Device device;
	if (device.open(openni::ANY_DEVICE) != 0)
	{
		puts("Device not found !");
		OpenNI::shutdown();
		return -1;
	}
	puts("Asus Xtion Pro opened");

	VideoStream depth, color;
	color.create(device, SENSOR_COLOR);
	color.start();

	depth.create(device, SENSOR_DEPTH);
	depth.start();

	VideoMode paramvideo;
	paramvideo.setResolution(COLS, ROWS);
	paramvideo.setFps(FPS);
	paramvideo.setPixelFormat(PIXEL_FORMAT_DEPTH_100_UM);

	depth.setVideoMode(paramvideo);

	paramvideo.setPixelFormat(PIXEL_FORMAT_RGB888);

	color.setVideoMode(paramvideo);

	// Otherwise, the streams can be synchronized with a reception in the order of our choice :
	device.setDepthColorSyncEnabled(true);
	device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);

	VideoStream** stream = new VideoStream*[2];
	stream[0] = &depth;
	stream[1] = &color;
	puts("Kinect initialization completed");

	puts("Continue? (y/n)");
	string s;
	cin >> s;

	if (s != "y") {
		puts("Shutdown OpenNI");
		depth.stop();
		depth.destroy();
		color.stop();
		color.destroy();
		device.close();
		OpenNI::shutdown();
		return 1;
	}

	util::ImageRecorder recorder(device, stream);
	/*recorder.addDepthOperation(ImageOperation([](Mat& img) -> void {
		Scalar imgMean = mean(img);
		img.setTo(0, (img < 5000) | (img > imgMean[0]));
	}));*/
	recorder.run();

	depth.stop();
	depth.destroy();
	color.stop();
	color.destroy();
	device.close();

	puts("Shutdown OpenNI");

	OpenNI::shutdown();

	return 0;
}
