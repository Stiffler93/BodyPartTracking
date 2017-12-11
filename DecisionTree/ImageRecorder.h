
#include "OpenNI.h"
#include "opencv2\opencv.hpp"
//#include "DecTree.h"
#include <string>
#include <sstream>

using namespace openni;
using namespace cv;
using namespace std;

typedef int Stream;

#define DEPTH_STREAM 0
#define COLOR_STREAM 1

#define MAT_ROWS 240
#define MAT_COLS 320

#define WINDOW_DEPTH "Depth Stream"

#define RECORD (int)'r'
#define FREE (int) 'f'
#define SAVE (int) 's'

#define IMG_FOLDER "C:\\Users\\Stefan\\Desktop\\Master\\Masterarbeit\\Programme\\BPT_TrainingImages\\"

namespace util {

#pragma once
	class ImageRecorder
	{
	public:
		ImageRecorder(Device& device, VideoStream** streams/*, tree::Node* decisionTree*/);
		~ImageRecorder();

		void run();

	private:
		Device& device;
		VideoStream** streams;
		VideoFrameRef depthFrame;
		Mat depthMat;

		int counter = 0;
		int numImage = 1;

		int img = 1;

		void createWindows();
		void initCV();
		void readStreams();
		void readFrame(VideoStream& stream, VideoFrameRef& frame);
		void processImage(Mat& img, Stream stream, const char* window);

		void handleKey(int key);
	};
}

