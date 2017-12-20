
#include "OpenNI.h"
#include "opencv2\opencv.hpp"
#include "DecTree.h"
#include <string>
#include <sstream>

using namespace openni;
using namespace cv;
using namespace std;

typedef int Stream;

#define DEPTH_STREAM 0
#define COLOR_STREAM 1

#define WINDOW_DEPTH "Depth Stream"

namespace util {

#pragma once
	class ImageRecorder
	{
	public:
		ImageRecorder(Device& device, VideoStream** streams, tree::Node* decisionTree, function<void(Mat& img, Mat& classifiedImg, 
			tree::Dataset**& featureMatrix, tree::Node* decTree)> classification);
		~ImageRecorder();

		void run();

	private:
		Device& device;
		VideoStream** streams;
		VideoFrameRef depthFrame;
		Mat depthMat;
		Mat classifiedMat;
		tree::Dataset** featureMatrix;
		tree::Node* decTree;

		int counter = 0;
		int numImage = 1;

		int img = 1;

		function<void(Mat& img, Mat& classifiedMat, tree::Dataset**& featureMatrix, tree::Node* decTree)> classification;

		void createWindows();
		void initCV();
		void readStreams();
		void readFrame(VideoStream& stream, VideoFrameRef& frame);
		void processImage(Mat& img, Stream stream, const char* window);
	};
}

