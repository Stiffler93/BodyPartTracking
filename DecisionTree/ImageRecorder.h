
#include "OpenNI.h"
#include "opencv2\opencv.hpp"
#include "DecTree.h"
#include "DecisionForest.h"
#include "BodyPartDetector.h"

typedef int Stream;

#define DEPTH_STREAM 0
#define COLOR_STREAM 1

#define WINDOW_DEPTH "Depth Stream"

namespace util {

#pragma once
	class ImageRecorder
	{
	public:
		ImageRecorder(openni::Device& device, openni::VideoStream** streams, tree::DecisionForest& decForest, std::function<void(cv::Mat& img, cv::Mat& classifiedImg,
			tree::Dataset**& featureMatrix, tree::DecisionForest& decForest, tree::BodyPartDetector& bpDetector)> classification);
		~ImageRecorder();

		void run();

	private:
		openni::Device& device;
		openni::VideoStream** streams;
		openni::VideoFrameRef depthFrame;
		cv::Mat depthMat;
		cv::Mat classifiedMat;
		tree::Dataset** featureMatrix;
		tree::DecisionForest& decForest;
		tree::BodyPartDetector bpDetector;

		int counter = 0;
		int numImage = 1;

		int img = 1;

		std::function<void(cv::Mat& img, cv::Mat& classifiedMat, tree::Dataset**& featureMatrix, tree::DecisionForest& decForest, tree::BodyPartDetector& bpDetector)> classification;

		void createWindows();
		void initCV();
		void readStreams();
		void readFrame(openni::VideoStream& stream, openni::VideoFrameRef& frame);
		void processImage(cv::Mat& img, Stream stream, const char* window);
	};
}

