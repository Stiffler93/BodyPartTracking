
#include "Types.h"
#include <functional>
#include "opencv2\opencv.hpp"

#define FPS 30
#define TRACK_FPS 2
#define TRACK_SECS 10
#define ROWS 240
#define COLS 320

#define PIXEL_MIN_DIFF 500
#define PIXEL_MAX_DIFF 3500
#define EROSION_SIZE 5
#define MAX_DISTANCE 20000
#define MIN_DISTANCE 5000
#define MIN_RANGE 5000
#define VIEW_ACCURACY 1000
#define DIST_INERTIA 8

typedef void(*imgOperator)(cv::Mat& img);
typedef std::function<void(cv::Mat& img)> imgListener;

namespace ergonomics {

#pragma once
	class DrawSkeleton {
	public:
		static BodyParts processBinaryImage(cv::Mat& img);
	private:
		static DrawSkeleton* _instance;
		DrawSkeleton();
		DrawSkeleton(const DrawSkeleton&);
		~DrawSkeleton();
	};

#pragma once
	class BodyClassification {
	public:
		void processFrame(cv::Mat& img);
		void showSkeleton(bool b);
		static BodyClassification* getInstance()
		{
			if (!_instance)
				_instance = new BodyClassification();
			return _instance;
		}
		static void release();
	private:
		static BodyClassification* _instance;
		BodyClassification();
		BodyClassification(const BodyClassification&);
		~BodyClassification();

		cv::Mat* image, *skeleton = NULL, *border = NULL;

		bool showSkeletonFlag = false;

		void skeletonize(cv::Mat& img);
	};
	
#pragma once
	class TrackingDistance {
	public:
		int getMinDist();
		int getMaxDist();
		cv::Mat getViewMask();
		bool postFrame(cv::Mat& frame);
		cv::Mat getModel();
		void getProcessedFrame(cv::Mat& img);

		void showModel(bool b);
		void showChanges(bool b);
		void showMovement(bool b);

		static TrackingDistance* getInstance()
		{
			if (!_instance)
				_instance = new TrackingDistance();
			return _instance;
		}

		static void release();
	private:
		int minDist = MIN_DISTANCE, maxDist = MAX_DISTANCE;
		int nthFrame, numFrames, frameCounter;
		int lastIndex = 0, totalFrames, distLastIndex = 0;
		int* lastMaxs;
		cv::Mat* lastFrames;
		cv::Mat* weightedModel, * currentFrame;
		cv::Mat* movingMask, * staticMask;
		cv::Mat* tmp1, * tmp2;
		cv::Mat* viewMask;

		cv::Mat* testMovement = NULL, * testModel = NULL;

		bool showModelFlag = false, showChangesFlag = false, showMovementFlag = false;

		static TrackingDistance* _instance;
		TrackingDistance();
		TrackingDistance(const TrackingDistance&);
		~TrackingDistance();

		void calcModel();
		void calcDifferences();
		void calcNewDists();
	};

#pragma once
	class ImageOperation
	{
	public:
		ImageOperation(imgOperator imgOp);
		~ImageOperation();

		void apply(cv::Mat& img);

	private:
		imgOperator imgOp;
	};

#pragma once
	class ImageListener
	{
	public:
		ImageListener(imgListener listener);
		~ImageListener();
		void trigger(cv::Mat& image);
	private:
		imgListener listener;
	};
}
