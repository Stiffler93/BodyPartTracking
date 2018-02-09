#ifndef BODY_PART_DETECTOR
#define BODY_PART_DETECTOR

#include "DecisionForest.h"
#include "opencv2\opencv.hpp"

#define LOC_HEAD 0
#define LOC_NECK 1
#define LOC_L_SHOULDER 2
#define LOC_R_SHOULDER 3
#define LOC_STERNUM 4
#define LOC_NUMBER 5

namespace tree {

	enum TrackingStatus {
		TRACKED, INACCURATE, UNTRACKED
	};

	typedef struct BodyPartLocation {
		int type, row = 0, col = 0, depth = 0;
		float accuracy;
	} BodyPartLocation;

	typedef struct BodyPartLocations {
		BodyPartLocation locs[LOC_NUMBER];
	} BodyPartLocations;

	class BodyPartDetector {
	private:
		tree::DecisionForest decForest;
		BodyPartLocation locs[LOC_NUMBER];
		tree::Record** featureMatrix;
		TrackingStatus status = UNTRACKED;
	public:
		BodyPartDetector();
		BodyPartDetector(tree::DecisionForest& decForest);
		~BodyPartDetector();
		BodyPartLocations getBodyPartLocations(cv::Mat& subject, bool isSubject = false);
		TrackingStatus state();
	};
}


#endif