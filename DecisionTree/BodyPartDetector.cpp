#include "BodyPartDetector.h"
#include "Features.hpp"
#include <vector>

using namespace cv;
using namespace tree;
using std::string;
using std::vector;

tree::BodyPartDetector::BodyPartDetector()
{
	featureMatrix = new tree::Record*[MAX_ROW];
	for (int i = 0; i < MAX_ROW; i++)
		featureMatrix[i] = new tree::Record[MAX_COL];

	for (int i = 0; i < LOC_NUMBER; i++) {
		BodyPartLocation loc;
		loc.accuracy = -1;
		loc.type = i;
		locs[i] = loc;
	}
}

BodyPartDetector::BodyPartDetector(tree::DecisionForest & decForest) : decForest(decForest)
{
	featureMatrix = new tree::Record*[MAX_ROW];
	for (int i = 0; i < MAX_ROW; i++)
		featureMatrix[i] = new tree::Record[MAX_COL];

	for (int i = 0; i < LOC_NUMBER; i++) {
		BodyPartLocation loc;
		loc.accuracy = -1;
		loc.type = i;
		locs[i] = loc;
	}
}

BodyPartDetector::~BodyPartDetector()
{
	//for (int i = 0; i < MAX_ROW; i++)
	//	delete[] featureMatrix[i];
	//delete[] featureMatrix;
}

BodyPartLocations BodyPartDetector::getBodyPartLocations(cv::Mat & subject, bool isSubject)
{
	Mat classifiedMat;
	classifiedMat.create(subject.rows, subject.cols, CV_8UC3);

	classifiedMat = Scalar(Vec3b(127, 127, 127));

	featurizeImage(subject, featureMatrix, isSubject);
	Vec3b color;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			tree::Record set = featureMatrix[row][col];

			if (set.outcome == OTHER) {
				string outcome = decForest.classify(set);
				color = getBGR(outcome);

				classifiedMat.at<Vec3b>(row, col) = color;
			}
		}
	}

	cv::imshow("Classified Pixels", classifiedMat);

	int partCounter[LOC_NUMBER];
	BodyPartLocation locations[LOC_NUMBER];

	vector<Mat> channels(3);
	split(classifiedMat, channels);

	Mat masks[LOC_NUMBER];
	Vec3b col;

	col = getBGR(HEAD);
	masks[LOC_HEAD] = (channels[0] == col[0] & channels[1] == col[1] & channels[2] == col[2]);

	col = getBGR(NECK);
	masks[LOC_NECK] = (channels[0] == col[0] & channels[1] == col[1] & channels[2] == col[2]);

	col = getBGR(LEFT_SHOULDER);
	masks[LOC_L_SHOULDER] = (channels[0] == col[0] & channels[1] == col[1] & channels[2] == col[2]);

	col = getBGR(RIGHT_SHOULDER);
	masks[LOC_R_SHOULDER] = (channels[0] == col[0] & channels[1] == col[1] & channels[2] == col[2]);

	col = getBGR(STERNUM);
	masks[LOC_STERNUM] = (channels[0] == col[0] & channels[1] == col[1] & channels[2] == col[2]);

	if (status != TRACKED) {	// initialize positions

		for (int i = 0; i < LOC_NUMBER; i++) {
			BodyPartLocation loc;
			loc.accuracy = -1;
			loc.type = i;
			locations[i] = loc;
			partCounter[i] = 0;
		}

		for (int row = 0; row < classifiedMat.rows; row++) {
			for (int col = 0; col < classifiedMat.cols; col++) {
				if (masks[LOC_HEAD].at<uchar>(row, col) > 0) {
					locations[LOC_HEAD].row += row;
					locations[LOC_HEAD].col += col;
					partCounter[LOC_HEAD]++;
					continue;
				}
				else if (masks[LOC_NECK].at<uchar>(row, col) > 0) {
					locations[LOC_NECK].row += row;
					locations[LOC_NECK].col += col;
					partCounter[LOC_NECK]++;
					continue;
				}
				else if (masks[LOC_L_SHOULDER].at<uchar>(row, col) > 0) {
					locations[LOC_L_SHOULDER].row += row;
					locations[LOC_L_SHOULDER].col += col;
					partCounter[LOC_L_SHOULDER]++;
					continue;
				}
				else if (masks[LOC_R_SHOULDER].at<uchar>(row, col) > 0) {
					locations[LOC_R_SHOULDER].row += row;
					locations[LOC_R_SHOULDER].col += col;
					partCounter[LOC_R_SHOULDER]++;
					continue;
				}
				else if (masks[LOC_STERNUM].at<uchar>(row, col) > 0) {
					locations[LOC_STERNUM].row += row;
					locations[LOC_STERNUM].col += col;
					partCounter[LOC_STERNUM]++;
					continue;
				}
			}
		}
	}
	else {
		for (int i = 0; i < LOC_NUMBER; i++) {
			locations[i] = locs[i];
			partCounter[i] = 1;
		}
	}

	BodyPartLocations bpLocs;

	for (int i = 0; i < LOC_NUMBER; i++) {

		int min_row, max_row, min_col, max_col;
		int origin_row = 0;
		int origin_col = 0;
		int new_origin_row = (int) ((float) locations[i].row / (float) partCounter[i]);
		int new_origin_col = (int) ((float) locations[i].col / (float) partCounter[i]);

		while (new_origin_row != origin_row || new_origin_col != origin_col) {
			origin_row = new_origin_row;
			origin_col = new_origin_col;

			min_row = max(0, origin_row - 30);
			max_row = min(MAX_ROW - 1, origin_row + 30);
			min_col = max(0, origin_col - 30);
			max_col = min(MAX_COL - 1, origin_col + 30);

			locations[i].row = 0;
			locations[i].col = 0;
			partCounter[i] = 0;

			for (int row = min_row; row <= max_row; row++) {
				for (int col = min_col; col <= max_col; col++) {
					if (masks[i].at<uchar>(row, col) > 0) {
						locations[i].row += row;
						locations[i].col += col;
						partCounter[i]++;
					}
				}
			}

			new_origin_row = (int) ((float) locations[i].row / (float) partCounter[i]);
			new_origin_col = (int) ((float) locations[i].col / (float) partCounter[i]);
		}
		
		locations[i].row = origin_row;
		locations[i].col = origin_col;
		locations[i].accuracy = 1;
	}

	bool isInaccurate = false;
	float accuaracy_sum = 0;

	for (int i = 0; i < LOC_NUMBER; i++) {
		if (locations[i].row <= 0 || locations[i].col <= 0 || locations[i].row >= MAX_ROW - 1 || locations[i].col >= MAX_COL - 1) {
			locations[i].accuracy = 0;
		}
		else if(status == TRACKED || status == INACCURATE) {
			int dist_r = abs(locs[i].row - locations[i].row);
			int dist_c = abs(locs[i].col - locations[i].col);

			if (dist_r > 10) {
				//locations[i].row = locs[i].row;
				locations[i].accuracy = (float) max(locations[i].accuracy - 0.1, (double) 0);
				isInaccurate = true;
			}

			if (dist_c > 10) {
				//locations[i].col = locs[i].col;
				locations[i].accuracy = (float) max(locations[i].accuracy - 0.1, (double) 0);
				isInaccurate = true;
			}
		}

		//check why sometimes row and col are -MAX Integer
		locations[i].row = max(min(MAX_ROW - 1, locations[i].row), 0);
		locations[i].col = max(min(MAX_COL - 1, locations[i].col), 0);
		//printf("<%d,%d>\r", locations[i].row, locations[i].col);
		//if (locations[i].row != locs[i].row || locations[i].col != locs[i].col) {
		//	printf("\nValues changed!\n");
		//}
		locations[i].depth = subject.at<ushort>(locations[i].row, locations[i].col);
		locs[i] = locations[i];
		bpLocs.locs[i] = locations[i];
		accuaracy_sum += locs[i].accuracy;
	}

	if (accuaracy_sum == 0) {
		status = UNTRACKED;
	}
	else if (isInaccurate) {
		status = INACCURATE;
	}
	else {
		status = TRACKED;
	}

	return bpLocs;
}

TrackingStatus tree::BodyPartDetector::state()
{
	return status;
}
