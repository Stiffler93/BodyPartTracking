#include "Features.hpp"
#include <iostream>
#include <cmath>

void getSubject(Mat& depImg, Mat& depImgSubject) {
	depImgSubject = depImg.clone();
	Scalar meanDepth = mean(depImgSubject);
	depImgSubject.setTo(0, depImg < MIN_DIST | depImg > meanDepth[0]);
	Mat tmp;
	depImgSubject.convertTo(tmp, CV_64FC1);

	Mat mask;

	threshold(tmp, mask, 0, 255, THRESH_BINARY);
	mask.convertTo(mask, CV_8U);

	Mat labels, stats, centroids, temp;
	int nLabels = connectedComponentsWithStats(mask, labels, stats, centroids);
	if (nLabels > 1) {
		int biggestLabel = 0, biggestLabelSize = 0;
		for (int i = 1; i <= nLabels; i++) {
			int count = countNonZero(labels == i);
			if (count > biggestLabelSize) {
				biggestLabelSize = count;
				biggestLabel = i;
			}
		}

		mask.setTo(0, labels != biggestLabel);
	}

	medianBlur(mask, mask, 3);

	double minVal, maxVal;
	minMaxIdx(depImgSubject, &minVal, &maxVal, NULL, NULL, mask != 0);

	// normalize depth values

	short min = (short)minVal;
	short max = (short)maxVal;

	depImgSubject -= min;
	double factor = (1.0 / (double) (max - min) *  (double) NORM_FACTOR);
	depImgSubject *= factor;

	depImgSubject.setTo(0, mask == 0);
}

string getCategory(int red, int green, int blue) {
	if (red >= 250 && green <= 5 && blue <= 5) {
		return LEFT_SHOULDER;
	}

	if (red <= 5 && green >= 250 && blue <= 5) {
		return RIGHT_SHOULDER;
	}

	if (red <= 5 && green <= 5 && blue >= 250) {
		return HEAD;
	}

	if (red <= 5 && green <= 5 && blue <= 5) {
		return STERNUM;
	}

	if (red <= 5 && green >= 250 && blue >= 250) {
		return NECK;
	}

	if (red >= 250 && green >= 250 && blue >= 250) {
		return OTHER;
	}

	return NONE;
}

Vec3b getBGR(string category)
{
	Vec3b color;
	if (category == LEFT_SHOULDER) {
		color[2] = 255;
		color[1] = 0;
		color[0] = 0;
	}
	else if (category == RIGHT_SHOULDER) {
		color[2] = 0;
		color[1] = 255;
		color[0] = 0;
	}
	else if (category == STERNUM) {
		color[2] = 0;
		color[1] = 0;
		color[0] = 0;
	}
	else if (category == HEAD) {
		color[2] = 0;
		color[1] = 0;
		color[0] = 255;
	}
	else if (category == NECK) {
		color[2] = 0;
		color[1] = 255;
		color[0] = 255;
	}
	else if (category == OTHER) {
		color[2] = 255;
		color[1] = 255;
		color[0] = 255;
	}
	else {
		color[2] = 127;
		color[1] = 127;
		color[0] = 127;
	}
	return color;
}


void feature1(Mat& depImg, Mat& feature1, int offset) {
	if (depImg.type() != 2) {
		cerr << "feature1() only supports CV_16U Mat types!" << endl;
		return;
	}

	feature1 = MAX_VAL;

	ushort pixel, luEdge, ldEdge, ruEdge, rdEdge;
	int refPixRow, refPixCol;
	offset = offset / 2;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = depImg.at<ushort>(row, col);
			if (pixel == 0) continue;

			refPixRow = row - offset;
			refPixCol = col - offset;
			if (refPixRow < 0 || refPixCol < 0)
				luEdge = 0;
			else
				luEdge = depImg.at<ushort>(refPixRow, refPixCol);

			refPixRow = row + offset;
			refPixCol = col - offset;
			if (refPixRow >= MAX_ROW || refPixCol < 0)
				ldEdge = 0;
			else
				ldEdge = depImg.at<ushort>(refPixRow, refPixCol);

			refPixRow = row - offset;
			refPixCol = col + offset;
			if (refPixRow < 0 || refPixCol >= MAX_COL)
				ruEdge = 0;
			else
				ruEdge = depImg.at<ushort>(refPixRow, refPixCol);

			refPixRow = row + offset;
			refPixCol = col + offset;
			if (refPixRow >= MAX_ROW || refPixCol >= MAX_COL)
				rdEdge = 0;
			else
				rdEdge = depImg.at<ushort>(refPixRow, refPixCol);
			
			if (luEdge == 0 || ldEdge == 0 || ruEdge == 0 || rdEdge == 0)
				feature1.at<ushort>(row, col) = NORM_FACTOR;
			else {
				short val = abs(luEdge + rdEdge - ldEdge - ruEdge) / 2; // division by 2 because max value can be 2 * NORM_FACTOR but max is supposed to be NORM_FACTOR
				feature1.at<ushort>(row, col) = val;
			}
		}
	}
}

void feature2(Mat & depImg, Mat & feature2, int offset)
{
	if (depImg.type() != 2) {
		cerr << "feature2() only supports CV_16U Mat types!" << endl;
		return;
	}

	feature2 = 0;

	ushort pixel, ref;
	int refPix;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = depImg.at<ushort>(row, col);
			if (pixel == 0) continue;

			refPix = row - offset;
			if (refPix < 0)
				ref = 0;
			else
				ref = depImg.at<ushort>(refPix, col);

			feature2.at<ushort>(row, col) = max(abs(pixel - ref), 1);
		}
	}

	double dMaxVal;
	minMaxIdx(feature2, NULL, &dMaxVal);
	ushort maxVal = (ushort)dMaxVal;

	Mat maxValMat = Mat(feature2.rows, feature2.cols, feature2.type());
	maxValMat = Scalar(maxVal);

	divide(maxValMat, feature2, feature2);
	feature2.setTo(Scalar(MAX_VAL), depImg == 0);
}

void feature3(Mat& depImg, Mat& feature3, int offset) {
	if (depImg.type() != 2) {
		cerr << "feature3() only supports CV_16U Mat types!" << endl;
		return;
	}

	feature3 = 0;

	ushort pixel, ref;
	int refPix;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = depImg.at<ushort>(row, col);
			if (pixel == 0) continue;

			refPix = col - offset;
			if (refPix < 0)
				ref = 0;
			else
				ref = depImg.at<ushort>(row, refPix);

			feature3.at<ushort>(row, col) = max(abs(pixel - ref), 1);
		}
	}

	double dMaxVal;
	minMaxIdx(feature3, NULL, &dMaxVal);
	ushort maxVal = (ushort)dMaxVal;

	Mat maxValMat = Mat(feature3.rows, feature3.cols, feature3.type());
	maxValMat = Scalar(maxVal);

	divide(maxValMat, feature3, feature3);
	feature3.setTo(Scalar(MAX_VAL), depImg == 0);
}

void feature4(Mat& depImg, Mat& feature4, int offset) 
{
	if (depImg.type() != 2) {
		cerr << "feature4() only supports CV_16U Mat types!" << endl;
		return;
	}

	feature4 = 0;

	ushort pixel, ref;
	int refPix;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = depImg.at<ushort>(row, col);
			if (pixel == 0) continue;

			refPix = col + offset;
			if (refPix >= MAX_COL)
				ref = 0;
			else
				ref = depImg.at<ushort>(row, refPix);

			feature4.at<ushort>(row, col) = max(abs(pixel - ref), 1);
		}
	}

	double dMaxVal;
	minMaxIdx(feature4, NULL, &dMaxVal);
	ushort maxVal = (ushort)dMaxVal;

	Mat maxValMat = Mat(feature4.rows, feature4.cols, feature4.type());
	maxValMat = Scalar(maxVal);

	divide(maxValMat, feature4, feature4);
	feature4.setTo(Scalar(MAX_VAL), depImg == 0);
}

void feature5(Mat& depImg, Mat& feature5, int offset) {
	if (depImg.type() != 2) {
		cerr << "feature5() only supports CV_16U Mat types!" << endl;
		return;
	}

	feature5 = 0;

	ushort pixel, ref1, ref2;
	int refPix;
	offset = offset / 2;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = depImg.at<ushort>(row, col);
			if (pixel == 0) continue;

			refPix = col - offset;
			if (refPix < 0)
				ref1 = 0;
			else
				ref1 = depImg.at<ushort>(row, refPix);

			refPix = col + offset;
			if (refPix >= MAX_COL)
				ref2 = 0;
			else
				ref2 = depImg.at<ushort>(row, refPix);

			feature5.at<ushort>(row, col) = max(abs(ref2 - ref1), 1);
		}
	}

	feature5.setTo(Scalar(MAX_VAL), depImg == 0);
}

void feature6(Mat& depImg, Mat& feature6, int offset) {
	if (depImg.type() != 2) {
		cerr << "feature6() only supports CV_16U Mat types!" << endl;
		return;
	}

	feature6 = 0;

	ushort pixel, ref1, ref2;
	int refPix;
	offset = offset / 2;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = depImg.at<ushort>(row, col);
			if (pixel == 0) continue;

			refPix = row - offset;
			if (refPix < 0)
				ref1 = 0;
			else
				ref1 = depImg.at<ushort>(refPix, col);

			refPix = row + offset;
			if (refPix >= MAX_ROW)
				ref2 = 0;
			else
				ref2 = depImg.at<ushort>(refPix, col);

			feature6.at<ushort>(row, col) = abs(ref2 - ref1);
		}
	}

	feature6.setTo(Scalar(MAX_VAL), depImg == 0);
}

void feature7(Mat& depImg, Mat& feature7, int offset) {
	if (depImg.type() != 2) {
		cerr << "feature7() only supports CV_16U Mat types!" << endl;
		return;
	}

	feature7 = 0;

	ushort pixel, ref1, ref2;
	int refPix;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = depImg.at<ushort>(row, col);
			if (pixel == 0) continue;

			refPix = row - offset;
			if (refPix < 0)
				ref1 = 0;
			else
				ref1 = depImg.at<ushort>(refPix, col);

			refPix = col - offset;
			if (refPix < 0)
				ref2 = 0;
			else
				ref2 = depImg.at<ushort>(row, refPix);

			feature7.at<ushort>(row, col) = abs(ref2 - ref1);
		}
	}

	feature7.setTo(Scalar(MAX_VAL), depImg == 0);
}

void feature8(Mat& depImg, Mat& feature8, int offset) {
	if (depImg.type() != 2) {
		cerr << "feature8() only supports CV_16U Mat types!" << endl;
		return;
	}

	feature8 = 0;

	ushort pixel, ref1, ref2;
	int refPix;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = depImg.at<ushort>(row, col);
			if (pixel == 0) continue;

			refPix = row - offset;
			if (refPix < 0)
				ref1 = 0;
			else
				ref1 = depImg.at<ushort>(refPix, col);

			refPix = col + offset;
			if (refPix >= MAX_COL)
				ref2 = 0;
			else
				ref2 = depImg.at<ushort>(row, refPix);

			feature8.at<ushort>(row, col) = abs(ref2 - ref1);
		}
	}

	feature8.setTo(Scalar(MAX_VAL), depImg == 0);
}

void feature9(Mat& depImg, Mat& feature9, int offset) {
	if (depImg.type() != 2) {
		cerr << "feature9() only supports CV_16U Mat types!" << endl;
		return;
	}

	feature9 = 0;

	ushort pixel, ref1, ref2;
	int refPixRow, refPixCol;
	offset = (int) sqrt(offset * offset / 2) / 2;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = depImg.at<ushort>(row, col);
			if (pixel == 0) continue;

			refPixRow = row - offset;
			refPixCol = col - offset;
			if (refPixRow < 0 || refPixCol < 0)
				ref1 = 0;
			else
				ref1 = depImg.at<ushort>(refPixRow, refPixCol);

			refPixRow = row + offset;
			refPixCol = col + offset;
			if (refPixRow >= MAX_ROW || refPixCol >= MAX_COL)
				ref2 = 0;
			else
				ref2 = depImg.at<ushort>(refPixRow, refPixCol);

			feature9.at<ushort>(row, col) = abs(ref2 - ref1);
		}
	}

	feature9.setTo(Scalar(MAX_VAL), depImg == 0);
}

void feature10(Mat& depImg, Mat& feature10, int offset) {
	if (depImg.type() != 2) {
		cerr << "feature10() only supports CV_16U Mat types!" << endl;
		return;
	}

	feature10 = 0;

	ushort pixel, ref1, ref2;
	int refPixRow, refPixCol;
	offset = (int) sqrt(offset * offset / 2) / 2;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = depImg.at<ushort>(row, col);
			if (pixel == 0) continue;

			refPixRow = row + offset;
			refPixCol = col - offset;
			if (refPixRow >= MAX_ROW || refPixCol < 0)
				ref1 = 0;
			else
				ref1 = depImg.at<ushort>(refPixRow, refPixCol);

			refPixRow = row - offset;
			refPixCol = col + offset;
			if (refPixRow < 0 || refPixCol >= MAX_COL)
				ref2 = 0;
			else
				ref2 = depImg.at<ushort>(refPixRow, refPixCol);

			feature10.at<ushort>(row, col) = abs(ref2 - ref1);
		}
	}

	feature10.setTo(Scalar(MAX_VAL), depImg == 0);
}

void featurizeImage(Mat& depImg, tree::Dataset**& featureMatrix) {

	Mat feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10;
	feat1.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat2.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat3.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat4.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat5.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat6.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat7.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat8.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat9.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat10.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);

	Mat subject = Mat(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	getSubject(depImg, subject);

	feature1(subject, feat1, 30);
	feature2(subject, feat2, 20);
	feature3(subject, feat3, 20);
	feature4(subject, feat4, 20);
	feature5(subject, feat5, 40);
	feature6(subject, feat6, 60);
	feature7(subject, feat7, 30);
	feature8(subject, feat8, 30);
	feature9(subject, feat9, 50);
	feature10(subject, feat10, 50);

	tree::Dataset set;
	for(int row = 0; row < MAX_ROW; row++) 
		for (int col = 0; col < MAX_COL; col++) {
			if (subject.at<ushort>(row, col) == 0) {
				set.outcome = NONE;
				featureMatrix[row][col] = set;
				continue;
			}
			set.feature[0] = feat1.at<ushort>(row, col);
			set.feature[1] = feat2.at<ushort>(row, col);
			set.feature[2] = feat3.at<ushort>(row, col);
			set.feature[3] = feat4.at<ushort>(row, col);
			set.feature[4] = feat5.at<ushort>(row, col);
			set.feature[5] = feat6.at<ushort>(row, col);
			set.feature[6] = feat7.at<ushort>(row, col);
			set.feature[7] = feat8.at<ushort>(row, col);
			set.feature[8] = feat9.at<ushort>(row, col);
			set.feature[9] = feat10.at<ushort>(row, col);
			set.outcome = OTHER;

			featureMatrix[row][col] = set;
		}
}
