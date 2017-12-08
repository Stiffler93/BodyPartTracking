#include "Features.hpp"
#include <iostream>

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
	//printf("nLabels: %d\n", nLabels);
	if (nLabels > 1) {
		int biggestLabel = 0, biggestLabelSize = 0;
		for (int i = 1; i <= nLabels; i++) {
			/*temp = 0;
			temp.setTo(1, labels == i);*/
			int count = countNonZero(labels == i);
			if (count > biggestLabelSize) {
				biggestLabelSize = count;
				biggestLabel = i;
			}
		}

		//printf("Biggest Component is %d\n", biggestLabel);
		mask.setTo(0, labels != biggestLabel);
	}

	medianBlur(mask, mask, 3);

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
				feature1.at<ushort>(row, col) = MAX_VAL;
			else
				feature1.at<ushort>(row, col) = abs(luEdge + rdEdge - ldEdge - ruEdge);
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