#include "Features.hpp"
#include <iostream>
#include <cmath>

using namespace cv;
using std::string;

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

void getHorizontalIntegral(cv::Mat& image, cv::Mat& horizIntegral)
{ 
	if (image.type() != CV_16UC1 || horizIntegral.type() != CV_16UC1)
		throw std::exception("getHorizontalIntegral() one or both input images are no depth image!");
	if (horizIntegral.size() != image.size())
		throw std::exception("getHorizontalIntegral() inputs do not have same size!");

	int counter;
	for (int row = 0; row < image.rows; row++) {
		counter = 0;

		for (int col = 0; col < image.cols; col++) {
			if (image.at<ushort>(row, col) != 0)
				counter++;
			horizIntegral.at<ushort>(row, col) = counter;
		}
	}
}

void getVerticalIntegral(cv::Mat & image, cv::Mat & vertIntegral)
{
	if (image.type() != CV_16UC1 || vertIntegral.type() != CV_16UC1)
		throw std::exception("getHorizontalIntegral() one or both input images are no depth image!");
	if (vertIntegral.size() != image.size())
		throw std::exception("getHorizontalIntegral() inputs do not have same size!");

	int counter;

	for (int col = 0; col < image.cols; col++) {
		counter = 0;

		for (int row = 0; row < image.rows; row++) {
			if (image.at<ushort>(row, col) != 0)
				counter++;
			vertIntegral.at<ushort>(row, col) = counter;
		}
	}
}

void getIntegral(cv::Mat & image, cv::Mat & integral)
{
	if (image.type() != CV_16UC1 || integral.type() != CV_16UC1)
		throw std::exception("getHorizontalIntegral() one or both input images are no depth image!");
	if (integral.size() != image.size())
		throw std::exception("getHorizontalIntegral() inputs do not have same size!");

	//threshold(image, integral, 1, 1, THRESH_BINARY);
	integral = 0;
	integral.setTo(1, image > 0);

	for (int row = 1; row < image.rows; row++) {
		integral.at<ushort>(row, 0) = integral.at<ushort>(row, 0) + integral.at<ushort>(row - 1, 0);
	}

	for (int col = 1; col < image.cols; col++) {
		integral.at<ushort>(0, col) += integral.at<ushort>(0, col-1);
	}

	for (int row = 1; row < image.rows; row++) {
		for (int col = 1; col < image.cols; col++) {
			int A = integral.at<ushort>(row - 1, col - 1);
			int B = integral.at<ushort>(row - 1, col);
			int C = integral.at<ushort>(row, col - 1);
			int D = integral.at<ushort>(row, col);
			short value = min((int) MAX_VAL, B + C + D - A);
			integral.at<ushort>(row, col) = value;
		}
	}
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


void feature1(Mat& subject, Mat& feature1, Mat& depthImg, int offset) {
	if (subject.type() != 2) {
		std::cerr << "feature1() only supports CV_16U Mat types!" << std::endl;
		return;
	}

	feature1 = MAX_VAL;

	ushort pixel, luEdge, ldEdge, ruEdge, rdEdge;
	int refPixRow, refPixCol;
	offset = offset / 2;
	int tmpOffset;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = subject.at<ushort>(row, col);
			if (pixel == 0) continue;

			tmpOffset = (int) round((double) offset / ((double)depthImg.at<ushort>(row, col) / (double) ONE_METER));

			refPixRow = row - tmpOffset;
			refPixCol = col - tmpOffset;
			if (refPixRow < 0 || refPixCol < 0)
				luEdge = 0;
			else
				luEdge = subject.at<ushort>(refPixRow, refPixCol);

			refPixRow = row + tmpOffset;
			refPixCol = col - tmpOffset;
			if (refPixRow >= MAX_ROW || refPixCol < 0)
				ldEdge = 0;
			else
				ldEdge = subject.at<ushort>(refPixRow, refPixCol);

			refPixRow = row - tmpOffset;
			refPixCol = col + tmpOffset;
			if (refPixRow < 0 || refPixCol >= MAX_COL)
				ruEdge = 0;
			else
				ruEdge = subject.at<ushort>(refPixRow, refPixCol);

			refPixRow = row + tmpOffset;
			refPixCol = col + tmpOffset;
			if (refPixRow >= MAX_ROW || refPixCol >= MAX_COL)
				rdEdge = 0;
			else
				rdEdge = subject.at<ushort>(refPixRow, refPixCol);
			
			if (luEdge == 0 || ldEdge == 0 || ruEdge == 0 || rdEdge == 0)
				feature1.at<ushort>(row, col) = NORM_FACTOR;
			else {
				short val = abs(luEdge + rdEdge - ldEdge - ruEdge) / 2; // division by 2 because max value can be 2 * NORM_FACTOR but max is supposed to be NORM_FACTOR
				feature1.at<ushort>(row, col) = val;
			}
		}
	}
}

void feature2(Mat & subject, Mat & feature2, Mat& depthImg, int offset)
{
	if (subject.type() != 2) {
		std::cerr << "feature2() only supports CV_16U Mat types!" << std::endl;
		return;
	}

	feature2 = 0;
	int tmpOffset;

	ushort pixel, ref;
	int refPix;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = subject.at<ushort>(row, col);
			if (pixel == 0) continue;

			tmpOffset = (int) round((double)offset / ((double)depthImg.at<ushort>(row, col) / (double)ONE_METER));

			refPix = row - tmpOffset;
			if (refPix < 0)
				ref = 0;
			else
				ref = subject.at<ushort>(refPix, col);

			feature2.at<ushort>(row, col) = max(abs(pixel - ref), 1);
		}
	}

	double dMaxVal;
	minMaxIdx(feature2, NULL, &dMaxVal);
	ushort maxVal = (ushort)dMaxVal;

	Mat maxValMat = Mat(feature2.rows, feature2.cols, feature2.type());
	maxValMat = Scalar(maxVal);

	divide(maxValMat, feature2, feature2);
	feature2.setTo(Scalar(MAX_VAL), subject == 0);
}

void feature3(Mat& subject, Mat& feature3, Mat& depthImg, int offset) {
	if (subject.type() != 2) {
		std::cerr << "feature3() only supports CV_16U Mat types!" << std::endl;
		return;
	}

	feature3 = 0;
	int tmpOffset;

	ushort pixel, ref;
	int refPix;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = subject.at<ushort>(row, col);
			if (pixel == 0) continue;

			tmpOffset = (int) round((double)offset / ((double)depthImg.at<ushort>(row, col) / (double)ONE_METER));

			refPix = col - tmpOffset;
			if (refPix < 0)
				ref = 0;
			else
				ref = subject.at<ushort>(row, refPix);

			feature3.at<ushort>(row, col) = max(abs(pixel - ref), 1);
		}
	}

	double dMaxVal;
	minMaxIdx(feature3, NULL, &dMaxVal);
	ushort maxVal = (ushort)dMaxVal;

	Mat maxValMat = Mat(feature3.rows, feature3.cols, feature3.type());
	maxValMat = Scalar(maxVal);

	divide(maxValMat, feature3, feature3);
	feature3.setTo(Scalar(MAX_VAL), subject == 0);
}

void feature4(Mat& subject, Mat& feature4, Mat& depthImg, int offset) 
{
	if (subject.type() != 2) {
		std::cerr << "feature4() only supports CV_16U Mat types!" << std::endl;
		return;
	}

	feature4 = 0;
	int tmpOffset;

	ushort pixel, ref;
	int refPix;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = subject.at<ushort>(row, col);
			if (pixel == 0) continue;

			tmpOffset = (int) round((double)offset / ((double)depthImg.at<ushort>(row, col) / (double)ONE_METER));

			refPix = col + tmpOffset;
			if (refPix >= MAX_COL)
				ref = 0;
			else
				ref = subject.at<ushort>(row, refPix);

			feature4.at<ushort>(row, col) = max(abs(pixel - ref), 1);
		}
	}

	double dMaxVal;
	minMaxIdx(feature4, NULL, &dMaxVal);
	ushort maxVal = (ushort)dMaxVal;

	Mat maxValMat = Mat(feature4.rows, feature4.cols, feature4.type());
	maxValMat = Scalar(maxVal);

	divide(maxValMat, feature4, feature4);
	feature4.setTo(Scalar(MAX_VAL), subject == 0);
}

void feature5(Mat& subject, Mat& feature5, Mat& depthImg, int offset) {
	if (subject.type() != 2) {
		std::cerr << "feature5() only supports CV_16U Mat types!" << std::endl;
		return;
	}

	feature5 = 0;
	int tmpOffset;

	ushort pixel, ref1, ref2;
	int refPix;
	offset = offset / 2;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = subject.at<ushort>(row, col);
			if (pixel == 0) continue;

			tmpOffset = (int) round((double)offset / ((double)depthImg.at<ushort>(row, col) / (double)ONE_METER));

			refPix = col - tmpOffset;
			if (refPix < 0)
				ref1 = 0;
			else
				ref1 = subject.at<ushort>(row, refPix);

			refPix = col + tmpOffset;
			if (refPix >= MAX_COL)
				ref2 = 0;
			else
				ref2 = subject.at<ushort>(row, refPix);

			feature5.at<ushort>(row, col) = max(abs(ref2 - ref1), 1);
		}
	}

	feature5.setTo(Scalar(MAX_VAL), subject == 0);
}

void feature6(Mat& subject, Mat& feature6, Mat& depthImg, int offset) {
	if (subject.type() != 2) {
		std::cerr << "feature6() only supports CV_16U Mat types!" << std::endl;
		return;
	}

	feature6 = 0;
	int tmpOffset;

	ushort pixel, ref1, ref2;
	int refPix;
	offset = offset / 2;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = subject.at<ushort>(row, col);
			if (pixel == 0) continue;

			tmpOffset = (int) round((double)offset / ((double)depthImg.at<ushort>(row, col) / (double)ONE_METER));

			refPix = row - tmpOffset;
			if (refPix < 0)
				ref1 = 0;
			else
				ref1 = subject.at<ushort>(refPix, col);

			refPix = row + tmpOffset;
			if (refPix >= MAX_ROW)
				ref2 = 0;
			else
				ref2 = subject.at<ushort>(refPix, col);

			feature6.at<ushort>(row, col) = abs(ref2 - ref1);
		}
	}

	feature6.setTo(Scalar(MAX_VAL), subject == 0);
}

void feature7(Mat& subject, Mat& feature7, Mat& depthImg, int offset) {
	if (subject.type() != 2) {
		std::cerr << "feature7() only supports CV_16U Mat types!" << std::endl;
		return;
	}

	feature7 = 0;
	int tmpOffset;

	ushort pixel, ref1, ref2;
	int refPix;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = subject.at<ushort>(row, col);
			if (pixel == 0) continue;

			tmpOffset = (int) round((double)offset / ((double)depthImg.at<ushort>(row, col) / (double)ONE_METER));

			refPix = row - tmpOffset;
			if (refPix < 0)
				ref1 = 0;
			else
				ref1 = subject.at<ushort>(refPix, col);

			refPix = col - tmpOffset;
			if (refPix < 0)
				ref2 = 0;
			else
				ref2 = subject.at<ushort>(row, refPix);

			feature7.at<ushort>(row, col) = abs(ref2 - ref1);
		}
	}

	feature7.setTo(Scalar(MAX_VAL), subject == 0);
}

void feature8(Mat& subject, Mat& feature8, Mat& depthImg, int offset) {
	if (subject.type() != 2) {
		std::cerr << "feature8() only supports CV_16U Mat types!" << std::endl;
		return;
	}

	feature8 = 0;
	int tmpOffset;

	ushort pixel, ref1, ref2;
	int refPix;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = subject.at<ushort>(row, col);
			if (pixel == 0) continue;

			tmpOffset = (int) round((double)offset / ((double)depthImg.at<ushort>(row, col) / (double)ONE_METER));

			refPix = row - tmpOffset;
			if (refPix < 0)
				ref1 = 0;
			else
				ref1 = subject.at<ushort>(refPix, col);

			refPix = col + tmpOffset;
			if (refPix >= MAX_COL)
				ref2 = 0;
			else
				ref2 = subject.at<ushort>(row, refPix);

			feature8.at<ushort>(row, col) = abs(ref2 - ref1);
		}
	}

	feature8.setTo(Scalar(MAX_VAL), subject == 0);
}

void feature9(Mat& subject, Mat& feature9, Mat& depthImg, int offset) {
	if (subject.type() != 2) {
		std::cerr << "feature9() only supports CV_16U Mat types!" << std::endl;
		return;
	}

	feature9 = 0;
	int tmpOffset;

	ushort pixel, ref1, ref2;
	int refPixRow, refPixCol;
	offset = (int) sqrt(offset * offset / 2) / 2;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = subject.at<ushort>(row, col);
			if (pixel == 0) continue;

			tmpOffset = (int) round((double)offset / ((double)depthImg.at<ushort>(row, col) / (double)ONE_METER));

			refPixRow = row - tmpOffset;
			refPixCol = col - tmpOffset;
			if (refPixRow < 0 || refPixCol < 0)
				ref1 = 0;
			else
				ref1 = subject.at<ushort>(refPixRow, refPixCol);

			refPixRow = row + tmpOffset;
			refPixCol = col + tmpOffset;
			if (refPixRow >= MAX_ROW || refPixCol >= MAX_COL)
				ref2 = 0;
			else
				ref2 = subject.at<ushort>(refPixRow, refPixCol);

			feature9.at<ushort>(row, col) = abs(ref2 - ref1);
		}
	}

	feature9.setTo(Scalar(MAX_VAL), subject == 0);
}

void feature10(Mat& subject, Mat& feature10, Mat& depthImg, int offset) {
	if (subject.type() != 2) {
		std::cerr << "feature10() only supports CV_16U Mat types!" << std::endl;
		return;
	}

	feature10 = 0;
	int tmpOffset;

	ushort pixel, ref1, ref2;
	int refPixRow, refPixCol;
	offset = (int) sqrt(offset * offset / 2) / 2;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = subject.at<ushort>(row, col);
			if (pixel == 0) continue;

			tmpOffset = (int) round((double)offset / ((double)depthImg.at<ushort>(row, col) / (double)ONE_METER));

			refPixRow = row + tmpOffset;
			refPixCol = col - tmpOffset;
			if (refPixRow >= MAX_ROW || refPixCol < 0)
				ref1 = 0;
			else
				ref1 = subject.at<ushort>(refPixRow, refPixCol);

			refPixRow = row - tmpOffset;
			refPixCol = col + tmpOffset;
			if (refPixRow < 0 || refPixCol >= MAX_COL)
				ref2 = 0;
			else
				ref2 = subject.at<ushort>(refPixRow, refPixCol);

			feature10.at<ushort>(row, col) = abs(ref2 - ref1);
		}
	}

	feature10.setTo(Scalar(MAX_VAL), subject == 0);
}

void feature11(cv::Mat & subject, cv::Mat & feature11, int offset, cv::Mat& horizIntegral)
{
	if (subject.type() != 2) {
		std::cerr << "feature11() only supports CV_16U Mat types!" << std::endl;
		return;
	}

	if (horizIntegral.empty()) {
		horizIntegral.create(subject.rows, subject.size, CV_16UC1);
		getHorizontalIntegral(subject, horizIntegral);
	}

	feature11 = 0;
	ushort pixel, ref1, ref2, ref3;
	int refPix1, refPix3;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = subject.at<ushort>(row, col);
			if (pixel == 0) continue;

			refPix1 = min(col + offset, MAX_COL-1);
			ref1 = horizIntegral.at<ushort>(row, refPix1);

			refPix3 = max(col - offset - 1, 0);
			ref3 = horizIntegral.at<ushort>(row, refPix3);

			ref2 = horizIntegral.at<ushort>(row, col);

			int numPixelsToRight = ref1 - ref2;
			int numPixelsToLeft = ref2 - ref3;

			feature11.at<ushort>(row, col) = MAX_COL + numPixelsToRight - numPixelsToLeft;
		}
	}

	feature11.setTo(Scalar(MAX_VAL), subject == 0);
}

void feature12(cv::Mat & subject, cv::Mat & feature12, int offset, cv::Mat& vertIntegral)
{
	if (subject.type() != 2) {
		std::cerr << "feature12() only supports CV_16U Mat types!" << std::endl;
		return;
	}

	if (vertIntegral.empty()) {
		vertIntegral.create(subject.rows, subject.size, CV_16UC1);
		getVerticalIntegral(subject, vertIntegral);
	}

	feature12 = 0;
	ushort pixel, ref1, ref2, ref3;
	int refPix1, refPix3;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = subject.at<ushort>(row, col);
			if (pixel == 0) continue;

			refPix1 = min(row + offset, MAX_ROW - 1);
			ref1 = vertIntegral.at<ushort>(refPix1, col);

			refPix3 = max(row - offset - 1, 0);
			ref3 = vertIntegral.at<ushort>(refPix3, col);

			ref2 = vertIntegral.at<ushort>(row, col);

			int numPixelsDownwards = ref1 - ref2;
			int numPixelsUpwards = ref2 - ref3;

			feature12.at<ushort>(row, col) = MAX_ROW + numPixelsDownwards - numPixelsUpwards;
		}
	}

	feature12.setTo(Scalar(MAX_VAL), subject == 0);
}

void feature13(cv::Mat & subject, cv::Mat & feature13, int offset, cv::Mat & integral)
{
	if (subject.type() != 2) {
		std::cerr << "feature13() only supports CV_16U Mat types!" << std::endl;
		return;
	}

	if (integral.empty()) {
		integral.create(subject.rows, subject.size, CV_16UC1);
		getVerticalIntegral(subject, integral);
		getHorizontalIntegral(subject, integral);
	}

	offset /= 2;

	feature13 = 0;
	ushort pixel, A, B, C, D;
	int refPix1, refPix2;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			pixel = subject.at<ushort>(row, col);
			if (pixel == 0) continue;

			refPix1 = max(row - offset - 1, 0);
			refPix2 = max(col - offset - 1, 0);
			if (row - offset - 1 < 0 || col - offset - 1 < 0) {
				A = 0;
			}
			else {
				A = integral.at<ushort>(refPix1, refPix2);
			}

			refPix1 = max(row - offset - 1, 0);
			refPix2 = min(col + offset, MAX_COL - 1);
			if (row - offset - 1 < 0) {
				B = 0;
			}
			else {
				B = integral.at<ushort>(refPix1, refPix2);
			}

			refPix1 = min(row + offset, MAX_ROW - 1);
			refPix2 = max(col - offset - 1, 0);
			if (col - offset - 1 < 0) {
				C = 0;
			}
			else {
				C = integral.at<ushort>(refPix1, refPix2);
			}

			refPix1 = min(row + offset, MAX_ROW - 1);
			refPix2 = min(col + offset, MAX_COL - 1);
			D = integral.at<ushort>(refPix1, refPix2);

			feature13.at<ushort>(row, col) = A + D - B - C;
		}
	}

	feature13.setTo(Scalar(MAX_VAL), subject == 0);
}

void featurizeImage(Mat& depImg, tree::Dataset**& featureMatrix) {

	Mat feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10;
	Mat feat11, feat12, feat13, feat14, feat15, feat16, feat17, feat18, feat19, feat20;
	Mat feat21, feat22, feat23, feat24, feat25, feat26;
	Mat horizIntegral, vertIntegral, integral;
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
	feat11.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat12.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat13.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat14.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat15.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat16.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat17.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat18.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat19.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat20.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat21.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat22.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat23.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat24.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat25.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	feat26.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	horizIntegral.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	vertIntegral.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	integral.create(MAX_ROW, MAX_COL, DEPTH_IMAGE);

	Mat subject = Mat(MAX_ROW, MAX_COL, DEPTH_IMAGE);
	getSubject(depImg, subject);

	imshow("Subject", subject);

	getVerticalIntegral(subject, vertIntegral);
	getHorizontalIntegral(subject, horizIntegral);
	getIntegral(subject, integral);

	feature1(subject, feat1, depImg, 30);
	feature2(subject, feat2, depImg, 20);
	feature3(subject, feat3, depImg, 20);
	feature4(subject, feat4, depImg, 20);
	feature5(subject, feat5, depImg, 40);
	feature6(subject, feat6, depImg, 60);
	feature7(subject, feat7, depImg, 30);
	feature8(subject, feat8, depImg, 30);
	feature9(subject, feat9, depImg, 50);
	feature10(subject, feat10, depImg, 50);
	feature1(subject, feat11, depImg, 15);
	feature2(subject, feat12, depImg, 10);
	feature3(subject, feat13, depImg, 10);
	feature4(subject, feat14, depImg, 10);
	feature5(subject, feat15, depImg, 20);
	feature6(subject, feat16, depImg, 30);
	feature7(subject, feat17, depImg, 15);
	feature8(subject, feat18, depImg, 15);
	feature9(subject, feat19, depImg, 25);
	feature10(subject, feat20, depImg, 25);
	feature11(subject, feat21, MAX_COL, horizIntegral);
	feature11(subject, feat22, 20, horizIntegral);
	feature12(subject, feat23, MAX_ROW, vertIntegral);
	feature12(subject, feat24, 20, vertIntegral);
	feature13(subject, feat25, 20, integral);
	feature13(subject, feat26, 50, integral);

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
			set.feature[10] = feat11.at<ushort>(row, col);
			set.feature[11] = feat12.at<ushort>(row, col);
			set.feature[12] = feat13.at<ushort>(row, col);
			set.feature[13] = feat14.at<ushort>(row, col);
			set.feature[14] = feat15.at<ushort>(row, col);
			set.feature[15] = feat16.at<ushort>(row, col);
			set.feature[16] = feat17.at<ushort>(row, col);
			set.feature[17] = feat18.at<ushort>(row, col);
			set.feature[18] = feat19.at<ushort>(row, col);
			set.feature[19] = feat20.at<ushort>(row, col);
			set.feature[20] = feat21.at<ushort>(row, col);
			set.feature[21] = feat22.at<ushort>(row, col);
			set.feature[22] = feat23.at<ushort>(row, col);
			set.feature[23] = feat24.at<ushort>(row, col);
			set.feature[24] = feat25.at<ushort>(row, col);
			set.feature[25] = feat26.at<ushort>(row, col);
			set.outcome = OTHER;

			featureMatrix[row][col] = set;
		}
}
