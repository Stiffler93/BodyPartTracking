#include "Ergonomics.h"
#include "ImageRecorder.h"

using namespace ergonomics;
using namespace cv;

#define WINDOW_MODEL "Background Model"
#define WINDOW_CHANGES "Changes between Model and Frame"
#define WINDOW_MOVEMENT "User Movement"

#define WINDOW_SKELETON "Body Skeleton"
#define WINDOW_BORDER "Body Border"
#define WINDOW_ANALYZED_BODY "Body Border and Skeleton"

TrackingDistance* TrackingDistance::_instance = 0;
BodyClassification* BodyClassification::_instance = 0;
//DrawSkeleton* DrawSkeleton::_instance = 0;

bool greaterZero(uchar* value);
bool greaterFifteen(uchar* value);

//static string s;

static void findFirst(Mat img, int * position, int * value, bool(*check)(uchar* value));
static void findLast(Mat img, int * position, int * value, bool(*check)(uchar* value));

TrackingDistance::TrackingDistance() 
{
	puts("Setup TrackingDistance");
	nthFrame = FPS / TRACK_FPS;
	numFrames = TRACK_FPS * TRACK_FPS;
	frameCounter = 0;

	lastFrames = new Mat[numFrames];
	lastMaxs = new int[DIST_INERTIA];

	for (int i = 0; i < numFrames; i++) {
		lastFrames[i] = Mat::zeros(ROWS, COLS, CV_64FC1);
	}

	for (int i = 0; i < DIST_INERTIA; i++) {
		lastMaxs[i] = MAX_DISTANCE;
	}

	weightedModel = new Mat(ROWS, COLS, CV_64FC1);
	currentFrame = new Mat(ROWS, COLS, CV_64FC1);

	movingMask = new Mat(ROWS, COLS, CV_8U);
	staticMask = new Mat(ROWS, COLS, CV_8U);
	viewMask = new Mat(ROWS, COLS, CV_8U);

	tmp1 = new Mat(ROWS, COLS, CV_8U);
	tmp2 = new Mat(ROWS, COLS, CV_8U);
}

TrackingDistance::~TrackingDistance()
{
	puts("Stop Tracking Distance");

	if (testModel != NULL) {
		destroyWindow(WINDOW_MODEL);
		delete testModel;
	}
	if (testMovement != NULL) {
		destroyWindow(WINDOW_CHANGES);
		delete testMovement;
	}

	if(showMovementFlag) destroyWindow(WINDOW_MOVEMENT);

	delete[] lastFrames;
	delete weightedModel, movingMask, staticMask;
	delete tmp1, tmp2, viewMask, lastMaxs;
}

void TrackingDistance::release()
{
	puts("Release TrackingDistance");
	delete _instance;
	_instance = NULL;
}

int TrackingDistance::getMinDist()
{
	return minDist;
}

int TrackingDistance::getMaxDist()
{
	return maxDist;
}

Mat ergonomics::TrackingDistance::getViewMask()
{
	return *viewMask;
}

bool TrackingDistance::postFrame(Mat& frame)
{
	if (++frameCounter != nthFrame) return false;

	frameCounter = 0;
	frame.convertTo(*currentFrame, CV_64FC1);
	calcModel();
	calcDifferences();
	calcNewDists();

	*tmp1 = Scalar(0);
	(*tmp1).convertTo(*tmp1, CV_64FC1);
	(*currentFrame).copyTo(*tmp1, *viewMask);
	(*tmp1).copyTo(lastFrames[lastIndex]);
	lastIndex = (lastIndex + 1) % numFrames;

	util::ImageSaver::save(*viewMask);

	return true;
}

Mat TrackingDistance::getModel()
{
	return *weightedModel;
}

void ergonomics::TrackingDistance::getProcessedFrame(Mat & img)
{
	lastFrames[lastIndex].copyTo(img, *viewMask);
}

void TrackingDistance::showModel(bool b)
{
	if (b == showModelFlag) return;

	if (b) {
		namedWindow(WINDOW_MODEL, CV_WINDOW_AUTOSIZE);
		testModel = new Mat(ROWS, COLS, CV_64FC1);
	}
	else {
		destroyWindow(WINDOW_MODEL);
		delete testModel;
		testModel = NULL;
	}

	showModelFlag = b;
}

void TrackingDistance::showChanges(bool b)
{
	if (showChangesFlag == b) return;

	if (b) {
		namedWindow(WINDOW_CHANGES, CV_WINDOW_AUTOSIZE);
		testMovement = new Mat(ROWS, COLS, CV_8U);
	}
	else {
		destroyWindow(WINDOW_CHANGES);
		delete testMovement;
		testMovement = NULL;
	}

	showChangesFlag = b;
}

void TrackingDistance::showMovement(bool b)
{
	if (showMovementFlag == b) return;

	if (b) namedWindow(WINDOW_MOVEMENT, CV_WINDOW_AUTOSIZE); 
	else destroyWindow(WINDOW_MOVEMENT);
	
	showMovementFlag = b;
}

void TrackingDistance::calcModel()
{
	Mat dest = *weightedModel;
	dest = Scalar(0);

	int totalWeight = numFrames;
	for (int i = 0; i < numFrames; i++) {
		int index = (lastIndex - i + numFrames) % numFrames;

		add(dest, lastFrames[index], dest, noArray(), CV_64FC1);
	}

	dest /= totalWeight;

	if (showModelFlag) {
		dest.copyTo(*testModel);
		(*testModel).convertTo(*testModel, CV_16UC1);
		imshow(WINDOW_MODEL, *testModel);
	}
}

void TrackingDistance::calcDifferences()
{
	Mat differences = abs(*currentFrame - *weightedModel);

	*movingMask = Scalar(0);
	*staticMask = Scalar(0);

	printf("Threshold: input type >%d<, output type >%d<\n", differences.type(), (*movingMask).type());
	threshold(differences, *movingMask, PIXEL_MIN_DIFF, 1, THRESH_BINARY);
	threshold(differences, *staticMask, PIXEL_MAX_DIFF, 1, THRESH_BINARY_INV);
	*movingMask = *movingMask & *staticMask;
	
	Mat erosion = getStructuringElement(MORPH_RECT,Size(EROSION_SIZE, EROSION_SIZE));
	erode(*movingMask, *movingMask, erosion);

	if (showChangesFlag) {
		(*movingMask).copyTo(*testMovement);
		imshow(WINDOW_CHANGES, *testMovement); //CV_64FC1
	}
}

void TrackingDistance::calcNewDists()
{
	*tmp1 = Scalar(0);
	*tmp2 = Scalar(0);
	threshold(*currentFrame, *tmp1, MIN_DISTANCE, 1, THRESH_BINARY);
	threshold(*currentFrame, *tmp2, MAX_DISTANCE, 1, THRESH_BINARY_INV);

	*staticMask = *tmp1 & *tmp2 & *movingMask;
	(*staticMask).convertTo(*staticMask, CV_8U);

	double min, max;	
	minMaxIdx(*currentFrame, &min, &max, 0, 0, *staticMask);

	if (showMovementFlag) {
		*tmp1 = Scalar(0);
		(*currentFrame).copyTo(*tmp1, *staticMask);

		imshow(WINDOW_MOVEMENT, *tmp1);
	}

	if (max == 0) max = maxDist;

	lastMaxs[distLastIndex] = (int) max;

	maxDist = 0;
	
	for (unsigned int i = 0; i < DIST_INERTIA; i++) {
		maxDist += lastMaxs[i];
	}

	maxDist /= DIST_INERTIA;
	if (maxDist < MIN_DISTANCE + MIN_RANGE) {
		maxDist = MIN_DISTANCE + MIN_RANGE;
		lastMaxs[distLastIndex] = maxDist;
	}

	distLastIndex = (distLastIndex + 1) % DIST_INERTIA;

	*tmp2 = Scalar(0);
	threshold(*currentFrame, *tmp2, minDist, 1, THRESH_BINARY);
	*viewMask = Scalar(0);
	threshold(*currentFrame, *viewMask, maxDist + VIEW_ACCURACY, 1, THRESH_BINARY_INV);
	*viewMask = *viewMask & *tmp2;
	(*viewMask).convertTo(*viewMask, CV_8U);

	printf("New Min: >%d<, Max: >%d<\n", minDist, maxDist);
}

///////////////////////////////////////////////////////////////////
//			BodyClassification
///////////////////////////////////////////////////////////////////

BodyClassification::BodyClassification()
{
	skeleton = new Mat(ROWS, COLS, CV_8U, Scalar(0));
	border = new Mat(ROWS, COLS, CV_8U, Scalar(0));
}

BodyClassification::~BodyClassification()
{
	if (showSkeletonFlag) destroyWindow(WINDOW_SKELETON);
	delete image, skeleton, border;
}


void BodyClassification::processFrame(Mat & img)
{
	img.copyTo(*image);
	skeletonize(*image);

}

void ergonomics::BodyClassification::showSkeleton(bool b)
{
	if (showSkeletonFlag == b) return;

	if (b) {
		namedWindow(WINDOW_SKELETON, CV_WINDOW_AUTOSIZE);
	}
	else {
		destroyWindow(WINDOW_SKELETON);
	}

	showSkeletonFlag = b;
}

void BodyClassification::release()
{
	puts("Release BodyClassification");
	delete _instance;
	_instance = NULL;
}

void BodyClassification::skeletonize(Mat& img)
{
	puts("skeletonize image");
	Mat temp;
	Mat eroded;

	Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));

	bool done = false;
	do
	{
		erode(img, eroded, element);
		dilate(eroded, temp, element); // temp = open(img)
		subtract(img, temp, temp);
		bitwise_or(*skeleton, temp, *skeleton);
		eroded.copyTo(img);

		done = (cv::countNonZero(img) == 0);
	} while (!done);

	if (showSkeletonFlag) {
		imshow(WINDOW_SKELETON, *skeleton);
	}
}

BodyParts DrawSkeleton::processBinaryImage(Mat & img)
{
	BodyParts bodyParts;

	if (img.type() != CV_8U) {
		puts("Image is not binary! -> RETURN");
		return bodyParts;
	}

	Mat tmpImg;
	img.convertTo(tmpImg, CV_8U);

	Mat verticalIntensity(1, img.cols, CV_64F), verticalIntensityAveraged(1, img.cols, CV_64F);
	int bellyHorizontal;
	int neckHorizontal, topHeadHorizontal, centerHeadHorizontal;
	int leftHeadVertical, rightHeadVertical;
	int shoulderHorizontal, leftShoulderVertical, rightShoulderVertical;
	Point_<int> cervicalSpineTop, cervicalSpineBottom, sternum, shoulderLeft, shoulderRight;

	int pos, tmp;

	// recude image to columnvector with number of pixels
	reduce(tmpImg, verticalIntensity, 1, CV_REDUCE_SUM, CV_64F);
	// average vertical Intensity
	int kernel[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	filter2D(verticalIntensity, verticalIntensityAveraged, CV_8U, Mat(9, 1, CV_8U, kernel));

	puts("Step1");

	// bellyLine is where max value of vertical intensity is
	minMaxIdx(verticalIntensityAveraged, NULL, NULL, NULL, &bellyHorizontal);

	Mat upperhalf = verticalIntensity;

	int from = upperhalf.rows / 10;
	
	findFirst(upperhalf, &pos, &tmp, &greaterFifteen);
	from += pos;

	upperhalf.forEach<uchar>([from, bellyHorizontal](uchar pixel, const int* position) {
		if (*position <= from || *position >= bellyHorizontal) {
			pixel = 0;
		}
	});

	puts("Step2");

	Mat neck = Mat::zeros(upperhalf.size(), upperhalf.type());
	int kernel2[] = {-1, 1};
	filter2D(upperhalf, neck, CV_8U, Mat(2, 1, CV_8U, kernel2));

	pos = tmp = 0;
	findFirst(neck, &pos, &tmp, [](uchar* number) {
		if (*number > 0) {
			*number = 0;
			return true;
		}
		return false;
	});

	float kernel3[]{ 1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5 };
	filter2D(neck, neck, CV_8U, Mat(5, 1, CV_32F, kernel3));
	minMaxIdx(neck, NULL, NULL, NULL, &neckHorizontal);

	puts("Step3");

	pos = tmp = 0;
	findFirst(verticalIntensity, &pos, &tmp, &greaterZero);

	topHeadHorizontal = pos;
	centerHeadHorizontal = topHeadHorizontal + (neckHorizontal - topHeadHorizontal) / 2;

	Mat headWidth, headHorizontalDistribution(1, img.cols, CV_64F);
	tmpImg.convertTo(headWidth, CV_64F);

	headWidth.forEach<uchar>([centerHeadHorizontal, topHeadHorizontal](uchar pixel, const int* position) {
		if (*position > centerHeadHorizontal || *position < topHeadHorizontal) {
			pixel = 0;
		}
	});

	reduce(headWidth, headHorizontalDistribution, 1, CV_REDUCE_SUM);
	
	puts("Step4");

	findFirst(headHorizontalDistribution, &leftHeadVertical, &tmp, &greaterZero);
	findLast(headHorizontalDistribution, &rightHeadVertical, &tmp, &greaterZero);

	shoulderHorizontal = neckHorizontal + (bellyHorizontal - neckHorizontal) / 2;

	int shoulderLeftTemp, shoulderRightTemp;
	int neckLeft, neckRight;

	findFirst(tmpImg.row(shoulderHorizontal), &shoulderLeftTemp, NULL, &greaterZero);
	findLast(tmpImg.col(shoulderHorizontal), &shoulderRightTemp, NULL, &greaterZero);
	findFirst(tmpImg.row(neckHorizontal), &neckLeft, NULL, &greaterZero);
	findLast(tmpImg.col(neckHorizontal), &neckRight, NULL, &greaterZero);

	puts("Step5");

	leftShoulderVertical = neckLeft - (neckLeft - shoulderLeftTemp) / 2;
	rightShoulderVertical = neckRight + (shoulderRightTemp - neckRight) / 2;

	int centerHead = leftHeadVertical + (rightHeadVertical - leftHeadVertical) / 2;
	bodyParts.cervicalSpineTop.row = centerHeadHorizontal;
	bodyParts.cervicalSpineTop.col = centerHead;
	cervicalSpineTop = Point_<int>(centerHeadHorizontal, centerHead);

	double tmp1 = (leftShoulderVertical + leftHeadVertical) / 2;
	double tmp2 = (rightShoulderVertical - leftShoulderVertical) / 2 + (rightHeadVertical - leftHeadVertical) / 2;
	int centerSternum = (int) (tmp1 + tmp2 / 2);

	bodyParts.sternum.row = shoulderHorizontal;
	bodyParts.sternum.col = centerSternum;
	sternum = Point_<int>(shoulderHorizontal, centerSternum);

	int centerNeck = leftShoulderVertical + (rightShoulderVertical - leftShoulderVertical) / 2;
	tmp1 = centerHead + centerSternum - 2 * centerNeck;
	centerNeck += (int) tmp1 / 3;

	puts("Step6");

	cervicalSpineBottom = Point_<int>(neckHorizontal, centerNeck);
	bodyParts.cervicalSpineBottom.row = neckHorizontal;
	bodyParts.cervicalSpineBottom.col = centerNeck;
	
	tmp1 = leftShoulderVertical + (leftHeadVertical - leftShoulderVertical) / 2;
	tmp2 = neckHorizontal + (shoulderHorizontal - neckHorizontal) / 2;

	shoulderLeft = Point_<int>((int) tmp1, (int) tmp2);
	bodyParts.leftShoulder.row = (int) tmp1;
	bodyParts.leftShoulder.col = (int) tmp2;

	tmp1 = rightShoulderVertical + (rightShoulderVertical - rightHeadVertical) / 2;

	shoulderRight = Point_<int>((int) tmp1, (int) tmp2);
	bodyParts.rightShoulder.row = (int) tmp1;
	bodyParts.rightShoulder.col = (int) tmp2;

	return bodyParts;
}

//void ergonomics::DrawSkeleton::release()
//{
//	delete _instance;
//	_instance = NULL;
//}

ergonomics::DrawSkeleton::~DrawSkeleton()
{
}

static void findFirst(Mat img, int * position, int * value, bool (*check)(uchar* value))
{
	if (img.rows != 1 && img.cols != 1) {
		puts("only one dimensional images are allowed to process!");
		printf("firstValueNotEqualZerso() failed for Image size %dx%d\n", img.rows, img.cols);
		return;
	}

	int length = (int) img.rows * img.cols;
	
	uchar val = 0;
	for (int i = 0; i < length; i++) {
		val = img.at<uchar>(i);
		if (check(&val)) {
			*position = i;
			*value = val;
			break;
		}
	}
}

static void findLast(Mat img, int * position, int * value, bool(*check)(uchar *value))
{
	if (img.rows != 1 && img.cols != 1) {
		puts("only one dimensional images are allowed to process!");
		printf("firstValueNotEqualZerso() failed for Image size %dx%d\n", img.rows, img.cols);
		return;
	}

	int length = (int)img.rows * img.cols;

	uchar val = 0;
	for (int i = length - 1; i >= 0; i--) {
		val = img.at<uchar>(i);
		if (check(&val)) {
			*position = i;
			*value = val;
			break;
		}
	}
}

bool greaterZero(uchar* value)
{
	if (*value > 0)
		return true;
	return false;
}

bool greaterFifteen(uchar* value)
{
	if (*value > 15)
		return true;
	return false;
}
