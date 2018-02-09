#include "DecTree.h"
#include "TreeSettings.h"
#include <fstream>
#include <string>
#include "TreeConstants.h"
#include "TreeUtils.h"
#include "opencv2\opencv.hpp"
#include "ImageRecorder.h"
#include "Features.hpp"
#include "Tests.h"
#include "DecisionForest.h"
#include "ErgonomicEvaluation.h"
#include "3DWorldTransformations.h"

using std::string;
using std::to_string;

void classification(cv::Mat& image, cv::Mat& classifiedMat, tree::Record**& featureMatrix, tree::DecisionForest& decForest, tree::BodyPartDetector& bpDetector);
void realTest(tree::DecisionForest& decForest);
void testDecForestWithTrainingData(tree::DecisionForest& decForest);

int main(int argc, char** argv) 
{
	printf("Start Decision Tree Test: \n");
	std::vector<string> trees;
	trees.push_back(tree::dataFolder() + "tree_straight_273_26_sub_1.txt");
	trees.push_back(tree::dataFolder() + "tree_straight_273_26_sub_2.txt");
	trees.push_back(tree::dataFolder() + "tree_straight_273_26_sub_3.txt");
	//trees.push_back(tree::treeFile());
	tree::DecisionForest decForest(trees);

	realTest(decForest);
	//testDecForestWithTrainingData(decForest);

	string s;
	std::cin >> s;

	return 0;
}

void realTest(tree::DecisionForest& decForest)
{
	openni::Status statOpenNI;
	printf("OpenNI initialization...\n");
	statOpenNI = openni::OpenNI::initialize();
	if (statOpenNI != openni::Status::STATUS_OK) {
		puts("OpenNI initialization failed!");
		return;
	}

	puts("Asus Xtion Pro initialization...");
	openni::Device device;
	if (device.open(openni::ANY_DEVICE) != 0)
	{
		puts("Device not found !");
		puts("Abort test");
		openni::OpenNI::shutdown();
		return;
	}
	puts("Asus Xtion Pro opened");

	openni::VideoStream depth;
	depth.create(device, openni::SENSOR_DEPTH);
	depth.start();

	world::CoordinateTransformator* transformator = world::CoordinateTransformator::getInstance();
	transformator->init(&depth);

	openni::VideoMode paramvideo;
	paramvideo.setResolution(MAX_COL, MAX_ROW);
	paramvideo.setFps(FPS);
	paramvideo.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_100_UM);

	depth.setVideoMode(paramvideo);

	openni::VideoStream** stream = new openni::VideoStream*[1];
	stream[0] = &depth;
	puts("Kinect initialization completed");

	puts("Continue? (y/n)");
	string s;
	std::cin >> s;

	if (s != "y") {
		puts("Shutdown OpenNI");
		depth.stop();
		depth.destroy();
		device.close();
		openni::OpenNI::shutdown();
		return;
	}

	util::ImageRecorder recorder(device, stream, decForest, &classification);
	recorder.run();

	printf("Release OpenNI resources.");
	depth.stop();
	depth.destroy();
	printf(".");
	device.close();
	printf(".");
	openni::OpenNI::shutdown();
	printf("done.\n");
}

void classification(cv::Mat& image, cv::Mat& classifiedMat, tree::Record**& featureMatrix, tree::DecisionForest& decForest, tree::BodyPartDetector& bpDetector) {
	classifiedMat = 0;
	
	tree::BodyPartLocations locs = bpDetector.getBodyPartLocations(image);
	//ergonomics::ErgonomicEvaluation::getInstance().classify(locs);
	ergonomics::ErgonomicEvaluation::getInstance().process(locs);

	//printf("BPT Status = %d\r", bpDetector.state());
	//printf("Head<%d|%d>, Neck<%d|%d>, LShould<%d|%d>, RShould<%d|%d>, Sternum<%d|%d>\r", locs.locs[LOC_HEAD].col, locs.locs[LOC_HEAD].row,
	//	locs.locs[LOC_NECK].col, locs.locs[LOC_NECK].row, locs.locs[LOC_L_SHOULDER].col, locs.locs[LOC_L_SHOULDER].row, locs.locs[LOC_R_SHOULDER].col, 
	//	locs.locs[LOC_R_SHOULDER].row, locs.locs[LOC_STERNUM].col, locs.locs[LOC_STERNUM].row);

	double min, max;
	cv::minMaxIdx(image, &min, &max);

	cv::Mat norm;
	norm.create(image.size(), image.type());
	image.copyTo(norm);

	norm -= min;
	norm = norm / max * 255;
	norm.convertTo(norm, CV_8U);

	cv::Mat norm3ch;
	std::vector<cv::Mat> normArr = { norm, norm, norm };
	cv::merge(normArr, norm3ch);
	
	cv::circle(norm3ch, cv::Point(locs.locs[LOC_HEAD].col, locs.locs[LOC_HEAD].row), 5, cv::Scalar(getBGR(HEAD)));
	cv::circle(norm3ch, cv::Point(locs.locs[LOC_NECK].col, locs.locs[LOC_NECK].row), 5, cv::Scalar(getBGR(NECK)));
	cv::circle(norm3ch, cv::Point(locs.locs[LOC_L_SHOULDER].col, locs.locs[LOC_L_SHOULDER].row), 5, cv::Scalar(getBGR(LEFT_SHOULDER)));
	cv::circle(norm3ch, cv::Point(locs.locs[LOC_R_SHOULDER].col, locs.locs[LOC_R_SHOULDER].row), 5, cv::Scalar(getBGR(RIGHT_SHOULDER)));
	cv::circle(norm3ch, cv::Point(locs.locs[LOC_STERNUM].col, locs.locs[LOC_STERNUM].row), 5, cv::Scalar(getBGR(STERNUM)));

	cv::imshow("Skeleton Joints", norm3ch);
}

void testDecForestWithTrainingData(tree::DecisionForest& decForest) {
	printf("Start Test with own Training Data: \n");
	string dataset = tree::datasetFile();
	std::ifstream features(dataset);

	int total = 0, correctClass = 0, falseClass = 0;

	tree::Record record;
	while(getNextRecord(features, record)) {
		total++;

		string outcome = decForest.classify(record);
		if (outcome == record.outcome) {
			correctClass++;
		}
		else {
			falseClass++;
		}
	}

	features.close();

	std::printf("\n\nTest finished.\n");
	std::printf("Tested a total of %d Datasets.\n", total);
	std::printf("Correct: >%d<, False: >%d<\n", correctClass, falseClass);
	std::printf("Correctly classified %lf percent.\n", (float)correctClass / (float)total * 100);
}
