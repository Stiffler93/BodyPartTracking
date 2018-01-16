#include "OpenNI.h"
#include "opencv2\opencv.hpp"
#include "3DWorldTransformations.h"
#include "TreeConstants.h"
#include "TreeSettings.h"
#include "DecisionForest.h"
#include "BodyPartDetector.h"
#include "ErgonomicEvaluation.h"
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <windows.h>
#include <strsafe.h>

using std::string;
using std::map;

std::vector<string> getFilesInDirectory(string trDataFolder);
std::vector<string> getUnifiedTrainingData(string trDataFolder);
void readStrains(std::ifstream strainFile, map<int, ergonomics::Strains>& strainMap);

int main(int argc, char** argv) {
	// needs to be done for WorldTransformation!!!
	openni::Status statOpenNI;
	printf("OpenNI initialization...\n");
	statOpenNI = openni::OpenNI::initialize();
	if (statOpenNI != openni::Status::STATUS_OK) {
		puts("OpenNI initialization failed!");
		return 1;
	}

	puts("Asus Xtion Pro initialization...");
	openni::Device device;
	if (device.open(openni::ANY_DEVICE) != 0)
	{
		puts("Device not found !");
		puts("Abort test");
		openni::OpenNI::shutdown();
		return 1;
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
	// needs to be done for WorldTransformation!!!



	printf("Start Creating knowledge.\n");

	std::vector<string> images = getUnifiedTrainingData(tree::trainingFolder());
	
	for (string image : images) {
		string depStr = tree::depthImagesFolder() + image;
		cv::Mat depImg = cv::imread(depStr, CV_LOAD_IMAGE_ANYDEPTH);

		double min, max;
		cv::minMaxIdx(depImg, &min, &max);

		printf("Min = %lf, Max = %lf\n", min, max);

		std::vector<string> trees;
		trees.push_back(tree::dataFolder() + "tree_straight_273_26_sub_1.txt");
		trees.push_back(tree::dataFolder() + "tree_straight_273_26_sub_2.txt");
		trees.push_back(tree::dataFolder() + "tree_straight_273_26_sub_3.txt");
		tree::DecisionForest decForest(trees);

		tree::BodyPartDetector bpDetector = tree::BodyPartDetector(decForest);

		cv::Mat classifiedMat;
		classifiedMat.create(MAX_ROW, MAX_COL, CV_8UC3);

		tree::Dataset** featureMatrix = new tree::Dataset*[MAX_ROW];
		for (int i = 0; i < MAX_ROW; i++)
			featureMatrix[i] = new tree::Dataset[MAX_COL];

		cv::Mat img;
		tree::BodyPartLocations locs = bpDetector.getBodyPartLocations(img);
		ergonomics::ErgonomicEvaluation::getInstance().classify(locs);
	}
	
	

	return 0;
}

std::vector<string> getFilesInDirectory(string trDataFolder)
{
	std::vector<string> results;

	WIN32_FIND_DATA ffd;
	TCHAR szDir[MAX_PATH];
	size_t length_of_arg;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	DWORD dwError = 0;

	StringCchLength(trDataFolder.c_str(), MAX_PATH, &length_of_arg);

	if (length_of_arg > (MAX_PATH - 3))
	{
		printf("Path is too long!\n");
		return results;
	}

	StringCchCopy(szDir, MAX_PATH, trDataFolder.c_str());
	StringCchCat(szDir, MAX_PATH, TEXT("\\*"));

	hFind = FindFirstFile(szDir, &ffd);

	if (INVALID_HANDLE_VALUE == hFind)
	{
		printf("Error occured in getTrainingData() -> INVALID_HANDLE_VALUE\n");
		return results;
	}

	do
	{
		if (!(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
		{
			results.push_back(string(ffd.cFileName));
		}
	} while (FindNextFile(hFind, &ffd) != 0);

	dwError = GetLastError();
	if (dwError != ERROR_NO_MORE_FILES)
	{
		printf("Error raised!\n");
		return std::vector<string>();
	}

	FindClose(hFind);

	return results;
}

std::vector<string> getUnifiedTrainingData(string trDataFolder)
{
	std::vector<string> trdata;
	std::vector<string> colData = getFilesInDirectory(tree::classifiedImagesFolder());
	std::vector<string> depData = getFilesInDirectory(tree::depthImagesFolder());

	std::vector<string>::iterator colIt = colData.begin(), depIt = depData.begin();
	while (colIt != colData.end() && depIt != depData.end()) {
		if (*colIt == *depIt) {
			trdata.push_back(*colIt);
			++colIt;
			++depIt;
		}
		else if (*colIt < *depIt) ++colIt;
		else ++depIt;
	}

	return trdata;
}

void readStrains(std::ifstream strainFile, map<int, ergonomics::Strains>& strainMap) {
	string line;

	while (strainFile >> line) {
		int posSem = line.find(';', 0);
		int image = atoi(line.substr(0, posSem - 1).c_str());
		ergonomics::Strains str;
		str.ofString(line.substr(posSem + 1, line.length() - posSem - 2));
		strainMap.insert(std::pair<int, ergonomics::Strains>(image, str));
	}
}