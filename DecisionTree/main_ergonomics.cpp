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
#include <iostream>

using std::string;
using std::map;

std::vector<string> getFilesInDirectory(string trDataFolder);
std::vector<string> getUnifiedTrainingData(string trDataFolder);
void readStrains(std::ifstream& strainFile, map<int, ergonomics::Strains>& strainMap);

int main_ergo(int argc, char** argv) {
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

	map<int,ergonomics::Strains> strains;
	std::ifstream strainFile(tree::strainsFile());
	readStrains(strainFile, strains);
	strainFile.close();

	std::ofstream knowledge(tree::knowledgeFile());

	std::vector<string> images = getUnifiedTrainingData(tree::trainingFolder());
	tree::Record** featureMatrix = new tree::Record*[MAX_ROW];
	for (int i = 0; i < MAX_ROW; i++)
		featureMatrix[i] = new tree::Record[MAX_COL];

	std::vector<string> trees;
	trees.push_back(tree::dataFolder() + "tree_straight_273_26_sub_1.txt");
	trees.push_back(tree::dataFolder() + "tree_straight_273_26_sub_2.txt");
	trees.push_back(tree::dataFolder() + "tree_straight_273_26_sub_3.txt");
	tree::DecisionForest decForest(trees);

	tree::BodyPartDetector bpDetector = tree::BodyPartDetector(decForest);

	int processed = 0;
	int not_processed = 0;
	
	for (string image : images) {
		string depStr = tree::depthImagesFolder() + image;
		cv::Mat depImg = cv::imread(depStr, CV_LOAD_IMAGE_ANYDEPTH);

		int imgNumber = atoi(image.substr(0, image.find_first_of('.')).c_str());
		bool isSubject = false;
		if (imgNumber >= 49 && imgNumber <= 146) {
			printf("Skip Image %d\n", imgNumber);
			continue;
		}

		map<int, ergonomics::Strains>::iterator it = strains.find(imgNumber);
		if (it == strains.end()) {
			not_processed++;
			printf("Strains for Image<%3d> were not found!\n", imgNumber);
			continue;
		}

		tree::BodyPartLocations locs = bpDetector.getBodyPartLocations(depImg, isSubject);

		ergonomics::Dataset dataset;
		dataset.mm = ergonomics::ErgonomicEvaluation::getInstance().classify(locs);
		dataset.strains = it->second;

		knowledge << dataset.toString() << std::endl;

		processed++;
	}
	
	knowledge.close();

	puts("Finished successfully");
	printf("Processed: %d, Not processed: %d\n", processed, not_processed);

	string strin;
	std::cin >> strin;

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

void readStrains(std::ifstream& strainFile, map<int, ergonomics::Strains>& strainMap) {
	string line;

	while (strainFile >> line) {
		size_t posSem = line.find(';', 0);
		int image = atoi(line.substr(0, posSem).c_str());
		ergonomics::Strains str;
		str.ofString(line.substr(posSem + 1, line.length() - posSem - 1));
		strainMap.insert(std::pair<int, ergonomics::Strains>(image, str));
	}

	printf("Finished readStrains()\n");
}