#include "opencv2\opencv.hpp"
#include "DecTree.h"
#include "TreeConstants.h"
#include "TreeSettings.h"
#include "Features.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <windows.h>
#include <strsafe.h>

using std::string;

void pause();
bool checkTrainingDataFolder(string trDataFolder);
std::vector<string> getFilesInDirectory(string trDataFolder);
std::vector<string> getUnifiedTrainingData(string trDataFolder);

int main(int argc, char** argv) 
{
	if (!checkTrainingDataFolder(tree::trainingFolder())) {
		printf("Exit Program.\n");
		pause();
		return 1;
	}

	std::vector<string> trData = getUnifiedTrainingData(tree::trainingFolder());

	printf("Found following Training Data:\n");
	for (string data : trData)
		printf("%s\n", data.c_str());

	if (trData.empty()) {
		printf("No Training Data found!\n");
		return 2;
	}

	cv::Mat feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10;
	cv::Mat feat11, feat12, feat13, feat14, feat15, feat16, feat17, feat18, feat19, feat20;
	cv::Mat feat21, feat22, feat23, feat24, feat25, feat26;
	cv::Mat horizIntegral, vertIntegral, integral;

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

	std::ofstream features(tree::datasetFile());

	srand(time(0));
	//double randNum;

	size_t recordNum = 0;
	size_t totalNumRecords = 0;

	for (string image : trData) {

		printf("Handle image %s\n", image.c_str());

		string colStr = tree::classifiedImagesFolder() + image;
		cv::Mat colImg = cv::imread(colStr);

		string depStr = tree::depthImagesFolder() + image;
		cv::Mat depImg = cv::imread(depStr, CV_LOAD_IMAGE_ANYDEPTH);

		cv::Mat subject = cv::Mat(MAX_ROW, MAX_COL, DEPTH_IMAGE);
		
		int imgNumber = atoi(image.substr(0, image.find_first_of('.')).c_str());
		getSubject(depImg, subject);
		
		if (!tree::BPT_PROCESS_REAL_DATA || (imgNumber < 49 || imgNumber > 146)) {
			subject = cv::Mat(MAX_ROW, MAX_COL, DEPTH_IMAGE);
			getSubject(depImg, subject);
		}
		else {
			subject = depImg;
			continue;
		}

		//int key = 0;
		//while (key != 27) {
		//	cv::imshow("ColImg", colImg);
		//	cv::imshow("DepImg", depImg);
		//	cv::imshow("Subject", subject);
		//	key = cv::waitKey(50);
		//}

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

		size_t numDatasets = 0;
		tree::Record set;
		cv::Vec3b colors;

		for (int row = 0; row < MAX_ROW; row++) {
			for (int col = 0; col < MAX_COL; col++) {
				if (subject.at<ushort>(row, col) == 0) {
					continue;
				}

				colors = colImg.at<cv::Vec3b>(row, col);
				set.outcome = getCategory(colors[2], colors[1], colors[0]);

				if (set.outcome == NONE) {
					continue;
				}

				//// reduce pixels by half
				//randNum = (double)rand() / (double)RAND_MAX;
				//if (randNum >= 0.5) {
				//	continue;
				//}

				//// OTHER pixels are relatively the most pixels, so reduce them further
				//if (set.outcome == OTHER) {
				//	randNum = (double)rand() / (double)RAND_MAX;
				//	if (randNum >= 0.80) {
				//		continue;
				//	}
				//}

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

				features.width(12);
				features.fill('0');
				features << recordNum++ << " " << set.toString() << std::endl;
				numDatasets++;
			}
		}

		totalNumRecords += numDatasets;
		printf("Algorithm classified %zd pixels for Image %s.\n", numDatasets, image.c_str());
		printf("\nFinished image %s.\n", image.c_str());
	}

	features.close();

	printf("Classified all images successfully.\n");
	printf("Program finished Successfully.\n");

	pause();
	return 0;
}

void pause()
{
	string c;
	std::cin >> c;
}

bool checkTrainingDataFolder(string trDataFolder)
{
	struct stat buf;

	if (stat(trDataFolder.c_str(), &buf) != 0)
	{
		printf("%s does not exist!\n", trDataFolder.c_str());
		return false;
	}

	bool error = false;

	if (stat(tree::classifiedImagesFolder().c_str(), &buf) != 0)
	{
		printf("%s does not exist!\n", tree::classifiedImagesFolder().c_str());
		error = true;
	}

	if (stat(tree::depthImagesFolder().c_str(), &buf) != 0)
	{
		printf("%s does not exist!\n", tree::depthImagesFolder().c_str());
		error = true;
	}

	return !error;
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
