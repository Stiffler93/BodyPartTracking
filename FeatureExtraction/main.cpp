#include <string>
#include <iostream>
#include <vector>
#include <windows.h>
#include <strsafe.h>
#include "opencv2\opencv.hpp"
#include "Features.hpp"
#include "DecTree.h"

using namespace std;
using namespace cv;
using namespace tree;

void pause();
bool checkTrainingDataFolder(string trDataFolder);
vector<string> getFilesInDirectory(string trDataFolder);
vector<string> getUnifiedTrainingData(string trDataFolder);
void show(Mat& colImg, Mat& depImg);

int main(int argc, char** argv) 
{
	if (!checkTrainingDataFolder(trainingFolder())) {
		printf("Exit Program.\n");
		pause();
		return 1;
	}

	vector<string> trData = getUnifiedTrainingData(trainingFolder());

	printf("Found following Training Data:\n");
	for (string data : trData)
		printf("%s\n", data.c_str());

	if (trData.empty()) {
		printf("No Training Data found!\n");
		return 2;
	}

	namedWindow("ColImage");
	namedWindow("Reference");

	string colStr = classifiedImagesFolder() + trData[0];
	Mat colImg = imread(colStr);

	string depStr = depthImagesFolder() + trData[0];
	Mat depImg = imread(depStr, CV_LOAD_IMAGE_ANYDEPTH);

	Mat subject = Mat(depImg.rows, depImg.cols, depImg.type());
	Mat mask;

	getSubject(depImg, subject);

	Mat feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10;
	feat1.create(depImg.rows, depImg.cols, depImg.type());
	feat2.create(depImg.rows, depImg.cols, depImg.type());
	feat3.create(depImg.rows, depImg.cols, depImg.type());
	feat4.create(depImg.rows, depImg.cols, depImg.type());
	feat5.create(depImg.rows, depImg.cols, depImg.type());
	feat6.create(depImg.rows, depImg.cols, depImg.type());
	feat7.create(depImg.rows, depImg.cols, depImg.type());
	feat8.create(depImg.rows, depImg.cols, depImg.type());
	feat9.create(depImg.rows, depImg.cols, depImg.type());
	feat10.create(depImg.rows, depImg.cols, depImg.type());
	
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

	printf("Classify pixels:\n");
	size_t classified = 0;

	Mat reference;
	reference.create(240, 320, CV_8UC3);
	reference = 0;
	vector<Dataset> datasets;

	for (int row = 0; row < MAX_ROW; row++) {
		for (int col = 0; col < MAX_COL; col++) {
			if (subject.at<ushort>(row, col) == 0)
				continue;

			Vec3b colors = colImg.at<Vec3b>(row, col);
			Dataset set;
			set.outcome = getCategory(colors[2], colors[1], colors[0]);
			
			if (set.outcome == NONE)
				continue;

			Vec3b color;
			if (set.outcome == LEFT_SHOULDER) {
				color = Vec3b(0, 0, 255);
			}
			else if (set.outcome == RIGHT_SHOULDER) {
				color = Vec3b(0, 255, 0);
			}
			else if (set.outcome == NECK) {
				color = Vec3b(255, 255, 0);
			}
			else if (set.outcome == HEAD) {
				color = Vec3b(255, 0, 0);
			}
			else if (set.outcome == STERNUM) {
				color = Vec3b(100, 100, 100);
			}
			else if (set.outcome == OTHER) {
				color = Vec3b(255, 255, 255);
			}

			reference.at<Vec3b>(row, col) = color;

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

			datasets.push_back(set);
			if (++classified % 100 == 0)
				printf("Already classified %zd pixels.\n", classified);
		}
	}

	size_t size = datasets.size();
	printf("Algorithm classified %zd pixels.\n", size);

	size_t written = 0;
	ofstream features(datasetFile());
	for (Dataset set : datasets) {
		features << set.toString() << endl;
		if(++written % 100 == 0)
			printf("Wrote %zd of %zd pixels.\n", written, size);
	}

	int key = -1;
	while (key != 27) {
		imshow("ColImage", colImg);
		imshow("Reference", reference);
		key = waitKey(10);
	}
		
	features.close();

	printf("Wrote ALL pixels successfully.\n");

	printf("Program finished Successfully.\n");
	pause();
	return 0;
}

void pause()
{
	string c;
	cin >> c;
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

	if (stat(classifiedImagesFolder().c_str(), &buf) != 0)
	{
		printf("%s does not exist!\n", classifiedImagesFolder().c_str());
		error = true;
	}

	if (stat(depthImagesFolder().c_str(), &buf) != 0)
	{
		printf("%s does not exist!\n", depthImagesFolder().c_str());
		error = true;
	}

	return !error;
}

vector<string> getFilesInDirectory(string trDataFolder)
{
	vector<string> results;

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
		return vector<string>();
	}

	FindClose(hFind);

	return results;
}

vector<string> getUnifiedTrainingData(string trDataFolder)
{
	vector<string> trdata;
	vector<string> colData = getFilesInDirectory(classifiedImagesFolder());
	vector<string> depData = getFilesInDirectory(depthImagesFolder());

	vector<string>::iterator colIt = colData.begin(), depIt = depData.begin();
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

void show(Mat& colImg, Mat& depImg)
{
	imshow("Color Image", colImg);
	imshow("Depth Image", depImg);
}