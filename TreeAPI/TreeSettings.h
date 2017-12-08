#ifndef BPT_TREE_SETTINGS
#define BPT_TREE_SETTINGS

#include <string>

using namespace std;
const static bool BPT_TRACE = true;
const static int BPT_NUM_FEATURES = 10;
const static int BPT_NUM_DATASETS = 7960;
const static string BPT_FOLDER_ROOT = "C:\\Users\\Stefan\\Desktop\\Master\\Masterarbeit\\Programme\\BodyPartTracking\\";
const static string BPT_FOLDER_DATA = "Data\\";
const static string BPT_FOLDER_TRAINING = "TrainingImages\\TestSamples\\";
const static string BPT_FOLDER_CLASSIFIED_IMAGES = "color_processed\\";
const static string BPT_FOLDER_DEPTH_IMAGES = "depth\\";
const static string BPT_FOLDER_FEATURES = "features\\";
const static string BPT_FILE_TREE = "tree.txt";
const static string BPT_FILE_DEBUG = "debug.txt";
const static string BPT_FILE_DATASET = "dataset.txt";

bool isTraceActive() {
	return BPT_TRACE;
}

int numFeatures() {
	return BPT_NUM_FEATURES;
}

int numDatasets() {
	return BPT_NUM_DATASETS;
}

string rootFolder() {
	return BPT_FOLDER_ROOT;
}

string dataFolder() {
	return rootFolder() + BPT_FOLDER_DATA;
}

string treeFile() {
	return dataFolder() + BPT_FILE_TREE;
}

string debugFile() {
	return dataFolder() + BPT_FILE_DEBUG;
}

string trainingFolder() {
	return dataFolder() + BPT_FOLDER_TRAINING;
}

string datasetFile() {
	return dataFolder() + BPT_FILE_DATASET;
}

string classifiedImagesFolder() {
	return trainingFolder() + BPT_FOLDER_CLASSIFIED_IMAGES;
}

string depthImagesFolder() {
	return trainingFolder() + BPT_FOLDER_DEPTH_IMAGES;
}

#endif