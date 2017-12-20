#include "TreeSettings.h"

using namespace tree;

bool tree::isTraceActive() {
	return BPT_TRACE;
}

int tree::numFeatures() {
	return BPT_NUM_FEATURES;
}

int tree::numDatasets() {
	return BPT_NUM_DATASETS;
}

string tree::rootFolder() {
	return BPT_FOLDER_ROOT;
}

string tree::dataFolder() {
	return rootFolder() + BPT_FOLDER_DATA;
}

string tree::treeFile() {
	return dataFolder() + BPT_FILE_TREE;
}

string tree::debugFile() {
	return dataFolder() + BPT_FILE_DEBUG;
}

string tree::trainingFolder() {
	return dataFolder() + BPT_FOLDER_TRAINING;
}

string tree::datasetFile() {
	return dataFolder() + BPT_FILE_DATASET;
}

string tree::classifiedImagesFolder() {
	return trainingFolder() + BPT_FOLDER_CLASSIFIED_IMAGES;
}

string tree::depthImagesFolder() {
	return trainingFolder() + BPT_FOLDER_DEPTH_IMAGES;
}