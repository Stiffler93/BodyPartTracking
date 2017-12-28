#ifndef BPT_TREE_SETTINGS
#define BPT_TREE_SETTINGS

#include <string>
#include <fstream>

using namespace std;

namespace tree {

	const static int BPT_NUM_FEATURES = 10;
	const static bool BPT_TRACE = true;
	const static int BPT_NUM_DATASETS = 12;
	const static string BPT_FOLDER_ROOT = "C:\\Users\\Stefan\\Desktop\\Master\\Masterarbeit\\Programme\\BodyPartTracking\\";
	const static string BPT_FOLDER_DATA = "Data\\";
	const static string BPT_FOLDER_TRAINING = "TrainingImages\\StraightPostures\\";
	const static string BPT_FOLDER_CLASSIFIED_IMAGES = "color_processed\\";
	const static string BPT_FOLDER_DEPTH_IMAGES = "depth\\";
	const static string BPT_FOLDER_FEATURES = "features\\";
	const static string BPT_FILE_TREE = "tree_parallel_test.txt";
	const static string BPT_FILE_DEBUG = "debug.txt";
	const static string BPT_FILE_DATASET = "dataset_parallel_test.txt";

	bool isTraceActive();
	int numFeatures();
	int numDatasets();
	string rootFolder();
	string dataFolder();
	string treeFile();
	string debugFile();
	string trainingFolder();
	string datasetFile();
	string classifiedImagesFolder();
	string depthImagesFolder();
}

#endif

