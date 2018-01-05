#ifndef BPT_TREE_SETTINGS
#define BPT_TREE_SETTINGS

#include <string>
#include <fstream>

using namespace std;

namespace tree {

	const static int BPT_NUM_FEATURES = 10;
	const static bool BPT_TRACE = true;
	const static int BPT_NUM_DATASETS = 650000;
	const static short BPT_STOP_EVALUATION_LIMIT = 50;
	const static short ONE_METER = 10000;
	const static string BPT_FOLDER_ROOT = "C:\\Users\\Stefan\\Desktop\\Master\\Masterarbeit\\Programme\\BodyPartTracking\\";
	const static string BPT_FOLDER_DATA = "Data\\";
	const static string BPT_FOLDER_TRAINING = "TrainingImages\\StraightPostures\\";
	const static string BPT_FOLDER_CLASSIFIED_IMAGES = "color_processed\\";
	const static string BPT_FOLDER_DEPTH_IMAGES = "depth\\";
	const static string BPT_FOLDER_FEATURES = "features\\";
	const static string BPT_FILE_TREE = "tree_straight_test_sequential.txt"; //"tree_performance_5_parallel.txt"; 
	const static string BPT_FILE_DEBUG = "debug.txt";
	const static string BPT_FILE_DATASET = "dataset_straight_test.txt"; //"dataset_performance_5.txt"; 

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

