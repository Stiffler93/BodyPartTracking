#ifndef BPT_TREE_SETTINGS
#define BPT_TREE_SETTINGS

#include <string>

namespace tree {

	const static int BPT_NUM_FEATURES = 26;
	const static bool BPT_TRACE = false;
	const static int BPT_NUM_DATASETS = 4169084;
	const static short BPT_STOP_EVALUATION_LIMIT = 50;
	const static double BPT_STOP_EVALUATION_IMPURITY = 0.03;
	const static double BPT_DATASET_SUBSET_PROPORTION = 0.8;
	const static double BPT_UNIQUE_VAL_INVALIDATION_RATE = 1.0;
	const static std::string BPT_FOLDER_ROOT = "C:\\Users\\Stefan\\Desktop\\Master\\Masterarbeit\\Programme\\BodyPartTracking\\";
	const static std::string BPT_FOLDER_DATA = "Data\\";
	const static std::string BPT_FOLDER_TRAINING = "TrainingImages\\StraightPostures\\";
	const static std::string BPT_FOLDER_COLOR_IMAGES = "color\\";
	const static std::string BPT_FOLDER_CLASSIFIED_IMAGES = "color_processed\\";
	const static std::string BPT_FOLDER_DEPTH_IMAGES = "depth\\";
	const static std::string BPT_FOLDER_FEATURES = "features\\";
	const static std::string BPT_FILE_TREE = "tree.txt";
	const static std::string BPT_FILE_DEBUG = "debug.txt";
	const static std::string BPT_FILE_DATASET = "dataset.txt";
	const static std::string BPT_FILE_STRAINS = "strains.txt";
	const static std::string BPT_FILE_MEASUREMENTS = "measurements.txt";
	const static std::string BPT_FILE_KNOWLEDGE = "knowledge.txt";


	bool isTraceActive();
	int numFeatures();
	int numDatasets();
	std::string rootFolder();
	std::string dataFolder();
	std::string treeFile();
	std::string debugFile();
	std::string trainingFolder();
	std::string datasetFile();
	std::string strainsFile();
	std::string measurementsFile();
	std::string knowledgeFile();
	std::string imagesFolder();
	std::string classifiedImagesFolder();
	std::string depthImagesFolder();
}

#endif

