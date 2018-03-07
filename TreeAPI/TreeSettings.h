#ifndef BPT_TREE_SETTINGS
#define BPT_TREE_SETTINGS

#include <string>

namespace tree {

	const static int BPT_NUM_FEATURES = 26;
	const static bool BPT_TRACE = false;
	const static int BPT_NUM_RECORDS = 1000000;
	const static bool BPT_PROCESS_REAL_DATA = true;
	
	const static bool BPT_DATASET_WITH_RECORDS = true;
	const static int BPT_DATASET_RECORD_CHARS = 12;
	const static int BPT_DATASET_VALUE_CHARS = 4;
	const static double BPT_DATASET_SUBSET_PROPORTION = 1.0;
	const static int BPT_NUM_RECORDS_IN_BUFFER = 1000;
	const static short BPT_STOP_EVALUATION_LIMIT = 1;
	const static double BPT_STOP_EVALUATION_IMPURITY = 0.03;
	const static double BPT_UNIQUE_VAL_INVALIDATION_RATE = 1.0;
	const static int BPT_KNN_NUM_NEIGHBORS = 5;
	const static std::string BPT_FOLDER_ROOT = "C:\\Users\\Stefan\\Desktop\\Master\\Masterarbeit\\Programme\\BodyPartTracking\\";
	const static std::string BPT_FOLDER_DATA = "Data\\";
	const static std::string BPT_FOLDER_TRAINING = "TrainingImages\\StraightPostures\\"; // SyntheticData\\";
	const static std::string BPT_FOLDER_COLOR_IMAGES = "color\\";
	const static std::string BPT_FOLDER_CLASSIFIED_IMAGES = "color_processed\\";
	const static std::string BPT_FOLDER_DEPTH_IMAGES = "depth\\";
	const static std::string BPT_FOLDER_FEATURES = "features\\";
	const static std::string BPT_FILE_TREE = "performance_tests\\trees\\parallel_1000000.tree";
	const static std::string BPT_FILE_DEBUG = "debug.txt";
	const static std::string BPT_FILE_DATASET = "performance_tests\\dataset_1000000.orig"; 
	const static std::string BPT_FILE_DATASET_ORDERED = "performance_tests\\dataset_1000000.orig_ordered"; 
	const static std::string BPT_FILE_DATASET_MAP = "performance_tests\\dataset_1000000.orig_map"; 
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
	std::string datasetFileOrdered();
	std::string datasetFileMap();
	std::string strainsFile();
	std::string measurementsFile();
	std::string knowledgeFile();
	std::string imagesFolder();
	std::string classifiedImagesFolder();
	std::string depthImagesFolder();
}

#endif

