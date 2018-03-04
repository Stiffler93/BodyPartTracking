#ifndef BPT_TREE_SETTINGS
#define BPT_TREE_SETTINGS

#include <string>

namespace tree {

	const static int BPT_NUM_FEATURES = 26;
	const static bool BPT_TRACE = true;
	const static int BPT_NUM_RECORDS = /*20; // */1000;
	// only temporary!!
	const static int BPT_NUM_CATEGORY_1 = 0;
	const static int BPT_NUM_CATEGORY_2 = 0;
	const static int BPT_NUM_CATEGORY_3 = 0;
	const static int BPT_NUM_CATEGORY_4 = 0;
	const static int BPT_NUM_CATEGORY_5 = 0;
	const static int BPT_NUM_CATEGORY_6 = 0;
	
	const static int BPT_DATASET_RECORD_CHARS = 12;
	const static int BPT_DATASET_VALUE_CHARS = 4;
	const static double BPT_DATASET_SUBSET_PROPORTION = 1.0; // 0.7;
	const static int BPT_NUM_RECORDS_IN_BUFFER = 100;
	//const static int BPT_RECORD_SIZE_BYTE = 133;
	const static short BPT_STOP_EVALUATION_LIMIT = 50;
	const static double BPT_STOP_EVALUATION_IMPURITY = 0.03;
	const static double BPT_UNIQUE_VAL_INVALIDATION_RATE = 1.0;
	const static int BPT_KNN_NUM_NEIGHBORS = 5;
	const static std::string BPT_FOLDER_ROOT = "C:\\Users\\Stefan\\Desktop\\Master\\Masterarbeit\\Programme\\BodyPartTracking\\";
	const static std::string BPT_FOLDER_DATA = "Data\\";
	const static std::string BPT_FOLDER_TRAINING = "TrainingImages\\SyntheticData\\";
	const static std::string BPT_FOLDER_COLOR_IMAGES = "color\\";
	const static std::string BPT_FOLDER_CLASSIFIED_IMAGES = "color_processed\\";
	const static std::string BPT_FOLDER_DEPTH_IMAGES = "depth\\";
	const static std::string BPT_FOLDER_FEATURES = "features\\";
	const static std::string BPT_FILE_TREE = "tree.txt";
	const static std::string BPT_FILE_DEBUG = "debug.txt";
	const static std::string BPT_FILE_DATASET = /*"dataset_test.txt";//*/"dataset_synthetic.test";
	const static std::string BPT_FILE_DATASET_ORDERED = /*"dataset_test.txt_ordered"; // */"dataset_synthetic.test_ordered";
	const static std::string BPT_FILE_DATASET_MAP = /*"dataset_test.txt_map"; // */"dataset_synthetic.test_map";
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

