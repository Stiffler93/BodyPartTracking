#ifndef BPT_TREE_SETTINGS
#define BPT_TREE_SETTINGS

#include <string>

using namespace std;

//const static bool BPT_TRACE;
const static int BPT_NUM_FEATURES;
//const static int BPT_NUM_DATASETS;
//const static string BPT_FOLDER_ROOT;
//const static string BPT_FOLDER_DATA;
//const static string BPT_FOLDER_TRAINING;
//const static string BPT_FOLDER_CLASSIFIED_IMAGES;
//const static string BPT_FOLDER_DEPTH_IMAGES;
//const static string BPT_FOLDER_FEATURES;
//const static string BPT_FILE_TREE;
//const static string BPT_FILE_DEBUG;
//const static string BPT_FILE_DATASET;

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

#endif