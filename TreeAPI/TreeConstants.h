
#ifndef TREE_CONSTS
#define TREE_CONSTS

#include <string>

const static unsigned short MAX_VAL = 65535;
const static int MAX_ROW = 240;
const static int MAX_COL = 320;
const static int FPS = 30;
const static int DEPTH_IMAGE = 2; //= CV_16U
const static int MIN_DIST = 5000;
const static int NUM_CATEGORIES = 6; // NONE not counted!
const static int NORM_FACTOR = 1000; // max value of pixels in normalized image
const static int ONE_METER = 10000;
const static int NUM_AREAS_OF_INTEREST = 2;
const static int NUM_COUNTERS_PER_AREA = 4;

const static std::string LEFT_SHOULDER = "LeftShoulder";
const static std::string RIGHT_SHOULDER = "RightShoulder";
const static std::string HEAD = "Head";
const static std::string NECK = "Neck";
const static std::string STERNUM = "Sternum";
const static std::string OTHER = "Other";
const static std::string NONE = "None";

#endif